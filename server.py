"""
FastAPI backend — serves the RAG Voice Agent REST API.

Run with:
    uvicorn server:app --reload --port 8000
"""

import os
import uuid
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager

import edge_tts
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-14s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger("voice-agent")

# ─── Constants ────────────────────────────────────────────────────────────────

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")


# ─── Lifespan — load models once at startup ──────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    rag.init()
    log.info("Voice Agent API is ready.")
    yield


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice Agent API",
    description="RAG-based voice assistant — upload PDFs, ask questions, get audio answers.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated audio files at /audio/<filename>.mp3
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


# ─── Request / Response schemas ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []


class QueryResponse(BaseModel):
    answer: str
    audio_url: str
    sources: list[dict]


# ─── TTS helper ──────────────────────────────────────────────────────────────

async def _synthesize(text: str) -> str:
    """Generate a speech MP3 via Edge-TTS. Returns the filename."""
    fname = f"{uuid.uuid4().hex}.mp3"
    await edge_tts.Communicate(text, TTS_VOICE).save(str(AUDIO_DIR / fname))
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF → extract text → chunk → batch-embed → store in Supabase.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    if len(contents) > 50_000_000:                       # 50 MB hard limit
        raise HTTPException(status_code=400, detail="File exceeds 50 MB limit.")

    try:
        pages    = rag.extract_pdf(contents, file.filename)
        chunks   = rag.chunk_document(pages)
        uploaded = rag.embed_and_upload(chunks)
        return {
            "filename":        file.filename,
            "pages_extracted": len(pages),
            "chunks_created":  len(chunks),
            "chunks_uploaded": uploaded,
        }
    except Exception as exc:
        log.exception("PDF processing failed")
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Ask a question → retrieve relevant chunks → generate LLM answer → TTS.
    """
    start_time = time.time()
    
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        t0 = time.time()
        chunks = rag.retrieve(q)
        t1 = time.time()
        log.info(f"⏱️ Retrieval step took: {t1 - t0:.3f} seconds")

        # Nothing relevant found
        if not chunks:
            fallback = (
                "I couldn't find anything relevant in the uploaded documents. "
                "Could you try rephrasing your question?"
            )
            
            t_fallback_tts_start = time.time()
            fname = await _synthesize(fallback)
            log.info(f"⏱️ TTS synthesis (fallback) step took: {time.time() - t_fallback_tts_start:.3f} seconds")
            log.info(f"⏱️ Total query processing took: {time.time() - start_time:.3f} seconds")
            
            return QueryResponse(answer=fallback, audio_url=f"/audio/{fname}", sources=[])

        t2 = time.time()
        answer = rag.generate_answer(q, chunks, req.history or None)
        t3 = time.time()
        log.info(f"⏱️ Generation step took: {t3 - t2:.3f} seconds")

        t4 = time.time()
        fname  = await _synthesize(answer)
        t5 = time.time()
        log.info(f"⏱️ TTS synthesis step took: {t5 - t4:.3f} seconds")

        sources = [
            {"content": c["content"], "similarity": round(c["similarity"], 3)}
            for c in chunks
        ]
        
        log.info(f"⏱️ Total query processing took: {time.time() - start_time:.3f} seconds")
        return QueryResponse(answer=answer, audio_url=f"/audio/{fname}", sources=sources)

    except Exception as exc:
        log.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")


@app.get("/documents")
async def list_docs():
    """List all unique source filenames stored in the vector DB."""
    return {"documents": rag.list_documents()}


@app.delete("/documents/{filename:path}")
async def delete_doc(filename: str):
    """Delete every chunk belonging to a specific document."""
    try:
        n = rag.delete_document(filename)
        return {"filename": filename, "chunks_deleted": n}
    except Exception as exc:
        log.exception("Delete failed")
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")
