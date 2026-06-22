"""
RAG Pipeline — PDF extraction, sentence-aware chunking, embedding,
vector retrieval, and LLM answer generation.

All document-processing and AI logic lives here.
"""

import os
import re
import logging

import fitz  # PyMuPDF
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from openai import OpenAI

load_dotenv()
log = logging.getLogger("voice-agent")

# ─── Configuration (tuneable via .env) ────────────────────────────────────────

SUPABASE_URL     = os.getenv("SUPABASE_URL")
SUPABASE_KEY     = os.getenv("SUPABASE_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "150"))
MATCH_COUNT      = int(os.getenv("MATCH_COUNT", "4"))
MATCH_THRESHOLD  = float(os.getenv("MATCH_THRESHOLD", "0.50"))
LLM_MODEL        = os.getenv("LLM_MODEL", "zai-glm-4.7")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ─── Global clients (set once by init()) ──────────────────────────────────────

supabase:    Client | None              = None
llm_client:  OpenAI | None              = None
embedder:    SentenceTransformer | None  = None


def init():
    """Initialise Supabase, Cerebras, and the embedding model.  Call once."""
    global supabase, llm_client, embedder

    missing = [v for v, n in [
        (SUPABASE_URL, "SUPABASE_URL"),
        (SUPABASE_KEY, "SUPABASE_KEY"),
        (CEREBRAS_API_KEY, "CEREBRAS_API_KEY"),
    ] if not v]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")

    supabase   = create_client(SUPABASE_URL, SUPABASE_KEY)
    llm_client = OpenAI(base_url="https://api.cerebras.ai/v1", api_key=CEREBRAS_API_KEY)

    log.info("Loading embedding model '%s' …", EMBEDDING_MODEL)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    log.info("Embedding model ready.")


# ═══════════════════════════════════════════════════════════════════════════════
# PDF Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf(pdf_bytes: bytes, filename: str) -> list[dict]:
    """
    Extract text from a PDF.

    Returns a list of page dicts:
        [{"page": 1, "text": "…", "source": "manual.pdf"}, …]
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        text = doc.load_page(i).get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text, "source": filename})
    doc.close()
    log.info("Extracted %d non-empty pages from '%s'", len(pages), filename)
    return pages


# ═══════════════════════════════════════════════════════════════════════════════
# Sentence-Aware Chunking
# ═══════════════════════════════════════════════════════════════════════════════

# Split on sentence-ending punctuation followed by whitespace + uppercase letter.
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_sentences(text: str) -> list[str]:
    """Regex sentence splitter (lightweight — no nltk/spacy needed)."""
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def chunk_document(pages: list[dict]) -> list[dict]:
    """
    Sentence-aware chunking that flows across page boundaries.

    Each chunk carries metadata:
        {"content", "source", "page", "chunk_index"}
    """
    if not pages:
        return []

    source = pages[0]["source"]

    # Flatten all pages into (sentence, page_number) pairs
    all_sents:  list[str] = []
    sent_pages: list[int] = []
    for p in pages:
        sents = _split_sentences(p["text"])
        all_sents.extend(sents)
        sent_pages.extend([p["page"]] * len(sents))

    chunks: list[dict] = []
    idx = 0
    cur_sents:  list[str] = []
    cur_pages:  list[int] = []
    cur_len = 0

    for i, sent in enumerate(all_sents):
        slen = len(sent)

        # Flush when the next sentence would exceed the chunk budget
        if cur_len + slen > CHUNK_SIZE and cur_sents:
            chunks.append({
                "content":     " ".join(cur_sents),
                "source":      source,
                "page":        cur_pages[0],
                "chunk_index": idx,
            })
            idx += 1

            # Keep trailing sentences as overlap (sentence-aligned, not char-based)
            ov_sents, ov_pages, ov_len = [], [], 0
            for j in range(len(cur_sents) - 1, -1, -1):
                if ov_len + len(cur_sents[j]) > CHUNK_OVERLAP:
                    break
                ov_sents.insert(0, cur_sents[j])
                ov_pages.insert(0, cur_pages[j])
                ov_len += len(cur_sents[j])
            cur_sents, cur_pages, cur_len = ov_sents, ov_pages, ov_len

        cur_sents.append(sent)
        cur_pages.append(sent_pages[i])
        cur_len += slen

    # Flush remainder
    if cur_sents:
        chunks.append({
            "content":     " ".join(cur_sents),
            "source":      source,
            "page":        cur_pages[0],
            "chunk_index": idx,
        })

    log.info("Created %d chunks from '%s'", len(chunks), source)
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding & Upload
# ═══════════════════════════════════════════════════════════════════════════════

def embed_and_upload(chunks: list[dict]) -> int:
    """Batch-embed all chunks and insert into Supabase. Returns count uploaded."""
    if not chunks:
        return 0

    texts = [c["content"] for c in chunks]
    log.info("Batch-embedding %d chunks …", len(texts))
    vectors = embedder.encode(texts, show_progress_bar=True).tolist()

    BATCH = 50
    uploaded = 0
    for i in range(0, len(chunks), BATCH):
        rows = [
            {
                "content":         chunks[j]["content"],
                "embedding":       vectors[j],
                "source_filename": chunks[j]["source"],
                "page_number":     chunks[j]["page"],
                "chunk_index":     chunks[j]["chunk_index"],
            }
            for j in range(i, min(i + BATCH, len(chunks)))
        ]
        supabase.table("document_chunks").insert(rows).execute()
        uploaded += len(rows)
        log.info("Uploaded %d / %d chunks", uploaded, len(chunks))

    return uploaded


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """
    Semantic search via Supabase pgvector.

    Returns [{id, content, similarity}, …] sorted by relevance.
    """
    vec = embedder.encode(query).tolist()
    res = supabase.rpc("match_documents", {
        "query_embedding": vec,
        "match_threshold": MATCH_THRESHOLD,
        "match_count":     top_k or MATCH_COUNT,
    }).execute()
    return res.data


# ═══════════════════════════════════════════════════════════════════════════════
# Answer Generation
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a highly empathetic, friendly, and deeply human-like voice assistant.
Your goal is to help your user understand their documents while sounding exactly
like a helpful, knowledgeable human colleague having a real conversation.

Rules you must follow:

- Speak in a very natural, relaxed, and conversational tone. Use conversational
  fillers naturally (e.g., "Well,", "You know,", "Ah,", or "Hmm") where appropriate,
  but do not overuse them or begin every response the same way. Vary your openings
  so your speech feels natural and spontaneous.

- Actively display empathy and a helpful attitude. Avoid sounding sterile or purely factual.

- First, check the provided document context to find the best possible answer.
  Always prioritize the document over your own knowledge.

- Never narrate the act of searching, looking, finding, or reading. Do not say things like
  "I see that," "I found that," "looking at this," "I have looked through," "the manual says,"
  "the document mentions," "the instructions say," "let me see," or "it says here."
  Speak as if the information simply comes to mind, the way a knowledgeable colleague
  naturally answers from memory rather than reporting back from a search.

- If the user explicitly asks about the document itself (for example "Is this mentioned anywhere?"
  or "Does the manual cover this?"), it is perfectly natural to answer in terms of what the
  document covers. Even then, never describe yourself as searching or reading it.

- If the document does not contain an answer to the user's question, say so plainly and
  naturally without referencing any search process. For example:
  "Hmm, that's not something covered here."
  or
  "I don't actually know that one off the top of my head."

  Do not invent an answer.

  Tailor your follow-up to the type of question instead of always offering an explanation.

  - If it is a concept, offer to explain it from general knowledge.
  - If it is a factual question, offer to answer it from general knowledge.
  - If it is a procedure, offer general troubleshooting or procedural guidance.
  - If it is a specification, measurement, or value, explain that the document does not provide it
    and offer relevant background only if it would genuinely help.

- When the document DOES answer the question, first answer directly using the document.

  Then naturally expand the explanation like a helpful colleague would.

  However, never invent technical mechanisms, engineering details, specifications,
  causes, internal processes, design decisions, or reasoning that are not supported
  by the document.

  If the document explains WHAT something does but not WHY it works, do not speculate.
  If the document provides only part of the explanation, do not fill in the missing parts
  with assumptions.

- You may use general knowledge only to provide high-level background or make the answer
  easier to understand after giving the document-backed answer.

  Make it clear when you are switching from information supported by the document
  to general background knowledge. General knowledge should clarify the topic—not
  replace the document or introduce unsupported technical claims.

- If the user asks "why" and the document does not explain the reason,
  simply say that the reason is not explained here rather than inventing one.
  If helpful, you may then provide clearly separated general background.

- If something is strongly implied by the document but never explicitly stated,
  avoid presenting it as a confirmed fact.
  Prefer wording like:
  "It appears..."
  "The document suggests..."
  "It isn't stated directly..."
  instead of expressing certainty.

- Never infer relationships between products, companies, organizations, or people
  unless they are explicitly stated in the document.

  This includes manufacturers, designers, developers, creators, owners, authors,
  CEOs, inventors, parent companies, subsidiaries, or similar relationships.

  A product name, company branding, document title, logo, company mention,
  or surrounding context is NOT sufficient evidence to establish these relationships.

  If the relationship seems implied but is not explicitly stated, clearly say that
  the document suggests it but does not confirm it.

- Correct false assumptions politely.
  If the user's question is based on an incorrect premise, gently explain the correct
  information rather than answering the incorrect assumption.

- Give longer, comprehensive, and detailed answers whenever the document supports them.
  Prefer helping the user truly understand rather than simply repeating information.

- Never use robotic phrasing like
  "Based on the provided context"
  or
  "As an AI."

- If the user asks about you, your identity, or your capabilities,
  answer honestly that you are an AI assistant while maintaining the same warm,
  conversational personality. Never pretend to be a real person, an employee of the
  company, or someone with firsthand experience.

- Your response will be spoken aloud through text-to-speech.
  Never use markdown, bullet points, numbered lists, emojis, code blocks,
  tables, or special characters.

  Write flowing, conversational paragraphs that sound natural when spoken aloud.
"""

def generate_answer(query: str, chunks: list[dict],
                    history: list[dict] | None = None) -> str:
    """Build a prompt from context + conversation history and call the LLM."""
    context = "\n\n---\n\n".join(c["content"] for c in chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])          # keep last ~3 turns
    messages.append({
        "role": "user",
        "content": f"Document context:\n{context}\n\nQuestion: {query}",
    })

    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════════════
# Document Management
# ═══════════════════════════════════════════════════════════════════════════════

def list_documents() -> list[str]:
    """Return sorted unique filenames stored in the DB."""
    try:
        res = supabase.table("document_chunks").select("source_filename").execute()
        return sorted({r["source_filename"] for r in res.data if r.get("source_filename")})
    except Exception:
        return []


def delete_document(filename: str) -> int:
    """Delete every chunk belonging to *filename*. Returns count deleted."""
    res = (
        supabase.table("document_chunks")
        .delete()
        .eq("source_filename", filename)
        .execute()
    )
    return len(res.data) if res.data else 0
