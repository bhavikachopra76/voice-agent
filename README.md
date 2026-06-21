# Voice Agent (RAG-based)

A RAG (Retrieval-Augmented Generation) based Voice Agent project designed to extract knowledge from PDF documents, store and retrieve chunks using Supabase Vector Database (`pgvector`), query Cerebras LLM for ultra-fast conversational answers, and output responses as synthesized neural audio.

---

## Quick Start: Running the Project

Right now, running **[query.py](file:///c:/Users/Bhavika/Downloads/voice%20agent/query.py)** is how the voice agent runs. 

### How it works:
1. **Reads Query**: Uses a sample test question (e.g., *"What should I do if the unit begins drifting to the left during initial spin-up?"*).
2. **Retrieves Context**: Computes the embedding vector for the question using the local `BAAI/bge-small-en-v1.5` model, then performs a cosine similarity search against the document chunks stored in Supabase.
3. **Generates Answer**: Sends the retrieved context chunks and user question to **Cerebras AI** (via their ultra-fast LLM inference API).
4. **Synthesizes & Plays Voice**: Uses **Edge-TTS** to generate premium neural speech (`en-US-AriaNeural`), saves it to `response.mp3`, and plays it back locally using **Pygame**.

---

## Detailed Step-by-Step Setup

### Step 1: Clone the Repository
Open your terminal (PowerShell, Command Prompt, or bash) and run:
```bash
git clone https://github.com/bhavikachopra76/voice-agent.git
cd voice-agent
```

### Step 2: Install Python & Package Manager (uv)
1. **Python 3.11**: Make sure Python 3.11 is installed on your computer. You can check with `python --version`.
2. **uv (Recommended)**: We use `uv` for lightning-fast workspace environment management.
   * **On Windows (PowerShell)**:
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   * **On macOS/Linux**:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   * **Alternative (pip)**:
     ```bash
     pip install uv
     ```

### Step 3: Set Up the Virtual Environment
Create the virtual environment locked to Python 3.11 and sync all dependencies from [pyproject.toml](file:///c:/Users/Bhavika/Downloads/voice%20agent/pyproject.toml):
```bash
# Force the virtual environment to use Python 3.11
uv venv --python 3.11

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate

# Install all project packages into the environment
uv sync
```

---

## Step 4: Configure Environment Variables (`.env`)

Create your local `.env` file from the provided template:
```bash
# Windows (PowerShell):
Copy-Item .env.example .env
# Windows (Command Prompt):
copy .env.example .env
# macOS/Linux:
cp .env.example .env
```

Open [.env](file:///c:/Users/Bhavika/Downloads/voice%20agent/.env) in your text editor and populate the following values in detail:

```ini
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-or-service-role-key

# LLM APIs
CEREBRAS_API_KEY=csk-your-unique-cerebras-key

# Hugging Face (Optional, useful if downloading protected HF models)
HF_TOKEN=hf_your-hugging-face-token
```

### How to get these keys:

1. **SUPABASE_URL & SUPABASE_KEY**:
   - Go to [Supabase Console](https://database.supabase.com/) and create a free project.
   - Once created, go to **Project Settings** (gear icon in sidebar) -> **API**.
   - Copy the **Project URL** (paste it as `SUPABASE_URL`).
   - Copy the **`anon` `public` API Key** (paste it as `SUPABASE_KEY`).

2. **CEREBRAS_API_KEY**:
   - Go to [Cerebras Developer Portal](https://cloud.cerebras.ai/) and register for a free account.
   - Go to the **API Keys** section and generate a new key.
   - Copy the key (starts with `csk-`) and paste it as `CEREBRAS_API_KEY`.

3. **HF_TOKEN**:
   - Create a free account on [Hugging Face](https://huggingface.co/).
   - Go to **Settings** -> **Access Tokens**.
   - Click **New Token**, name it, give it `Read` permission, and copy it.

---

## Step 5: Database Setup (Supabase pgvector)

Before running the scripts, you must set up the database schema in Supabase to support vector search.

1. Go to your **Supabase Project Dashboard**.
2. Click on the **SQL Editor** in the left sidebar.
3. Paste and run the following SQL script to enable the vector extension, create the document table, and set up the search function:

```sql
-- 1. Enable the pgvector extension to allow vector similarity search
create extension if space vector;

-- 2. Create the document_chunks table
create table document_chunks (
    id bigserial primary key,
    content text not null,
    embedding vector(384) -- 384 dimensions for the BAAI/bge-small-en-v1.5 model
);

-- 3. Create the similarity match function
create or replace function match_documents (
  query_embedding vector(384),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  content text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    document_chunks.id,
    document_chunks.content,
    1 - (document_chunks.embedding <=> query_embedding) as similarity
  from document_chunks
  where 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
  order by document_chunks.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

---

## Step 6: How to Run the Project

### Part 1: Extract, Chunk, and Upload PDF data
1. Place the PDF you want to query in the root project folder.
2. Rename it to `rag_test_document.pdf`.
3. Run the upload script to parse, chunk, embed, and store it in Supabase:
   ```bash
   uv run upload.py
   ```

### Part 2: Query the Agent (RAG + Audio Output)
1. Ensure the upload process finished successfully.
2. Run the main query script:
   ```bash
   uv run query.py
   ```
3. The script will perform a semantic lookup of the test question in Supabase, pass the result to Cerebras AI, and play the response audio directly through your speakers!
