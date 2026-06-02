# Facial Acne AI

A research-grounded facial acne assistant. **Qwen2.5-VL 7B** (via LM Studio) analyzes face photos and writes answers. **RAG** (retrieval-augmented generation) over ingested dermatology papers powers citations via **local embeddings** and the same vision-language model. Everything runs locally through LM Studio — no cloud APIs.

## How it works

```mermaid
flowchart LR
  Photo[Face photo] --> QwenVision[Qwen2.5-VL vision]
  QwenVision --> Obs[Structured observation]
  Question[Your question] --> Retrieve[Chroma + BGE embeddings]
  Obs --> Retrieve
  Retrieve --> Chunks[Research excerpts]
  Chunks --> QwenChat[Qwen2.5-VL via LM Studio]
  Obs --> QwenChat
  Question --> QwenChat
  QwenChat --> Answer[Grounded answer with citations]
```

| Component | Role |
|-----------|------|
| **Qwen2.5-VL 7B** (`LM_STUDIO_CHAT_MODEL`) | Structured photo observations and final answers (vision + chat). |
| **ChromaDB** (`chroma_db/`) | Vector store of chunked papers under `markdown/`. |
| **BGE embeddings** (LM Studio) | Turns questions and paper chunks into vectors for retrieval. |

**Text-only questions** skip the vision step but still use Qwen2.5-VL for answers. **Photos** require the vision model loaded in LM Studio.

---

## Requirements

- **Python 3.10+** (3.11 or 3.12 recommended)
- **[LM Studio](https://lmstudio.ai/)** — local OpenAI-compatible server for embeddings + Qwen2.5-VL
- Disk space for the vision model and embedding model in LM Studio (~5–10 GB total, depending on quant)

The repo already includes converted paper markdown in `markdown/`. You still need to **ingest** them once into `chroma_db/` (see below).

---

## 1. Clone and open the project

```bash
git clone <your-repo-url> Facial_Acne_AI
cd Facial_Acne_AI
```

---

## 2. Create a virtual environment

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade pip, then install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements-rag.txt
```

`requirements-rag.txt` installs everything needed for ingest, CLI, and the web UI (`chromadb`, `flask`, `openai`, etc.).

Optional — only if you need to convert new PDFs with Marker:

```bash
pip install -r requirements.txt   # marker-pdf (heavy)
```

---

## 3. Install and configure LM Studio

### Download LM Studio

1. Go to [https://lmstudio.ai/](https://lmstudio.ai/) and install LM Studio for your OS.
2. Open **Discover** (model search) and download:
   - **`text-embedding-bge-small-en-v1.5`** (or your preferred embed model) — embeddings for RAG
   - **`qwen/qwen2.5-vl-7b`** — vision + chat for photo analysis and answers

Use the exact names shown in LM Studio if they differ slightly (e.g. GGUF variant or quant suffix).

### Run the local server

1. Open the **Local Server** tab (or **Developer** → server, depending on LM Studio version).
2. Load **`qwen/qwen2.5-vl-7b`** as the active chat model and **start the server** (default port **1234**).
3. For ingestion and retrieval, LM Studio must also serve embeddings on the same server:
   - Either load your embedding model when ingesting/querying, or use LM Studio’s workflow that exposes `/v1/embeddings` for the embedding model you downloaded.
4. On the server screen, note the **model IDs** exactly as shown (e.g. `qwen/qwen2.5-vl-7b`, `text-embedding-bge-small-en-v1.5`). Your `.env` values must match those IDs.

Keep LM Studio running whenever you use this project.

---

## 4. Create your `.env` file

From the project root:

```bash
cp .env.example .env
```

Edit `.env` with your values. Example aligned with the models above:

```env
# LM Studio (default port)
LM_STUDIO_BASE_URL=http://localhost:1234
LM_STUDIO_API_KEY=lm-studio

# Must match the embedding model id on LM Studio’s server tab
LM_STUDIO_EMBED_MODEL=text-embedding-bge-small-en-v1.5

# Vision + chat — must match the model id on LM Studio’s server tab
LM_STUDIO_CHAT_MODEL=qwen/qwen2.5-vl-7b
LM_STUDIO_MAX_TOKENS=3072

# Vector database (created by ingest_papers.py)
CHROMA_PATH=./chroma_db
COLLECTION_NAME=acne_research

# Paper sources (already in repo)
MARKDOWN_DIR=./markdown
MANIFEST_PATH=./markdown/manifest.json

# Retrieval strictness (lower = stricter). Try 0.24 if results feel off-topic.
RAG_MAX_DISTANCE=0.28
```

**Important:** If ingest or chat fails with “model not found”, copy the model string from LM Studio’s server UI into `LM_STUDIO_EMBED_MODEL` or `LM_STUDIO_CHAT_MODEL` — IDs can include suffixes like `@q4_k_m`.

`.env` is gitignored; never commit API keys.

---

## 5. Ingest research papers into Chroma

With LM Studio running and the **embedding** model available:

```bash
source .venv/bin/activate   # if not already active

# Test embeddings endpoint
python ingest_papers.py --ping

# Build the vector database (first time or after paper updates)
python ingest_papers.py --reset
```

`--reset` deletes and rebuilds the collection. First full ingest can take several minutes depending on CPU/GPU.

Quick test retrieval:

```bash
python ingest_papers.py --query "benzoyl peroxide inflammatory acne"
```

Smaller trial ingest:

```bash
python ingest_papers.py --max-docs 3 --reset
```

---

## 6. Verify LM Studio

```bash
# Embeddings
python ask_acne.py --ping-embed

# Chat / vision model
python ask_acne.py --ping-chat
```

---

## 7. Run from the command line (optional)

**Text + RAG (no photo):**

```bash
python ask_acne.py --question "What does research say about benzoyl peroxide for inflammatory acne?"
```

**Photo + RAG + Qwen VL:**

```bash
python ask_acne.py --image path/to/face.jpg --question "What might help based on what you see and the literature?"
```

**Chat only (no paper retrieval):**

```bash
python ask_acne.py --no-rag --question "General question without citations"
```

---

## 8. Run the web UI

```bash
python web_app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

Custom host/port:

```bash
python web_app.py --host 0.0.0.0 --port 5000
```

### Testing the website

1. Confirm **LM Studio** server is on and **`qwen/qwen2.5-vl-7b`** is loaded.
2. Confirm **`chroma_db/`** exists (you ran `python ingest_papers.py --reset`).
3. Open the site, type a question (e.g. *What do studies suggest about salicylic acid for acne?*).
4. Leave **Use research (RAG)** enabled — answers should cite numbered sources from retrieved papers.
5. Upload a **JPG, PNG, or WebP** face photo (max 8 MB) and ask a follow-up — vision runs first, then RAG + Qwen VL. Photo upload fails if the vision model is not loaded in LM Studio.
6. Toggle RAG off to use only the local model without paper citations.

If you see *Chroma database not found*, run ingest again:

```bash
python ingest_papers.py --reset
```

---

## Troubleshooting

| Problem | What to check |
|--------|----------------|
| `ingest_papers.py --ping` fails | LM Studio server started; embedding model loaded; `LM_STUDIO_EMBED_MODEL` matches server model id. |
| `ask_acne.py --ping-chat` fails | `qwen/qwen2.5-vl-7b` loaded in LM Studio; `LM_STUDIO_CHAT_MODEL` matches server id. |
| Web UI: Chroma not found | Run `python ingest_papers.py --reset`. |
| Photo upload / vision errors | Vision model loaded and server running; model id matches `.env`; image under 8 MB (JPG/PNG/WebP). |
| Weak or irrelevant citations | Lower `RAG_MAX_DISTANCE` (e.g. `0.24`); re-ingest after changing embed model. |
| Wrong answers after model change | Re-run `python ingest_papers.py --reset` — embeddings must use the same model for ingest and query. |

---

## Project layout (short)

| Path | Purpose |
|------|---------|
| `markdown/` | Converted research papers + `manifest.json` |
| `chroma_db/` | Persistent vector DB (created by ingest; gitignored) |
| `rag/` | Chunking, retrieval, LM Studio clients, Qwen vision, pipeline |
| `ingest_papers.py` | Build / query Chroma index |
| `ask_acne.py` | CLI assistant |
| `web_app.py` | Flask web UI |
| `convert_pdfs.py` | Optional: PDF → markdown via Marker |

---

## Disclaimer

This tool is for **informational and educational** use. It does not diagnose, prescribe, or replace care from a licensed clinician. If you have concerns about your skin, see a dermatologist or qualified healthcare provider.
