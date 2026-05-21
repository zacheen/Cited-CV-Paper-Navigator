# Development Setup

This project uses a **two-phase workflow**:
1. **Phase 1 (Local)** — Crawl arXiv, parse PDFs, build ChromaDB vector database
2. **Phase 2 (GCP)** — Upload database, run LLM + Streamlit on GPU VM

---

## Phase 1: Local Ingestion

### Prerequisites

```bash
pip install PyMuPDF chromadb sentence-transformers
```

No GPU or Ollama needed for this phase.

### Hugging Face Token (recommended)

The first ingest (and the first app start on a fresh machine) downloads two
small models from Hugging Face Hub via `sentence-transformers`:

- `all-MiniLM-L6-v2` — embedder (~90 MB)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — reranker (~80 MB)

Without a token these downloads run anonymously and share a low per-IP rate
limit. To avoid the `Warning: You are sending unauthenticated requests to the
HF Hub` message — and the rate-limit risk it implies — create a free
read-only token at https://huggingface.co/settings/tokens and add it to
`.env`:

```
HF_TOKEN = hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
```

`src/config.py` loads `.env` on import, and `huggingface_hub` picks the
token up automatically — no code changes needed. The models cache to
`~/.cache/huggingface/hub/` after the first download, so subsequent runs
won't hit the network at all.

### Paper Selection Criteria

The ingestion script queries the [arXiv API](https://info.arxiv.org/help/api/index.html) with the following defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Category | `cs.CV` | Computer Vision and Pattern Recognition |
| Sort by | `submittedDate` | Most recently submitted papers first |
| Sort order | `descending` | Newest first |
| Max results | `800` | Target: 800 papers x ~20 chunks = ~16k entries |
| Query filter | (none) | Optional free-text filter via `--query` |

### Run Ingestion

```bash
# Default: download 800 latest cs.CV papers and build vector database
python update/ingest.py --max-papers 800

# Filter by topic
python update/ingest.py --query "object detection" --max-papers 800

# Process already-downloaded PDFs only (skip arXiv API + download)
python update/ingest.py --skip-download
```

### Output

```
data/
  pdfs/          # Downloaded PDF files (~800 files)
  chroma_db/     # Vector database (this gets uploaded to GCP)
```

### Verify

```bash
python -c "from src.processing.embedder import get_collection_stats; print(get_collection_stats())"
```

### Time Estimate

| Stage | Duration |
|-------|----------|
| arXiv API search | ~30 seconds |
| PDF download (rate-limited) | ~15 minutes |
| PDF parsing | ~5 minutes |
| Chunking + embedding + indexing | ~10-30 minutes |
| **Total** | **~30-60 minutes** |

---

## Phase 2: GCP Serving

For full GCP deployment instructions, see [gcp-deployment.md](gcp-deployment.md).

Summary:
1. Upload `data/` folder to GCP VM via `gcloud compute scp`
2. Docker container runs Ollama (LLaMA 3.2) + Streamlit
3. No ingestion runs on GCP — it only serves queries

### Local Development (optional)

If you want to test the full RAG locally (requires Ollama installed):

```bash
# Start Ollama and pull LLaMA
ollama serve &
ollama pull llama3.2

# Run the Streamlit app
streamlit run app.py
```

Open http://localhost:8501

---

## Project Structure

```
final_project/
  app.py                      # Streamlit RAG chatbot UI
  requirements.txt             # Python dependencies (all)
  Dockerfile                   # Single container (Ollama + Streamlit)
  entrypoint.sh                # Container startup (Ollama + Streamlit only)
  .gitignore
  CLAUDE.md                    # Agent guidance
  data/
    ingest.py                  # CLI ingestion pipeline (run locally)
    get_past_trend.py          # HF Daily Papers backfill
    get_today_trend.py         # HF Daily Papers daily cron entry point
    pdfs/                      # (gitignored) Downloaded PDFs
    chroma_db/                 # (gitignored) ChromaDB vector store
  src/
    config.py                  # Centralized configuration
    ingestion/
      arxiv_downloader.py      # arXiv API search + PDF download
      pdf_parser.py            # PyMuPDF text extraction
    processing/
      chunker.py               # Recursive text chunking
      embedder.py              # Embedding + ChromaDB indexing
    rag/
      retriever.py             # ChromaDB query retrieval
      generator.py             # Ollama/LLaMA answer generation
  docs/                        # Project documentation
```

## Testing Individual Modules

```bash
# Test arXiv search (returns metadata, no download)
python -c "from src.ingestion.arxiv_downloader import search_arxiv; print(search_arxiv(max_results=3))"

# Test PDF parsing
python -c "from src.ingestion.pdf_parser import parse_pdf; from pathlib import Path; print(parse_pdf(Path('data/pdfs/SOME_ID.pdf'))['text'][:500])"

# Check ChromaDB collection stats
python -c "from src.processing.embedder import get_collection_stats; print(get_collection_stats())"
```
