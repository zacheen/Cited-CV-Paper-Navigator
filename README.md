# CS 6120 NLP Final Project - RAG for Computer Vision Papers

> **Note (branch `Design_for_Gemini`):** this branch targets the Gemini API
> backend only. The Ollama / LLaMA 3.2 path is intentionally out of scope here
> — features on this branch (e.g. ReAct flow) are designed against Gemini's
> Native Function Calling and are not guaranteed to work with Ollama.

This project is a Retrieval-Augmented Generation system for computer vision papers. It ingests papers from arXiv, tracks trends from Hugging Face, stores paper content in ChromaDB, and supports both grounded follow-up Q&A and one-click summaries of papers published in the last 7 days.

## Presentation

You can find the project presentation here: [Google Slides](https://docs.google.com/presentation/d/19PZll2_cLFK4NmbU4aPdjUmWjZph7Jwt/edit?slide=id.g3d81aacba0c_0_0#slide=id.g3d81aacba0c_0_0), [PDF](res/RAG_for_CV_Papers.pdf)

## Project Goal

AI technology is advancing rapidly. Even within computer vision alone, many new papers are published every day. Our goal is to reduce the cost of keeping up with new work by automatically collecting recent papers, summarizing them with a single click, and supporting grounded follow-up questions.

## Problem and Motivation

It is difficult for researchers and students to monitor recent computer vision trends by manually reading every new paper. Generic LLM answers can also hallucinate when they are not tied to real sources. This project addresses both issues by combining paper ingestion, recent-paper summarization, and retrieval-backed question answering in one interface.

## Data Used

- Papers from arXiv
- Trends from Hugging Face daily papers

## Solution

The system combines these models, techniques, and algorithms:

- `PyMuPDF` for PDF text extraction
- recursive chunking for long paper text
- `all-MiniLM-L6-v2` embeddings for vector indexing
- `ChromaDB` as the vector database
- `Ollama` with `llama3.2` for generation
- `Streamlit` for the user interface

The project supports two main workflows:

1. Retrieval-backed Q&A
   The system retrieves relevant chunks from ChromaDB and sends only the retrieved context to the LLM.

2. Recent-paper summary
   The system fetches recent papers directly from database metadata by date, then summarizes their abstracts or first chunks. This avoids relying on vector similarity for the `last 7 days` summary button.

## Development Process

We encountered several challenges during development:

- Long summary instructions were poor vector-search queries, so the system sometimes failed to retrieve recent papers.
- Retrieved chunks could contain references to older papers, which made summaries less reliable.
- Streamlit source watching triggered optional `transformers` imports in deployment.
- Direct full-collection reads from ChromaDB caused SQLite variable-limit errors.

We addressed these problems by:

- changing the 7-day summary flow to read recent papers directly from database metadata instead of vector search
- storing `abstract` in ChromaDB metadata during ingestion
- using first-chunk fallback for older indexed papers that do not yet have abstract metadata
- disabling Streamlit file watching in deployment
- batching database reads to avoid the `too many SQL variables` error

## Experimental Results

This project currently focuses on system functionality and grounded interaction rather than a benchmark-style evaluation suite.

Current results include:

- one-click summary generation for papers indexed within the last 7 days
- grounded follow-up Q&A over indexed computer vision papers
- reduced hallucination risk by restricting answers to retrieved or directly selected paper content

System-level observations:

- recent-paper summary no longer depends on vector similarity search
- recent summary mode can be followed by additional questions in the same chat
- large recent-paper lookups are now stable through batched reads

If needed, future work can add formal metrics such as latency, retrieval hit rate, and answer-quality evaluation.

## Specific Queries We Can Issue

- Click the summary button to receive a summary of papers from the last 7 days
- Ask follow-up questions after the summary
- Ask general grounded questions about indexed computer vision papers

## Usage

### Recommended Batch Workflow

Follow these scripts in order:

1. `scripts/GCP_deploy/batch0-local-ingest.ps1`
   Install local ingestion dependencies, download papers, and build the local ChromaDB.

2. `scripts/GCP_deploy/batch1-create-vm.ps1`
   Create the GCP VM.

3. `scripts/GCP_deploy/batch2-upload.ps1`
   Upload project files and the ChromaDB data to the VM.

4. `scripts/GCP_deploy/batch3-verify-upload.ps1`
   Verify that code and database files were uploaded correctly.

5. `scripts/GCP_deploy/batch4-firewall.ps1`
   Open the Streamlit port.

6. `scripts/GCP_deploy/batch5-get-url.ps1`
   Get the external URL for the app.

7. `scripts/GCP_deploy/batch6-ssh.ps1`
   SSH into the VM.

8. `scripts/GCP_deploy/batch7-vm-setup.sh`
   Install Docker and the NVIDIA container toolkit on the VM.

9. `scripts/GCP_deploy/batch8-docker-run.sh`
   Build and run the Docker container.

10. `scripts/GCP_deploy/batch9-setup-cron.sh` (optional)
    Set up daily trend ingestion with cron.

This is the recommended end-to-end deployment path because the scripts are already organized for sequential execution.

### Local Deploy Batches

`scripts/local_deploy/` contains PowerShell wrappers for the common local tasks. Activate your Python environment (conda/venv) first, then run each script from the project root.

> **Note:** Local deploy uses the Gemini API as the LLM backend by default (configured via `.env` with `LLM_BACKEND=gemini` and `GOOGLE_API_KEY=...`). Only the GCP deployment uses Ollama + LLaMA 3.2 on the VM. No local Ollama server is required.

| Script | Purpose |
|--------|---------|
| `scripts/local_deploy/setup.ps1` | First-time setup: `pip install -r requirements.txt` |
| `scripts/local_deploy/ingest.ps1` | Crawl arXiv cs.CV papers and build the local ChromaDB |
| `scripts/local_deploy/get-past-trend.ps1` | Backfill past HF Daily CV papers |
| `scripts/local_deploy/get-today-trend.ps1` | Ingest today's HF Daily CV papers (manual run) |
| `scripts/local_deploy/run-app.ps1` | Launch the Streamlit app at http://localhost:8501 |

Run with Docker:

```bash
docker build -t cv-rag .
docker run -p 8501:8501 -p 11434:11434 cv-rag
```

If PDFs already exist locally, they are not downloaded again. Re-running ingestion will still re-parse and update ChromaDB metadata such as `abstract`.

### How to Use

1. Open the Streamlit UI.
2. Click `Summarize new papers from last 7 days` to get a recent-paper summary.
3. Ask follow-up questions in the same chat.
4. Use normal chat input for broader retrieval-backed questions.

## Architecture at a Glance

```text
arXiv API / Hugging Face Daily Papers
    -> PDF download
    -> PyMuPDF parsing
    -> chunking + abstract metadata
    -> MiniLM embeddings
    -> ChromaDB
    -> Streamlit UI
    -> Ollama / LLaMA 3.2
```

## Key Source Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI with chat, recent-paper summary button, and follow-up flow |
| `update/ingest.py` | arXiv ingestion pipeline (CLI) |
| `update/get_past_trend.py` | Hugging Face past-trend ingestion (CLI wrapper around `src/ingestion/past_trend.py`) |
| `update/get_today_trend.py` | Daily cron entry point for HF Daily Papers |
| `src/config.py` | central configuration and prompts |
| `src/ingestion/arxiv_downloader.py` | arXiv API search and PDF download |
| `src/ingestion/hf_downloader.py` | Hugging Face daily paper retrieval |
| `src/ingestion/pdf_parser.py` | PDF parsing with PyMuPDF |
| `src/processing/chunker.py` | text chunking and metadata attachment |
| `src/processing/embedder.py` | embedding and ChromaDB indexing |
| `src/rag/retriever.py` | vector retrieval and direct recent-paper lookup |
| `src/rag/generator.py` | Ollama/LLaMA response generation |
| `Dockerfile` | container setup |
| `entrypoint.sh` | container startup |

## Additional References

| Resource | Location |
|----------|----------|
| System architecture | [architecture/system-overview.md](architecture/system-overview.md) |
| Design decisions log | [architecture/design-decisions.md](architecture/design-decisions.md) |
| Local ingestion & setup | [development/setup.md](development/setup.md) |
| GCP deployment guide | [development/gcp-deployment.md](development/gcp-deployment.md) |
| Modeling-level cautions | [Caution.md](Caution.md) |
