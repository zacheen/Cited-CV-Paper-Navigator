"""Centralized configuration for the RAG pipeline."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# arXiv settings
ARXIV_CATEGORY = "cs.CV"
ARXIV_MAX_RESULTS = 800
ARXIV_SORT_BY = "submittedDate"

# Chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "arxiv_cv_papers"

# Reranker (second-stage retrieval refinement). The bi-encoder above is fast
# but coarse; the cross-encoder below scores (query, candidate) pairs jointly
# and produces a sharper ordering. We fetch RERANKER_INITIAL_K from ChromaDB
# and rerank down to top_k.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_INITIAL_K = 20

# Ollama / LLM settings
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TOP_K = 5

# Gemini (Google AI Studio) settings — free-tier alternative to Ollama on a VM.
# Get a free API key at https://aistudio.google.com/apikey and export it as
# GOOGLE_API_KEY before running.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# LLM backend selector: "ollama" (default, original behavior) or "gemini".
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# ReAct agent — maximum tool-calling rounds before forcing a final answer.
# Each round is one (LLM call, tool execution) pair. Past 3 rounds, latency
# starts to dominate without much extra retrieval signal in our experience.
REACT_MAX_STEPS = 3

# RAG prompt template
RAG_SYSTEM_PROMPT = (
    "You are a research assistant specializing in computer vision. "
    "Answer the user's question using only the provided context from "
    "arXiv computer vision papers. Do not use outside knowledge or infer "
    "details from papers that are not in the supplied context. "
    "Also ensure that the response content is relevant to the question; otherwise, discard it. "
    "If the context does not contain enough information, say so clearly. "
    "Cite the paper titles when possible."
)

RAG_USER_TEMPLATE = (
    "Context from relevant papers:\n"
    "---\n"
    "{context}\n"
    "---\n\n"
    "Question: {question}"
)
