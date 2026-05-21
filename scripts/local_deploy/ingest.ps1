# Local - Crawl arXiv cs.CV papers and build the local ChromaDB.
# Activate your conda/venv environment first, then run from project root.

python update/ingest.py --max-papers 800 # --query TOPIC
python -c "from src.processing.embedder import get_chunk_count_fast; print('total_chunks:', get_chunk_count_fast())"
