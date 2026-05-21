# Batch 0 [PS] - Local Ingestion (downloading papers and save them into vector DB)
# Run from project root: D:\dont_move\Northeastern_University\CS6120\final_project

pip install PyMuPDF chromadb sentence-transformers
python update/ingest.py --max-papers 800 # --query TOPIC
python -c "from src.processing.embedder import get_chunk_count_fast; print('total_chunks:', get_chunk_count_fast())"
