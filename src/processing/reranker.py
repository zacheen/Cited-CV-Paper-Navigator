"""Cross-encoder reranker for second-stage retrieval refinement.

Initial vector retrieval (cosine similarity over MiniLM bi-encoder
embeddings) is fast but coarse — it ranks by query/document embedding
similarity which only loosely tracks true semantic relevance. A
cross-encoder takes the (query, candidate) pair as a single input and
produces a real relevance score at the cost of one model forward pass per
candidate (no embedding cache).

The pipeline becomes: ChromaDB returns top-N=20 candidates fast, the
cross-encoder reranks those 20 down to top_k=5. On CPU, scoring 20 pairs
takes ~50ms — small relative to the LLM generation step that follows.
"""

import sys
import threading
from typing import TYPE_CHECKING

from src.config import RERANKER_MODEL

if TYPE_CHECKING:
    # Type-only import: keeps annotations precise for IDE/mypy without
    # pulling sentence_transformers (and its transformers/sklearn/torch
    # transitive chain, ~35s cold) into the main thread at module load.
    from sentence_transformers import CrossEncoder


# Module-level singleton. CrossEncoder allocates ~80MB and spins up torch
# tensors; we only want one instance per process. ``_reranker_lock`` makes
# concurrent first-callers cooperate (background warmup vs. first user
# query racing each other).
_reranker: "CrossEncoder | None" = None
_reranker_lock = threading.Lock()
_reranker_loaded = threading.Event()


def is_reranker_loaded() -> bool:
    """True iff the cross-encoder has been instantiated in this process."""
    return _reranker_loaded.is_set()


def get_reranker() -> "CrossEncoder":
    """Return the process-wide cross-encoder, loading it on first call."""
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is None:
            # Deferred import: this is the actual ~35s import cost. Kept
            # inside the function so the cost lands on whichever thread
            # first calls get_reranker() — normally the background warmup
            # thread spawned at app startup, not the main thread.
            from sentence_transformers import CrossEncoder

            print(
                f"[reranker] loading {RERANKER_MODEL}...",
                file=sys.stderr,
                flush=True,
            )
            _reranker = CrossEncoder(RERANKER_MODEL)
            _reranker_loaded.set()
            print("[reranker] loaded.", file=sys.stderr, flush=True)
    return _reranker


def rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    """Rerank retrieved chunks by cross-encoder relevance and truncate.

    Each result dict must have a ``"text"`` key. When the candidate pool
    already fits in top_k there is nothing to gain from rerank, so we skip
    the model call.

    Mutates each result in-place to add a ``"rerank_score"`` float (raw
    cross-encoder logit — not in [0,1]; ms-marco-MiniLM typically outputs
    roughly [-12, +12]). Higher is more relevant.

    Args:
        query: User query string used as the cross-encoder's left input.
        results: Retrieved chunk dicts from :func:`src.rag.retriever.retrieve`.
        top_k: Maximum number of results to return.

    Returns:
        A new list sorted by descending ``rerank_score``, truncated to
        ``top_k``. The input list is not modified, but the dicts inside
        gain a ``rerank_score`` key.
    """
    if not results or len(results) <= top_k:
        return results

    model = get_reranker()
    pairs = [(query, r["text"]) for r in results]
    scores = model.predict(pairs)

    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    return sorted(results, key=lambda r: r["rerank_score"], reverse=True)[:top_k]
