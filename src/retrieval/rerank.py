"""Cross-encoder reranking: fused top-30 -> final top-6.

Why a second model at all. Dense retrieval is a *bi-encoder*: query and passage are
embedded independently and compared by cosine, so the passage vector is computed
without ever having seen the query. That is what makes ANN search possible (vectors
are precomputed and indexed) and it is also the ceiling — no term-level interaction
between the two. A cross-encoder concatenates query and passage into one sequence and
runs full attention across both, so "does *this* passage answer *this* question" is
computed jointly. It is far more accurate and far too slow to run over a corpus:
3,456 chunks would be 3,456 forward passes per query. Hence the funnel — cheap
retrievers cut 3,456 to 30, the expensive model orders those 30.

Score semantics: ms-marco-MiniLM-L-6-v2 emits a single unbounded logit per pair,
trained on MS MARCO relevance. Higher is better; the value is not a probability and
is not comparable across queries. Used for ordering only, never thresholded.

The 512 budget applies here too, and tighter than on the embedding side: the pair
shares one sequence, so a 480-token chunk plus the query plus three special tokens
lands within a few tokens of the limit. Truncation is on the passage tail and the
model's own `max_length` handles it, but it is the reason `target_tokens` is not
allowed to drift upward without re-checking both models (see DECISIONS #17).
"""
from sentence_transformers import CrossEncoder

from src.common.config import CFG
from src.common.models import RetrievedChunk

_MODEL: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Module-level singleton — loading weights is seconds, scoring a batch is ms."""
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder(CFG["retrieval"]["rerank_model"])
    return _MODEL


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int | None = None,
           batch_size: int = 32) -> list[RetrievedChunk]:
    """Re-score candidates against the query and return the best `top_k`.

    Requires hydrated text: the cross-encoder reads the passage, so a candidate that
    arrived without its text (hydrate=False upstream) is scored on an empty string.
    That would fail quietly as bad ranking, so it raises instead.
    """
    top_k = top_k or CFG["retrieval"]["final_top_k"]
    if not candidates:
        return []
    missing = [c.meta.chunk_id for c in candidates if not c.text]
    if missing:
        raise ValueError(
            f"rerank needs chunk text; {len(missing)} candidate(s) arrived empty "
            f"(e.g. {missing[0]}) — retrieve with hydrate=True"
        )

    scores = get_reranker().predict(
        [(query, c.text) for c in candidates],
        batch_size=batch_size,
        show_progress_bar=False,
    )
    ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
    return [
        RetrievedChunk(meta=c.meta, text=c.text, score=float(s), retriever="rerank")
        for c, s in ranked[:top_k]
    ]
