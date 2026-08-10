"""Hybrid retrieval via reciprocal rank fusion (RRF).

Why RRF over weighted score fusion: dense cosine scores and BM25 scores live on
incomparable scales; RRF only uses ranks, so no per-corpus score normalization
is needed. Tradeoff: throws away score magnitude information.
Interview q: "why k=60?" -> standard from Cormack et al.; dampens the impact of
top-1 rank differences; system is insensitive to k in [20, 100] (show eval).
"""
from src.common.models import RetrievedChunk


def rrf_fuse(dense: list[RetrievedChunk], sparse: list[RetrievedChunk], k: int = 60,
             top_n: int = 30) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}
    for results in (dense, sparse):
        for rank, ch in enumerate(results, start=1):
            cid = ch.meta.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(cid, ch)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    out = []
    for cid, s in ranked:
        ch = by_id[cid]
        out.append(RetrievedChunk(meta=ch.meta, text=ch.text, score=s, retriever="hybrid"))
    return out
