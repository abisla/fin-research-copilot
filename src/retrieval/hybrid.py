"""Hybrid retrieval via reciprocal rank fusion (RRF), plus the Retriever facade.

Why RRF over weighted score fusion: dense cosine scores and BM25 scores live on
incomparable scales; RRF only uses ranks, so no per-corpus score normalization
is needed. Tradeoff: throws away score magnitude information.
Interview q: "why k=60?" -> standard from Cormack et al.; dampens the impact of
top-1 rank differences; system is insensitive to k in [20, 100] (show eval).

`Retriever` exists so the four retrieval strategies are *one* object with a mode
argument rather than four call paths. Phase 7 measures dense-only vs hybrid vs
hybrid+rerank and Phase 8 toggles between them live in the sidebar; both are only
honest comparisons if every arm shares the same filters, the same connections and
the same candidate widths. Anything mode-specific lives in `_candidates`, and the
models/handles are loaded once per Retriever, not once per query.
"""
from src.common.config import CFG
from src.common.db import get_conn, get_qdrant
from src.common.models import Filters, RetrievedChunk
from src.retrieval import dense as dense_mod
from src.retrieval import sparse as sparse_mod

MODES = ("dense", "bm25", "hybrid", "rerank")


def rrf_fuse(dense: list[RetrievedChunk], sparse: list[RetrievedChunk], k: int = 60,
             top_n: int = 30) -> list[RetrievedChunk]:
    """Fuse two ranked lists by sum of 1/(k + rank). Ties broken by insertion order."""
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}
    for results in (dense, sparse):
        for rank, ch in enumerate(results, start=1):
            cid = ch.meta.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            # Prefer whichever copy actually carries text: the two retrievers hydrate
            # from the same SQL row, but an un-hydrated eval run would otherwise let
            # an empty-text dense hit mask the sparse copy that has it.
            if cid not in by_id or (not by_id[cid].text and ch.text):
                by_id[cid] = ch
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    out = []
    for cid, s in ranked:
        ch = by_id[cid]
        out.append(RetrievedChunk(meta=ch.meta, text=ch.text, score=s, retriever="hybrid"))
    return out


class Retriever:
    """Holds the query-time handles: Qdrant client, DB connection, BM25 index.

    Each is expensive once and free thereafter, so a long-lived Retriever is the unit
    of reuse for the eval harness (40 questions x 4 modes) and the Streamlit session.
    Handles are opened lazily — constructing a Retriever to run `bm25` mode should not
    require Qdrant to be up, and vice versa.
    """

    def __init__(self, conn=None, client=None, bm25=None, collection: str | None = None):
        self._conn, self._client, self._bm25 = conn, client, bm25
        self._owns = {"conn": conn is None, "client": client is None}
        self.collection = collection

    @property
    def conn(self):
        if self._conn is None:
            self._conn = get_conn()
        return self._conn

    @property
    def client(self):
        if self._client is None:
            self._client = get_qdrant()
        return self._client

    @property
    def bm25(self) -> sparse_mod.BM25Index:
        if self._bm25 is None:
            self._bm25 = sparse_mod.load_index()
        return self._bm25

    def close(self) -> None:
        if self._owns["conn"] and self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._owns["client"] and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def dense(self, query: str, top_k: int | None = None, filters: Filters | None = None,
              hydrate: bool = True) -> list[RetrievedChunk]:
        return dense_mod.search(query, top_k=top_k, filters=filters, client=self.client,
                                conn=self.conn, collection=self.collection, hydrate=hydrate)

    def sparse(self, query: str, top_k: int | None = None,
               filters: Filters | None = None) -> list[RetrievedChunk]:
        return self.bm25.search(query, top_k=top_k, conn=self.conn,
                                **(filters or Filters()).to_kwargs())

    def _candidates(self, query: str, filters: Filters | None,
                    hydrate: bool) -> list[RetrievedChunk]:
        """Fused pool feeding both `hybrid` and `rerank` — identical for the two, so
        the reranker's contribution is measured in isolation."""
        cfg = CFG["retrieval"]
        return rrf_fuse(
            self.dense(query, top_k=cfg["dense_top_k"], filters=filters, hydrate=hydrate),
            self.sparse(query, top_k=cfg["sparse_top_k"], filters=filters),
            k=cfg["rrf_k"],
            top_n=cfg["dense_top_k"],
        )

    def search(self, query: str, mode: str = "rerank", top_k: int | None = None,
               filters: Filters | None = None, hydrate: bool = True) -> list[RetrievedChunk]:
        """Retrieve by strategy. `top_k` is the size of the *returned* list; the
        internal candidate widths always come from config so the arms stay comparable.

        mode:
          dense  — bi-encoder ANN only (the baseline the other arms must beat)
          bm25   — sparse only (wins on rare tokens: tickers, "Blackwell", "CET1")
          hybrid — RRF over dense top-30 + BM25 top-30
          rerank — hybrid, then cross-encoder over the fused pool (the default)
        """
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        cfg = CFG["retrieval"]

        if mode == "dense":
            return self.dense(query, top_k=top_k or cfg["dense_top_k"], filters=filters,
                              hydrate=hydrate)
        if mode == "bm25":
            return self.sparse(query, top_k=top_k or cfg["sparse_top_k"], filters=filters)

        fused = self._candidates(query, filters, hydrate)
        if mode == "hybrid":
            return fused[:top_k or cfg["dense_top_k"]]

        from src.retrieval.rerank import rerank      # lazy: don't load weights for other modes
        return rerank(query, fused, top_k=top_k or cfg["final_top_k"])


def search(query: str, mode: str = "rerank", top_k: int | None = None,
           filters: Filters | None = None) -> list[RetrievedChunk]:
    """One-shot convenience wrapper. Opens and closes its own handles — fine for a
    script or a notebook, wasteful in a loop; use a Retriever there."""
    with Retriever() as r:
        return r.search(query, mode=mode, top_k=top_k, filters=filters)
