"""Dense retrieval: bge-small query vector -> Qdrant HNSW search with payload filters.

Three things this module is deliberately strict about:

* **The query goes through `embed_query`, never `embed_passages`.** bge is asymmetric;
  the instruction prefix belongs on the query side only and its absence is silent
  (see DECISIONS #16). One import, one call site, no flag to forget.
* **Filters are pushed into the search**, as a Qdrant `Filter` over indexed payload
  fields, so the constraint runs inside the HNSW traversal rather than over its
  output. Post-filtering k hits returns fewer than k exactly when the filter is
  selective, which reads as a sparse corpus rather than as a bug (DECISIONS #15).
* **Text is hydrated from SQL**, because the payload deliberately doesn't carry it
  (DECISIONS #12). One `WHERE chunk_id IN (...)` per search, not per hit.

Scores are raw cosine in [-1, 1] (vectors are L2-normalized at encode time). They are
returned as-is rather than rescaled: RRF downstream consumes ranks only, and a
normalized-looking score would invite someone to threshold on it across corpora,
which cosine does not support.
"""
from src.common.config import CFG
from src.common.db import get_conn, get_qdrant
from src.common.models import Filters, RetrievedChunk
from src.indexing import store, vectors


def payload_to_meta(payload: dict):
    """Qdrant payload -> ChunkMeta. Inverse of vectors.payload_for()."""
    return store.row_to_meta({
        "chunk_id": payload["chunk_id"],
        "doc_id": payload["doc_id"],
        "ticker": payload["ticker"],
        "company": payload["company"],
        "doc_type": payload["doc_type"],
        "filing_date": payload.get("filing_date"),
        "fiscal_period": payload.get("fiscal_period"),
        "section": payload.get("section"),
        "source_url": payload.get("source_url"),
    })


def search(query: str, top_k: int | None = None, filters: Filters | None = None,
           client=None, conn=None, collection: str | None = None,
           hydrate: bool = True) -> list[RetrievedChunk]:
    """Top-k dense hits as RetrievedChunk, ordered by descending cosine similarity.

    `client`/`conn` are injectable so a caller holding long-lived handles (the
    Retriever in hybrid.py, the eval harness, Streamlit) pays the connection cost
    once instead of once per query. When not supplied they are opened and closed
    here, which keeps one-off scripted use a single call.

    `hydrate=False` skips the SQL text fetch — used by the eval harness, which
    scores chunk_ids and never reads the text.
    """
    from src.common.embeddings import embed_query      # lazy: loading the model is seconds

    top_k = top_k or CFG["retrieval"]["dense_top_k"]
    own_client = client is None
    client = client or get_qdrant()
    try:
        hits = client.query_points(
            collection or vectors.collection_name(),
            query=embed_query(query),
            limit=top_k,
            query_filter=(filters or Filters()).to_qdrant(),
            with_payload=True,
        ).points
    finally:
        if own_client:
            client.close()

    metas = [payload_to_meta(h.payload) for h in hits]
    texts: dict[str, str] = {}
    if hydrate and metas:
        own_conn = conn is None
        conn = conn or get_conn()
        try:
            texts = store.load_texts(conn, [m.chunk_id for m in metas])
        finally:
            if own_conn:
                conn.close()

    return [
        RetrievedChunk(meta=m, text=texts.get(m.chunk_id, ""), score=float(h.score),
                       retriever="dense")
        for m, h in zip(metas, hits)
    ]
