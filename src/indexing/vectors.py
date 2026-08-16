"""Qdrant collection management and chunk upsert.

Design notes worth defending:

* **Payload = metadata only, no text.** Text is hydrated from SQL by chunk_id (see
  store.py). Keeps one copy of the text and keeps the vector store's payload small
  enough to stay hot.
* **Deterministic point ids.** Qdrant ids must be int or UUID, and our chunk_id is a
  readable string, so the point id is uuid5(chunk_id). Same chunk_id -> same point,
  which makes re-indexing an overwrite instead of a duplicate. The human-readable
  chunk_id also rides along in the payload, because that is the key everything else
  (SQL, BM25, eval fixtures) joins on.
* **filing_date is stored twice**: `filing_date` as an ISO string for display/exact
  match, `filing_ts` as epoch seconds for range filters. Qdrant's datetime range
  support varies by version and storage backend; an integer Range works everywhere,
  including the embedded local-mode client.
* **Payload indexes are explicit.** Without them Qdrant still filters correctly but
  scans the payload; with them the filter is pushed into the HNSW traversal. That is
  the "hard filters first, similarity second" principle from CLAUDE.md, enforced at
  the storage layer rather than by post-filtering search results in Python.
"""
import uuid
from datetime import date, datetime, timezone

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from src.common.config import CFG
from src.common.models import Chunk

_NAMESPACE = uuid.UUID("6f6d1d3c-6c2e-5a2b-9a3f-1f0d5b8e7c41")   # fixed => ids are reproducible

# field -> payload index type. Filterable dimensions from CLAUDE.md principle #3.
INDEXED_FIELDS = {
    "ticker": PayloadSchemaType.KEYWORD,
    "doc_type": PayloadSchemaType.KEYWORD,
    "section": PayloadSchemaType.KEYWORD,
    "fiscal_period": PayloadSchemaType.KEYWORD,
    "doc_id": PayloadSchemaType.KEYWORD,
    "chunk_id": PayloadSchemaType.KEYWORD,
    "filing_ts": PayloadSchemaType.INTEGER,
}


def collection_name() -> str:
    return CFG["qdrant"]["collection"]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def _epoch(d: date | None) -> int | None:
    if d is None:
        return None
    if not isinstance(d, datetime):
        d = datetime(d.year, d.month, d.day)
    return int(d.replace(tzinfo=timezone.utc).timestamp())


def ensure_collection(client, name: str | None = None, recreate: bool = False) -> str:
    """Create the collection (HNSW m/ef_construct from config) and its payload indexes."""
    name = name or collection_name()
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config=VectorParams(size=CFG["embedding"]["dim"], distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(
                m=CFG["qdrant"].get("hnsw_m", 16),
                ef_construct=CFG["qdrant"].get("hnsw_ef_construct", 100),
            ),
        )
    for field, schema in INDEXED_FIELDS.items():
        try:
            client.create_payload_index(name, field_name=field, field_schema=schema)
        except Exception:
            pass    # already indexed — Qdrant has no create-if-not-exists for this
    return name


def payload_for(chunk: Chunk) -> dict:
    m = chunk.meta
    return {
        "chunk_id": m.chunk_id,
        "doc_id": m.doc_id,
        "ticker": m.ticker,
        "company": m.company,
        "doc_type": m.doc_type,
        "filing_date": m.filing_date.isoformat() if m.filing_date else None,
        "filing_ts": _epoch(m.filing_date),
        "fiscal_period": m.fiscal_period,
        "section": m.section,
        "source_url": m.source_url,
        "chunk_index": chunk.chunk_index,
        "n_tokens": chunk.n_tokens,
    }


def upsert_chunks(client, chunks: list[Chunk], vectors, name: str | None = None,
                  batch_size: int = 256) -> int:
    """Upsert chunk vectors + metadata payloads. `vectors` is aligned with `chunks`."""
    name = name or collection_name()
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    points = [
        PointStruct(id=point_id(c.meta.chunk_id), vector=list(map(float, v)), payload=payload_for(c))
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(points), batch_size):
        client.upsert(name, points=points[i:i + batch_size], wait=True)
    return len(points)


def delete_doc(client, doc_id: str, name: str | None = None) -> None:
    """Drop every point for a document — used before re-chunking it."""
    client.delete(
        name or collection_name(),
        points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        wait=True,
    )


def build_filter(tickers: list[str] | None = None, doc_types: list[str] | None = None,
                 sections: list[str] | None = None, date_from: date | None = None,
                 date_to: date | None = None) -> Filter | None:
    """Metadata filter for dense search. Returns None when nothing is constrained."""
    must = []
    for field, values in (("ticker", tickers), ("doc_type", doc_types), ("section", sections)):
        if values:
            must.append(FieldCondition(key=field, match=MatchAny(any=list(values))))
    if date_from or date_to:
        must.append(FieldCondition(
            key="filing_ts",
            range=Range(gte=_epoch(date_from), lte=_epoch(date_to)),
        ))
    return Filter(must=must) if must else None
