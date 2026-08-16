"""Shared datamodels. Every retriever returns RetrievedChunk so eval can swap them."""
from dataclasses import dataclass, field
from datetime import date


@dataclass
class ChunkMeta:
    chunk_id: str
    doc_id: str
    ticker: str
    company: str
    doc_type: str          # 10-K | 10-Q | 8-K | press_release | news
    filing_date: date | None
    fiscal_period: str | None
    section: str | None
    source_url: str | None


@dataclass
class Chunk:
    """A chunk as produced by the chunker and persisted to the `chunks` table.

    `meta.chunk_id` is the join key across all three stores: SQL row PK, Qdrant
    payload field, and BM25 corpus key.
    """
    meta: ChunkMeta
    text: str
    chunk_index: int        # position within its document, 0-based, ordered by section then offset
    n_tokens: int


@dataclass
class RetrievedChunk:
    meta: ChunkMeta
    text: str
    score: float
    retriever: str          # dense | bm25 | hybrid | rerank


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)   # chunk_ids actually cited
    route: str = ""
    insufficient_evidence: bool = False
