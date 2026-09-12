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
    """A generated answer plus everything needed to audit it.

    `citations` holds evidence *keys*, not [n] indices — a chunk_id, an article_id or a
    financials coordinate — because the numbering is per-question and meaningless once
    the answer is stored, while the keys are what Phase 7 compares to expected_chunk_ids.
    `hallucinated_citations` is kept rather than silently dropped: a model citing a
    chunk that was never supplied is a measurable failure, not a formatting slip.
    """
    text: str
    citations: list[str] = field(default_factory=list)   # evidence keys actually cited
    route: str = ""
    insufficient_evidence: bool = False
    hallucinated_citations: list[int] = field(default_factory=list)
    evidence: list = field(default_factory=list)         # list[answer.Evidence]; untyped to avoid a cycle
    decision: object | None = None                       # router.RouteDecision

    @property
    def uncited(self) -> bool:
        """A substantive answer with no citation at all — principle #5 says every claim
        cites or the answer says insufficient evidence, so this is a contract breach."""
        return not self.citations and not self.insufficient_evidence


@dataclass
class Filters:
    """Hard metadata constraints, applied before similarity (CLAUDE.md principle #4).

    One object instead of five kwargs threaded through dense, sparse, hybrid and the
    eval harness. It carries the *same* predicates to two very different engines —
    a Qdrant payload `Filter` on the dense side, a Python predicate over the score
    array on the sparse side — which is what makes an A/B between retrievers a
    comparison of retrievers rather than of filter semantics.

    Empty means unconstrained: `to_qdrant()` returns None (Qdrant treats that as no
    filter) and `to_kwargs()` returns all-None kwargs (sparse skips each predicate).
    """
    tickers: list[str] | None = None
    doc_types: list[str] | None = None
    sections: list[str] | None = None
    date_from: date | None = None
    date_to: date | None = None

    def to_kwargs(self) -> dict:
        """Filter kwargs for sparse.BM25Index.search() and vectors.build_filter()."""
        return {
            "tickers": self.tickers, "doc_types": self.doc_types,
            "sections": self.sections, "date_from": self.date_from,
            "date_to": self.date_to,
        }

    def to_qdrant(self):
        """Qdrant Filter, or None when nothing is constrained."""
        from src.indexing.vectors import build_filter    # lazy: avoids import cycle
        return build_filter(**self.to_kwargs())

    def is_empty(self) -> bool:
        return not any(self.to_kwargs().values())

    def describe(self) -> str:
        """Short human-readable form, for eval logs and the Streamlit sidebar."""
        parts = [f"{k}={v}" for k, v in self.to_kwargs().items() if v]
        return ", ".join(parts) if parts else "none"
