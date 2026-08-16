"""Section-aware chunking of parsed filings.

STATUS: signatures only — bodies are intentionally unimplemented (owner writes these).
Everything downstream (embedding, Qdrant upsert, BM25, retrieval) is already built
against the contract below, so filling these in is the only step between a parsed
filing and a searchable index.

Contract the rest of the pipeline relies on:
  * A chunk NEVER spans two sections. Section boundaries come from
    src/parsing/sec_parser.extract_sections(); mixing "Risk Factors" prose into an
    "MD&A" chunk is the failure mode this whole design exists to prevent, because
    `section` is a filterable payload field and a mixed chunk makes the filter lie.
  * Target CFG["chunking"]["target_tokens"] (750) with
    CFG["chunking"]["overlap_tokens"] (100) of overlap between consecutive chunks
    of the same section. 600-900 is the acceptable band; a section shorter than the
    floor still yields exactly one chunk rather than being dropped.
  * chunk_id is stable across re-runs for the same (doc_id, section, index). Qdrant
    point ids are uuid5(chunk_id) and the SQL write is an upsert on chunk_id, so a
    deterministic id is what makes re-indexing idempotent instead of duplicating.
  * n_tokens is measured with the embedding model's own tokenizer, not whitespace —
    it is the number that tells you whether a chunk is being silently truncated at
    bge-small's 512-wordpiece limit.
"""
from src.common.models import Chunk, ChunkMeta


def get_tokenizer():
    """Cached tokenizer for CFG["embedding"]["model"] (bge-small = BERT wordpiece).

    Loaded once at module level and reused; instantiating a transformers tokenizer
    per chunk dominates runtime otherwise.
    """
    raise NotImplementedError


def count_tokens(text: str) -> int:
    """Token count under the embedding model's tokenizer (no special tokens).

    Used both to size chunks and to populate Chunk.n_tokens / chunks.n_tokens.
    """
    raise NotImplementedError


def split_paragraphs(text: str) -> list[str]:
    """Split one section's text into paragraph units, the atoms chunks are packed from.

    sec_parser.html_to_text() emits one line per block element with blank lines
    already stripped, so "paragraph" here means a line (or run of lines) — not a
    `\\n\\n`-delimited block. Drop empties. A single paragraph longer than
    target_tokens is the caller's problem to split (see pack_paragraphs).
    """
    raise NotImplementedError


def pack_paragraphs(paragraphs: list[str], target_tokens: int,
                    overlap_tokens: int) -> list[str]:
    """Greedily pack paragraphs into ~target_tokens windows with token overlap.

    Accumulate paragraphs until adding the next one would exceed target_tokens, emit
    the window, then start the next window with trailing paragraphs worth roughly
    overlap_tokens so a fact straddling a boundary survives in at least one chunk.
    An oversized single paragraph (a wide financial table flattened to one line) must
    still be emitted — split it on token count rather than dropping it.

    Returns chunk texts in document order.
    """
    raise NotImplementedError


def make_chunk_id(doc_id: str, section: str, index: int) -> str:
    """Deterministic chunk_id, unique across the corpus and stable across re-runs.

    Suggested shape: f"{doc_id}::{section_slug}::{index:04d}" — doc_id is the SEC
    accession number, so this stays human-readable in eval files and failure reports
    (evals/questions.jsonl stores expected_chunk_ids by hand).
    """
    raise NotImplementedError


def make_meta(doc: dict, section: str, chunk_id: str) -> ChunkMeta:
    """Build a ChunkMeta by copying the document-level fields onto the chunk.

    filing_date must end up as a datetime.date (psycopg hands back a date, SQLite
    hands back an ISO string — normalize here, the indexer trusts the type).
    """
    raise NotImplementedError


def chunk_document(doc: dict, sections: list[tuple[str, str]]) -> list[Chunk]:
    """Chunk one parsed filing into Chunks carrying full metadata.

    Args:
        doc: a `documents` row as a dict — doc_id, ticker, company, doc_type,
             filing_date, fiscal_period, source_url. Every one of these becomes a
             Qdrant payload field, i.e. a retrieval filter, so copy them all onto
             each ChunkMeta rather than leaving any None.
        sections: [(section_name, text), ...] from extract_sections(). 8-Ks arrive as
             a single ("body", text) section.

    Returns:
        Chunks in document order with chunk_index numbered 0..n-1 across the whole
        document (continuing across sections, not restarting per section).
    """
    raise NotImplementedError
