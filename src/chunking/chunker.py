"""Section-aware chunking of parsed filings.

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
import re
from functools import lru_cache

from transformers import AutoTokenizer

from src.common.config import CFG
from src.common.models import Chunk, ChunkMeta
from src.indexing.store import as_date


@lru_cache(maxsize=1)
def get_tokenizer():
    """Cached tokenizer for CFG["embedding"]["model"] (bge-small = BERT wordpiece).

    Loaded once and reused; instantiating a transformers tokenizer per chunk
    dominates runtime otherwise.
    """
    return AutoTokenizer.from_pretrained(CFG["embedding"]["model"])


def count_tokens(text: str) -> int:
    """Token count under the embedding model's tokenizer (no special tokens).

    Used both to size chunks and to populate Chunk.n_tokens / chunks.n_tokens.
    """
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def split_paragraphs(text: str) -> list[str]:
    """Split one section's text into paragraph units, the atoms chunks are packed from.

    sec_parser.html_to_text() emits one line per block element with blank lines
    already stripped, so "paragraph" here means a line (or run of lines) — not a
    `\\n\\n`-delimited block. Drop empties. A single paragraph longer than
    target_tokens is the caller's problem to split (see pack_paragraphs).
    """
    return [line.strip() for line in text.splitlines() if line.strip()]


def _split_oversized(text: str, max_tokens: int) -> list[str]:
    """Cut one over-long paragraph into <= max_tokens pieces on token boundaries.

    Uses the fast tokenizer's offset mapping to slice the *original* string rather
    than decoding wordpieces back to text, which would mangle casing, punctuation and
    the "##" continuations that flattened financial tables are full of.
    """
    enc = get_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(s, e) for s, e in enc["offset_mapping"] if e > s]
    pieces = []
    for i in range(0, len(offsets), max_tokens):
        window = offsets[i:i + max_tokens]
        piece = text[window[0][0]:window[-1][1]].strip()
        if piece:
            pieces.append(piece)
    return pieces or ([text.strip()] if text.strip() else [])


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
    # (text, n_tokens) units, each guaranteed to fit inside one window.
    units: list[tuple[str, int]] = []
    for para in paragraphs:
        n = count_tokens(para)
        if n <= target_tokens:
            if n:
                units.append((para, n))
        else:
            units.extend((p, count_tokens(p)) for p in _split_oversized(para, target_tokens))
    if not units:
        return []

    # An overlap tail may overshoot overlap_tokens (paragraphs are atomic) but never
    # take over half a window — otherwise one fat trailing paragraph makes
    # consecutive chunks near-duplicates and inflates the corpus.
    max_tail = max(overlap_tokens, target_tokens // 2)

    def tail_of(window: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """Trailing paragraphs of an emitted window, to seed the next one."""
        tail: list[tuple[str, int]] = []
        total = 0
        for unit in reversed(window[:-1]):     # never re-emit the whole window
            if total >= overlap_tokens or total + unit[1] > max_tail:
                break
            tail.insert(0, unit)
            total += unit[1]
        return tail

    chunks: list[str] = []
    window: list[tuple[str, int]] = []
    window_tokens = 0
    for text, n in units:
        if window and window_tokens + n > target_tokens:
            chunks.append("\n".join(t for t, _ in window))
            window = tail_of(window)
            window_tokens = sum(tn for _, tn in window)
            # The unit that forced the emit still has to fit beside the tail; give up
            # overlap paragraph by paragraph rather than blow past target_tokens.
            while window and window_tokens + n > target_tokens:
                window_tokens -= window.pop(0)[1]
        window.append((text, n))
        window_tokens += n
    if window:
        chunks.append("\n".join(t for t, _ in window))
    return chunks


def make_chunk_id(doc_id: str, section: str, index: int) -> str:
    """Deterministic chunk_id, unique across the corpus and stable across re-runs.

    Shape: f"{doc_id}::{section_slug}::{index:04d}" — doc_id is the SEC accession
    number, so this stays human-readable in eval files and failure reports
    (evals/questions.jsonl stores expected_chunk_ids by hand). `index` counts within
    the section: the section is already part of the key, and numbering per section
    means resizing an earlier section does not renumber a later section's ids.
    """
    slug = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", (section or "body").lower())).strip("-")
    return f"{doc_id}::{slug or 'body'}::{index:04d}"


def make_meta(doc: dict, section: str, chunk_id: str) -> ChunkMeta:
    """Build a ChunkMeta by copying the document-level fields onto the chunk.

    filing_date must end up as a datetime.date (psycopg hands back a date, SQLite
    hands back an ISO string — normalize here, the indexer trusts the type).
    """
    return ChunkMeta(
        chunk_id=chunk_id,
        doc_id=doc["doc_id"],
        ticker=doc["ticker"],
        company=doc["company"],
        doc_type=doc["doc_type"],
        filing_date=as_date(doc.get("filing_date")),
        fiscal_period=doc.get("fiscal_period"),
        section=section,
        source_url=doc.get("source_url"),
    )


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
    target = CFG["chunking"]["target_tokens"]
    overlap = CFG["chunking"]["overlap_tokens"]

    chunks: list[Chunk] = []
    chunk_index = 0
    for section, text in sections:
        # Packing runs per section, so no window can straddle a section boundary.
        for i, body in enumerate(pack_paragraphs(split_paragraphs(text), target, overlap)):
            chunk_id = make_chunk_id(doc["doc_id"], section, i)
            chunks.append(Chunk(
                meta=make_meta(doc, section, chunk_id),
                text=body,
                chunk_index=chunk_index,
                n_tokens=count_tokens(body),
            ))
            chunk_index += 1
    return chunks
