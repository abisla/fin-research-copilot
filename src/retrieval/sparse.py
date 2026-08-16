"""BM25 sparse retrieval over the same chunks the dense index holds.

Why sparse at all when we have embeddings: dense retrieval is lossy on exactly the
tokens financial questions hinge on — tickers, product names ("Blackwell", "GB200"),
regulatory acronyms ("CET1", "SOFR"), and line-item wording ("deferred revenue").
Those are rare tokens; BM25's IDF weighting is what surfaces them. Phase 3 fuses the
two with RRF.

What is persisted: the *tokenized corpus* plus per-chunk metadata, not a pickled
BM25Okapi object. Rebuilding the index from tokens takes well under a second at this
corpus size, and a pickled third-party object breaks on library upgrade — a pickle of
plain lists and dicts does not. Chunk text is deliberately not stored here either; it
is hydrated from SQL at search time so the corpus can't drift from the source of
truth (see src/indexing/store.py).

Tokenization is deliberately simple and must stay identical between build and query —
lowercase, keep alphanumerics plus in-token periods/hyphens so "10-k", "q3", "1.5"
and "u.s." survive as single terms, drop stopwords. No stemming: "impairment" and
"impaired" being distinct terms costs little when the dense retriever handles
morphology, and stemming would break exact-match cases (the reason BM25 is here).
"""
import pickle
import re
from datetime import date
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.common.config import CFG
from src.common.db import get_conn
from src.common.models import Chunk, ChunkMeta, RetrievedChunk
from src.indexing import store

ROOT = Path(__file__).parents[2]

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

# Tiny hand-rolled list. Full NLTK stopwords strip "no"/"not"/"against", which carry
# meaning in risk-factor language ("no assurance", "not able to").
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "at", "by", "with",
    "as", "is", "are", "was", "were", "be", "been", "being", "that", "this", "these",
    "those", "it", "its", "from", "we", "our", "us", "which", "such", "may", "will",
    "would", "could", "should", "has", "have", "had", "do", "does", "did", "but",
}


def index_path() -> Path:
    return ROOT / CFG["retrieval"]["bm25_path"]


def tokenize(text: str) -> list[str]:
    return [t.strip(".-") for t in TOKEN_RE.findall(text.lower())
            if t not in STOPWORDS and len(t) > 1]


def build_index(chunks: list[Chunk] | None = None, path: Path | None = None) -> int:
    """Tokenize the corpus and persist it. Reads all chunks from SQL when not given."""
    path = path or index_path()
    if chunks is None:
        conn = get_conn()
        try:
            rows = store.load_chunk_rows(conn)
        finally:
            conn.close()
        records = [(r["chunk_id"], r["text"], store.row_to_meta(r)) for r in rows]
    else:
        records = [(c.meta.chunk_id, c.text, c.meta) for c in chunks]

    payload = {
        "chunk_ids": [cid for cid, _, _ in records],
        "corpus": [tokenize(text) for _, text, _ in records],
        "meta": [vars(meta) for _, _, meta in records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    return len(records)


class BM25Index:
    """Loaded BM25 index. Metadata is in memory; text is fetched from SQL per query."""

    def __init__(self, chunk_ids: list[str], corpus: list[list[str]], meta: list[dict]):
        self.chunk_ids = chunk_ids
        self.meta = [ChunkMeta(**m) for m in meta]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def _keep(self, meta: ChunkMeta, tickers, doc_types, sections, date_from, date_to) -> bool:
        if tickers and meta.ticker not in tickers:
            return False
        if doc_types and meta.doc_type not in doc_types:
            return False
        if sections and meta.section not in sections:
            return False
        if date_from and (meta.filing_date is None or meta.filing_date < date_from):
            return False
        if date_to and (meta.filing_date is None or meta.filing_date > date_to):
            return False
        return True

    def search(self, query: str, top_k: int | None = None, conn=None,
               tickers: list[str] | None = None, doc_types: list[str] | None = None,
               sections: list[str] | None = None, date_from: date | None = None,
               date_to: date | None = None) -> list[RetrievedChunk]:
        """Top-k BM25 hits as RetrievedChunk, text hydrated from SQL.

        Filters are applied to the score array *before* the top-k cut, so a filtered
        search returns k filtered hits rather than whatever survives filtering k
        unfiltered ones — the same semantics Qdrant gives the dense side.
        """
        top_k = top_k or CFG["retrieval"]["sparse_top_k"]
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        idxs = [i for i in range(len(self.chunk_ids))
                if self._keep(self.meta[i], tickers, doc_types, sections, date_from, date_to)]
        # Drop non-matching chunks instead of padding top-k with them: a zero-score
        # BM25 hit shares no query term, and feeding it to RRF would hand a rank slot
        # to a chunk the sparse retriever has no opinion about.
        idxs = [i for i in idxs if scores[i] > 0]
        idxs.sort(key=lambda i: scores[i], reverse=True)
        idxs = idxs[:top_k]

        ids = [self.chunk_ids[i] for i in idxs]
        own_conn = conn is None
        conn = conn or get_conn()
        try:
            texts = store.load_texts(conn, ids)
        finally:
            if own_conn:
                conn.close()
        return [
            RetrievedChunk(meta=self.meta[i], text=texts.get(self.chunk_ids[i], ""),
                           score=float(scores[i]), retriever="bm25")
            for i in idxs
        ]


def load_index(path: Path | None = None) -> BM25Index:
    path = path or index_path()
    if not path.exists():
        raise FileNotFoundError(f"no BM25 index at {path} — run scripts/build_index.py first")
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    return BM25Index(payload["chunk_ids"], payload["corpus"], payload["meta"])
