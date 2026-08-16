"""Chunk text persistence in the relational store — the single source of truth.

Chunk *text* lives here exactly once. Qdrant holds vectors + metadata payload, the
BM25 pickle holds tokens + metadata, and both key back to `chunks.chunk_id` to
hydrate text at read time. The alternative (copy the text into every store) is
faster to query and guarantees the three copies drift apart the first time chunking
params change. One join is cheap; a corpus that disagrees with itself is not.

Everything here runs on both storage modes, so no backend-specific SQL: `?` vs `%s`
placeholders are swapped per mode, rows are dicts built from cursor.description
(sqlite3.Row and psycopg's tuple rows have no common dict interface), and the
upsert is `ON CONFLICT ... DO UPDATE`, which SQLite and Postgres both speak.
"""
from datetime import date, datetime

from src.common.db import storage_mode
from src.common.models import Chunk, ChunkMeta

CHUNK_COLS = ["chunk_id", "doc_id", "section", "chunk_index", "n_tokens", "text"]

_SELECT = """
SELECT c.chunk_id, c.doc_id, c.section, c.chunk_index, c.n_tokens, c.text,
       d.ticker, d.company, d.doc_type, d.filing_date, d.fiscal_period, d.source_url
FROM chunks c
JOIN documents d ON d.doc_id = c.doc_id
"""


def placeholder() -> str:
    """`?` for SQLite, `%s` for Postgres: the only dialect difference in this module."""
    return "?" if storage_mode() == "local" else "%s"


def _dicts(cur) -> list[dict]:
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def as_date(value) -> date | None:
    """Normalize a filing_date from either backend (psycopg: date, SQLite: str)."""
    if value is None or isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value)[:10])


def row_to_meta(row: dict) -> ChunkMeta:
    return ChunkMeta(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        ticker=row["ticker"],
        company=row["company"],
        doc_type=row["doc_type"],
        filing_date=as_date(row["filing_date"]),
        fiscal_period=row["fiscal_period"],
        section=row["section"],
        source_url=row["source_url"],
    )


def write_chunks(conn, chunks: list[Chunk]) -> int:
    """Upsert chunks by chunk_id. Re-running the indexer overwrites, never duplicates."""
    if not chunks:
        return 0
    ph = placeholder()
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in CHUNK_COLS if c != "chunk_id")
    sql = (
        f"INSERT INTO chunks ({', '.join(CHUNK_COLS)}) "
        f"VALUES ({', '.join([ph] * len(CHUNK_COLS))}) "
        f"ON CONFLICT (chunk_id) DO UPDATE SET {updates}"
    )
    rows = [
        (c.meta.chunk_id, c.meta.doc_id, c.meta.section, c.chunk_index, c.n_tokens, c.text)
        for c in chunks
    ]
    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def delete_doc_chunks(conn, doc_id: str) -> int:
    cur = conn.cursor()
    cur.execute(f"DELETE FROM chunks WHERE doc_id = {placeholder()}", (doc_id,))
    conn.commit()
    return cur.rowcount


def count_doc_chunks(conn, doc_id: str) -> int:
    cur = conn.cursor()
    cur.execute(f"SELECT count(*) FROM chunks WHERE doc_id = {placeholder()}", (doc_id,))
    return cur.fetchone()[0]


def load_chunk_rows(conn, chunk_ids: list[str] | None = None,
                    doc_id: str | None = None) -> list[dict]:
    """Chunks joined to their document metadata. Full corpus when both args are None."""
    ph = placeholder()
    if chunk_ids is not None:
        if not chunk_ids:
            return []
        where = f" WHERE c.chunk_id IN ({', '.join([ph] * len(chunk_ids))})"
        params = tuple(chunk_ids)
    elif doc_id is not None:
        where, params = f" WHERE c.doc_id = {ph}", (doc_id,)
    else:
        where, params = "", ()
    cur = conn.cursor()
    cur.execute(_SELECT + where + " ORDER BY c.doc_id, c.chunk_index", params)
    return _dicts(cur)


def load_texts(conn, chunk_ids: list[str]) -> dict[str, str]:
    """chunk_id -> text, for hydrating retrieval hits that carry only ids + scores."""
    if not chunk_ids:
        return {}
    ph = placeholder()
    cur = conn.cursor()
    cur.execute(
        f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({', '.join([ph] * len(chunk_ids))})",
        tuple(chunk_ids),
    )
    return {r[0]: r[1] for r in cur.fetchall()}


def documents(conn, tickers: list[str] | None = None) -> list[dict]:
    """Manifest rows written by Phase 1 ingestion, oldest first."""
    ph = placeholder()
    sql = ("SELECT doc_id, ticker, company, doc_type, filing_date, fiscal_period, "
           "source_url, raw_path FROM documents")
    params: tuple = ()
    if tickers:
        sql += f" WHERE ticker IN ({', '.join([ph] * len(tickers))})"
        params = tuple(tickers)
    cur = conn.cursor()
    cur.execute(sql + " ORDER BY filing_date", params)
    rows = _dicts(cur)
    for r in rows:
        r["filing_date"] = as_date(r["filing_date"])
    return rows
