"""Forward guidance text, extracted from the 8-K earnings releases already ingested.

Guidance is not in XBRL. Companyfacts holds what *happened*; the outlook for next
quarter is narrative prose in the EX-99.1 press release (DECISIONS #10), so this reads
the chunks Phase 2 already indexed rather than fetching anything new.

Extraction is deliberately rule-based, not LLM: an outlook block in an earnings
release is formulaic ("NVIDIA's outlook for the third quarter of fiscal 2027 is as
follows:"), the period being guided is stated in that sentence, and a regex that
either matches or doesn't is auditable in a way a generated extraction isn't. When the
pattern misses, the row is absent — which the answer layer reports honestly — rather
than being a confident paraphrase of the wrong paragraph.

Coverage is a real limitation and is not evenly distributed: NVDA and MSFT publish
explicit outlook sections; JPM does not issue revenue guidance in this form at all, so
it has no rows here. That asymmetry is the honest state of the source data.
"""
import re

from src.common.db import storage_mode
from src.indexing.store import as_date, placeholder

# "outlook for the third quarter of fiscal 2027", "for fiscal year 2026", "for Q3 FY26"
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
PERIOD_RE = re.compile(
    r"(?:for\s+)?(?:the\s+)?(?:(first|second|third|fourth)\s+quarter|Q([1-4]))\s+"
    r"of\s+(?:fiscal\s+(?:year\s+)?)?(\d{4})", re.I)
FY_RE = re.compile(r"(?:full[\s-]?year|fiscal\s+year)\s+(?:fiscal\s+)?(\d{4})", re.I)

# The block header. Anchored to a line start so a passing mention of the word
# "outlook" mid-paragraph doesn't open a guidance block.
OUTLOOK_RE = re.compile(r"(?im)^\s*(outlook|financial outlook|business outlook|guidance)\b")

MAX_CHARS = 2000


def parse_period(text: str) -> str | None:
    """Fiscal period the guidance is *for*, e.g. 'FY2027Q3'. None when unstated.

    Note this is the guided period, not the reporting period — a Q2 release guides Q3,
    so storing the release's own fiscal_period here would misdate every row by a
    quarter.
    """
    m = PERIOD_RE.search(text)
    if m:
        q = ORDINALS[m.group(1).lower()] if m.group(1) else int(m.group(2))
        return f"FY{m.group(3)}Q{q}"
    m = FY_RE.search(text)
    return f"FY{m.group(1)}" if m else None


def extract_from_text(text: str) -> tuple[str, str | None] | None:
    """(guidance block, guided period) from one chunk, or None if it has no outlook."""
    m = OUTLOOK_RE.search(text)
    if not m:
        return None
    block = text[m.start():m.start() + MAX_CHARS].strip()
    return block, parse_period(block)


def candidate_chunks(conn, ticker: str | None = None) -> list[dict]:
    """8-K chunks that might carry an outlook block, newest filing first."""
    ph = placeholder()
    sql = ("SELECT c.chunk_id, c.text, d.doc_id, d.ticker, d.filing_date "
           "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
           "WHERE d.doc_type = '8-K'")
    params: tuple = ()
    if ticker:
        sql += f" AND d.ticker = {ph}"
        params = (ticker,)
    sql += " ORDER BY d.filing_date DESC, c.chunk_index"
    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def write_guidance(conn, rows: list[dict]) -> int:
    """Upsert on (ticker, fiscal_period, given_on) — the natural key from the schema.

    `source_doc_id` is safe to set here, unlike in `financials`: these rows come from
    filings we ingested by definition, so the FK always resolves.
    """
    if not rows:
        return 0
    ph = placeholder()
    cols = ["ticker", "fiscal_period", "given_on", "text", "source_doc_id"]
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in ("text", "source_doc_id"))
    sql = (f"INSERT INTO guidance ({', '.join(cols)}) "
           f"VALUES ({', '.join([ph] * len(cols))}) "
           f"ON CONFLICT (ticker, fiscal_period, given_on) DO UPDATE SET {updates}")
    cur = conn.cursor()
    cur.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
    conn.commit()
    return len(rows)


def extract_all(conn, ticker: str | None = None) -> list[dict]:
    """Guidance rows from every ingested 8-K. One row per (filing, guided period).

    Chunks overlap by design (Phase 2), so the same outlook block appears in two
    consecutive chunks. Keyed by (doc_id, period) with the longest block winning, so
    the fuller of the two overlapping copies is the one kept.
    """
    best: dict[tuple[str, str], dict] = {}
    for row in candidate_chunks(conn, ticker):
        found = extract_from_text(row["text"])
        if not found:
            continue
        block, period = found
        if period is None:
            continue
        key = (row["doc_id"], period)
        if key not in best or len(block) > len(best[key]["text"]):
            best[key] = {
                "ticker": row["ticker"],
                "fiscal_period": period,
                "given_on": as_date(row["filing_date"]),
                "text": block,
                "source_doc_id": row["doc_id"],
            }
    return list(best.values())


def load_all(conn, ticker: str | None = None) -> int:
    return write_guidance(conn, extract_all(conn, ticker))


def latest_guidance(conn, ticker: str, limit: int = 3) -> list[dict]:
    """Most recently issued guidance for a ticker, newest first."""
    ph = placeholder()
    cur = conn.cursor()
    cur.execute(
        "SELECT ticker, fiscal_period, given_on, text, source_doc_id FROM guidance "
        f"WHERE ticker = {ph} ORDER BY given_on DESC LIMIT {int(limit)}", (ticker,))
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        r["given_on"] = as_date(r["given_on"])
    return rows
