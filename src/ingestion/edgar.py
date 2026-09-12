"""SEC EDGAR ingestion.

Pulls the latest 10-K, last N 10-Qs, and recent earnings-related 8-Ks (Item
2.02 "Results of Operations and Financial Condition" — the free substitute
for paid earnings-call transcripts, see README) per ticker via the EDGAR
submissions API. Saves raw HTML to data/raw/{ticker}/{accession}.html and a
manifest row to the `documents` table.

Fair-access: SEC asks for a descriptive User-Agent with contact info and caps
automated traffic; both come from config.yaml and every request is
rate-limited. Idempotent: re-running skips any accession already present in
`documents`, so this is safe to schedule/re-run without a dedicated flow
runner.
"""
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.common.config import CFG
from src.common.db import get_conn, storage_mode

ROOT = Path(__file__).parents[2]
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{doc}"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{accession}-index.html"

EARNINGS_ITEM = "2.02"


class RateLimiter:
    """Simple fixed-interval limiter — good enough for a single-process crawler."""

    def __init__(self, max_rps: float):
        self.min_interval = 1.0 / max_rps
        self._last = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": CFG["edgar"]["user_agent"],
        "Accept-Encoding": "gzip, deflate",
    })
    return s


def _fetch(session: requests.Session, limiter: RateLimiter, url: str) -> requests.Response:
    limiter.wait()
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp


def _select_filings(recent: dict, n_10q: int, n_8k: int) -> list[dict]:
    """`recent` is submissions['filings']['recent'] — a dict of parallel arrays
    (one entry per index across accessionNumber/form/filingDate/...). Returns
    the filings we want to ingest, newest first within each form type."""
    n = len(recent["form"])
    rows = [{k: recent[k][i] for k in recent} for i in range(n)]
    rows.sort(key=lambda r: r["filingDate"], reverse=True)

    tenk = [r for r in rows if r["form"] == "10-K"][:1]
    tenq = [r for r in rows if r["form"] == "10-Q"][:n_10q]
    eightk = [
        r for r in rows
        if r["form"] == "8-K" and EARNINGS_ITEM in (r.get("items") or "")
    ][:n_8k]
    return tenk + tenq + eightk


def _doc_exists(conn, doc_id: str) -> bool:
    ph = "?" if storage_mode() == "local" else "%s"
    cur = conn.execute(f"SELECT 1 FROM documents WHERE doc_id = {ph}", (doc_id,))
    return cur.fetchone() is not None


def _insert_document(conn, row: dict):
    ph = "?" if storage_mode() == "local" else "%s"
    cols = ["doc_id", "ticker", "company", "doc_type", "filing_date",
            "fiscal_period", "source_url", "raw_path"]
    sql = f"INSERT INTO documents ({', '.join(cols)}) VALUES ({', '.join([ph] * len(cols))})"
    conn.execute(sql, tuple(row[c] for c in cols))


def _find_earnings_exhibit(session: requests.Session, limiter: RateLimiter, cik_int: int,
                            accession: str, accession_nodash: str) -> str | None:
    """An 8-K's primaryDocument is the Inline-XBRL cover form — mostly XBRL tag
    scaffolding, almost no narrative text. The earnings press release is
    filed as exhibit EX-99.1; prefer it when present. We read the EDGAR
    filing-index table's authoritative Type column rather than guessing from
    filenames, which vary by filer/law-firm convention."""
    index_url = INDEX_URL.format(cik_int=cik_int, accession_nodash=accession_nodash,
                                  accession=accession)
    resp = _fetch(session, limiter, index_url)
    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", class_="tableFile")
    if table is None:
        return None
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        if cells[3].get_text(strip=True).startswith("EX-99.1"):
            link = cells[2].find("a")
            return link["href"].rsplit("/", 1)[-1] if link and link.get("href") else None
    return None


def _fiscal_period(form: str, report_date: str) -> str:
    if not report_date:
        return ""
    year, month, _ = report_date.split("-")
    if form == "10-K":
        return f"FY{year}"
    q = (int(month) - 1) // 3 + 1
    return f"FY{year}Q{q}"


def ingest_ticker(ticker: str, cik: str, company: str, session: requests.Session,
                   limiter: RateLimiter, conn) -> int:
    resp = _fetch(session, limiter, SUBMISSIONS_URL.format(cik=cik))
    data = resp.json()
    filings = _select_filings(data["filings"]["recent"], CFG["edgar"]["n_10q"], CFG["edgar"]["n_8k"])

    cik_int = int(cik)
    raw_dir = ROOT / "data" / "raw" / ticker
    raw_dir.mkdir(parents=True, exist_ok=True)

    n_new = 0
    for f in filings:
        accession = f["accessionNumber"]
        if _doc_exists(conn, accession):
            continue

        accession_nodash = accession.replace("-", "")
        doc_name = f["primaryDocument"]
        if f["form"] == "8-K":
            exhibit = _find_earnings_exhibit(session, limiter, cik_int, accession, accession_nodash)
            if exhibit:
                doc_name = exhibit
        doc_url = ARCHIVES_URL.format(cik_int=cik_int, accession_nodash=accession_nodash, doc=doc_name)
        html_resp = _fetch(session, limiter, doc_url)

        raw_path = raw_dir / f"{accession}.html"
        raw_path.write_bytes(html_resp.content)

        _insert_document(conn, {
            "doc_id": accession,
            "ticker": ticker,
            "company": company,
            "doc_type": f["form"],
            "filing_date": f["filingDate"],
            "fiscal_period": _fiscal_period(f["form"], f.get("reportDate", "")),
            "source_url": doc_url,
            "raw_path": str(raw_path.relative_to(ROOT)),
        })
        conn.commit()
        n_new += 1
        print(f"  {ticker} {f['form']} {accession} ({f['filingDate']}) -> {raw_path.name}")
    return n_new


def ingest_all() -> int:
    session = _session()
    limiter = RateLimiter(CFG["edgar"]["max_rps"])
    conn = get_conn()
    total = 0
    try:
        for ticker, info in CFG["tickers"].items():
            print(f"ingesting {ticker} ({info['company']})...")
            total += ingest_ticker(ticker, info["cik"], info["company"], session, limiter, conn)
    finally:
        conn.close()
    return total


# --- public fetch seam -------------------------------------------------------
# Phase 4 (companyfacts XBRL) talks to the same host under the same fair-access
# rules, so it reuses this session/limiter rather than opening a second unthrottled
# path to data.sec.gov. SEC rate-limits per IP, not per script.

def sec_session() -> tuple[requests.Session, RateLimiter]:
    """A configured session plus its rate limiter, for any data.sec.gov caller."""
    return _session(), RateLimiter(CFG["edgar"]["max_rps"])


def sec_get_json(url: str, session: requests.Session | None = None,
                 limiter: RateLimiter | None = None) -> dict:
    """Rate-limited GET returning parsed JSON. Opens its own session if not given."""
    if session is None or limiter is None:
        session, limiter = sec_session()
    return _fetch(session, limiter, url).json()
