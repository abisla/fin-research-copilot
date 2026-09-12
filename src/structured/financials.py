"""Quarterly financials from the SEC companyfacts XBRL API.

    https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

Free, structured, and full of traps. Every one below was verified against the real
NVDA/MSFT/JPM payloads before it was coded; see DECISIONS #19.

**1. One tag holds several period lengths.** `Revenues` for NVDA contains 167
three-month facts, 37 six-month, 34 nine-month and 42 twelve-month facts, all mixed
together. Loading the tag naively puts Q1 next to a nine-month cumulative number and
calls both "quarterly revenue". Period length is derived from `start`/`end` and is the
first filter applied — nothing reaches the DB without being classified.

**2. `fy`/`fp` on a fact describe the FILING, not the fact.** NVDA's quarter ending
2010-05-02 appears four times carrying (fy=2010,fp=Q2), (2010,FY), (2011,Q1) and
(2011,FY) — four different labels for one period, because a fact is re-reported as a
comparative in every later filing that shows it. So fiscal period is derived from the
fact's own `end` date against the company's fiscal-year-end month, never read off
`fy`/`fp`.

**3. Fiscal calendars are not calendar years.** NVDA's FY2026 ended 2026-01-25, MSFT's
FY2026 ends 2026-06-30, JPM's is calendar. The FYE month is inferred from the company's
own annual facts rather than hardcoded, and 52/53-week filers (NVDA ends on a Sunday,
so quarter ends drift by days each year) are handled because only the *month* is used.

**4. The same period is reported many times.** 103 of 121 NVDA revenue periods have
more than one fact. Latest `filed` wins: that is the number the company currently
stands behind, including restatements.

**5. Tags are industry-specific.** NVDA reports `Revenues`; MSFT reports
`RevenueFromContractWithCustomerExcludingAssessedTax`; JPM, a bank, reports neither as
its top line and has no `GrossProfit`, no `CostOfRevenue` and no `OperatingIncomeLoss`
at all. Each metric is therefore an ordered *chain* of candidate tags, and a metric a
company genuinely does not report is left absent rather than back-filled from a
near-synonym that means something else.

**6. Q4 is usually never filed.** 10-Qs cover Q1-Q3 and the 10-K reports the year, so
Q4 must be derived — FY minus the nine-month cumulative (preferred: one subtraction) or
FY minus Q1+Q2+Q3. JPM files zero standalone Q4 revenue facts. Derivation is applied
only to additive flow metrics; see `ADDITIVE`.
"""
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.common.config import CFG
from src.common.db import get_conn, storage_mode

ROOT = Path(__file__).parents[2]
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Ordered candidate tags per metric. First tag with usable coverage wins; a later tag
# is a fallback, never a merge — mixing two revenue concepts in one series produces a
# chart with an invisible definitional break in it.
TAG_CHAINS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",   # MSFT, post-ASC606 filers
        "Revenues",                                              # NVDA
        "RevenuesNetOfInterestExpense",                          # JPM and banks: "total net revenue"
    ],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "operating_income": ["OperatingIncomeLoss"],                 # absent for banks
    "gross_profit": ["GrossProfit"],                             # absent for banks
}

UNITS = {"revenue": "usd", "eps_diluted": "usd_per_share", "operating_income": "usd",
         "gross_profit": "usd", "gross_margin": "pct"}

# Metrics where a period total is the sum of its parts, so Q4 = FY - 9M holds.
# EPS is deliberately excluded: diluted share count changes quarter to quarter, so
# annual EPS is not the sum of quarterly EPS and a subtracted Q4 would be subtly wrong.
ADDITIVE = {"revenue", "operating_income", "gross_profit"}

FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "8-K"}

# Period-length windows in days. Deliberately loose: 52/53-week fiscal calendars make
# a "quarter" anywhere from 84 to 98 days, and a "year" 364 or 371.
PERIOD_WINDOWS = {"Q": (80, 100), "H": (170, 190), "9M": (260, 285), "FY": (340, 380)}


@dataclass(frozen=True)
class Fact:
    """One XBRL fact, already classified and dated to a fiscal period."""
    fiscal_year: int
    fiscal_qtr: int          # 1-4, or 0 for the full year
    value: float
    period_end: date
    accn: str
    filed: date
    derived: bool = False


def cache_path(ticker: str) -> Path:
    return ROOT / "data" / "raw" / ticker / "companyfacts.json"


def fetch_companyfacts(ticker: str, cik: str, refresh: bool = False,
                       session=None, limiter=None) -> dict:
    """Companyfacts JSON, cached on disk (4-9 MB per ticker — don't re-pull casually)."""
    path = cache_path(ticker)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    from src.ingestion.edgar import sec_get_json          # lazy: keeps `requests` off the query path
    data = sec_get_json(COMPANYFACTS_URL.format(cik=cik), session, limiter)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def period_kind(start: date, end: date) -> str | None:
    """Classify a fact's duration. None = a period length we don't model."""
    n = (end - start).days
    for kind, (lo, hi) in PERIOD_WINDOWS.items():
        if lo <= n <= hi:
            return kind
    return None


def fiscal_year_end_month(gaap: dict) -> int:
    """Infer the company's fiscal-year-end month from its own annual facts.

    Taken as the modal end-month across annual-length facts of whichever revenue tag
    the company actually uses, which beats hardcoding a calendar per ticker and beats
    `dei:CurrentFiscalYearEndDate` (a "--MM-DD" string that some filers leave stale).
    """
    months: Counter = Counter()
    for tag in TAG_CHAINS["revenue"] + ["NetIncomeLoss"]:
        for fact in _raw_facts(gaap, tag):
            if fact["kind"] == "FY":
                months[fact["end"].month] += 1
    if not months:
        raise ValueError("no annual facts — cannot infer fiscal year end")
    return months.most_common(1)[0][0]


def fiscal_period(end: date, fye_month: int) -> tuple[int, int]:
    """(fiscal_year, fiscal_qtr) for a period ending on `end`.

    Fiscal year is named for the calendar year it ends in — NVDA's year ending
    2026-01-25 is FY2026, MSFT's ending 2025-06-30 is FY2025, JPM's is the calendar
    year. Verified against all three tickers' own `fp`/`fy` labels on the filings that
    originated each fact.
    """
    fy = end.year if end.month <= fye_month else end.year + 1
    qtr = ((end.month - fye_month - 1) % 12) // 3 + 1
    return fy, qtr


def _raw_facts(gaap: dict, tag: str) -> list[dict]:
    """Every duration fact for a tag, parsed and classified. Instants are skipped."""
    node = gaap.get(tag)
    if not node:
        return []
    out = []
    for unit, facts in node["units"].items():
        for f in facts:
            if not f.get("start") or f.get("form") not in FORMS:
                continue
            start, end = date.fromisoformat(f["start"]), date.fromisoformat(f["end"])
            kind = period_kind(start, end)
            if kind is None:
                continue
            out.append({"kind": kind, "end": end, "val": float(f["val"]), "unit": unit,
                        "accn": f["accn"], "filed": date.fromisoformat(f["filed"])})
    return out


def _pick_latest(facts: list[dict], fye_month: int, kind: str) -> dict[tuple[int, int], Fact]:
    """Collapse facts of one period length to one Fact per fiscal period.

    Latest `filed` wins, so a restatement supersedes the original and a 10-Q supersedes
    the 8-K earnings release that preceded it. Ties (same filing date, different
    accessions) break on accession for determinism.
    """
    best: dict[tuple[int, int], dict] = {}
    for f in facts:
        if f["kind"] != kind:
            continue
        fy, q = fiscal_period(f["end"], fye_month)
        key = (fy, 0 if kind != "Q" else q)
        cur = best.get(key)
        if cur is None or (f["filed"], f["accn"]) > (cur["filed"], cur["accn"]):
            best[key] = f
    return {k: Fact(k[0], k[1], f["val"], f["end"], f["accn"], f["filed"]) for k, f in best.items()}


# A tag is "current" if it is still being filed near the end of the available data.
# Deliberately generous: a filer that changed concepts mid-history keeps reporting the
# old tag as a comparative for a year or two afterwards.
RECENT_YEARS = 5


def select_tag(gaap: dict, metric: str, fye_month: int) -> tuple[str | None, dict]:
    """Pick the one tag in a metric's chain that the company actually reports.

    Chain order alone is not enough, because every candidate is a *real* tag with real
    facts — they are just from different eras. ASC 606 moved most filers onto
    `RevenueFromContractWithCustomerExcludingAssessedTax` in 2018; NVDA stayed on
    `Revenues`; JPM moved to the bank-specific `RevenuesNetOfInterestExpense` in 2014.
    Taking the first chain entry with *any* coverage picks whichever era happens to be
    listed first and silently truncates the series (NVDA revenue collapsed to
    FY2018-FY2020, JPM to FY2008-FY2014 — both wrong, both plausible-looking).

    So tags are scored on coverage instead: quarters filed in the last RECENT_YEARS
    fiscal years first, total quarters as the tie-break, latest period as the final
    one. The anchor is the newest period across the candidates rather than today's
    date, so the choice is reproducible from the data alone.

    Returns (winning tag, its quarterly series). (None, {}) when the company reports
    no tag in the chain — a bank has no gross profit, and that absence is the answer.
    """
    candidates = []
    for tag in TAG_CHAINS[metric]:
        raw = _raw_facts(gaap, tag)
        quarters = _pick_latest(raw, fye_month, "Q")
        if quarters:
            candidates.append((tag, raw, quarters))
    if not candidates:
        return None, {}

    anchor = max(f.period_end for _, _, qs in candidates for f in qs.values())
    cutoff = anchor.year - RECENT_YEARS

    def score(item):
        _, _, qs = item
        return (sum(1 for f in qs.values() if f.period_end.year >= cutoff),
                len(qs),
                max(f.period_end for f in qs.values()))

    tag, raw, quarters = max(candidates, key=score)
    if metric in ADDITIVE:
        quarters = quarters | _derive_q4(quarters, _pick_latest(raw, fye_month, "FY"),
                                         _pick_latest(raw, fye_month, "9M"))
    return tag, quarters


def extract_metric(gaap: dict, metric: str, fye_month: int) -> dict[tuple[int, int], Fact]:
    """Quarterly series for one metric, Q4 derived where the filer omits it."""
    return select_tag(gaap, metric, fye_month)[1]


def _derive_q4(quarters: dict[tuple[int, int], Fact], annual: dict[tuple[int, int], Fact],
               ytd9: dict[tuple[int, int], Fact]) -> dict[tuple[int, int], Fact]:
    """Q4 = FY - 9M, falling back to FY - (Q1+Q2+Q3). Only fills gaps.

    FY minus the nine-month cumulative is preferred because it is one subtraction
    between two numbers from the same filing family, where summing three quarters
    accumulates any restatement mismatch between them.
    """
    out: dict[tuple[int, int], Fact] = {}
    for (fy, _), fy_fact in annual.items():
        if (fy, 4) in quarters:
            continue
        nine = ytd9.get((fy, 0))
        if nine is not None:
            value = fy_fact.value - nine.value
        else:
            parts = [quarters.get((fy, q)) for q in (1, 2, 3)]
            if any(p is None for p in parts):
                continue
            value = fy_fact.value - sum(p.value for p in parts)
        out[(fy, 4)] = Fact(fy, 4, value, fy_fact.period_end, fy_fact.accn, fy_fact.filed,
                            derived=True)
    return out


def gross_margin(revenue: dict[tuple[int, int], Fact],
                 profit: dict[tuple[int, int], Fact]) -> dict[tuple[int, int], Fact]:
    """Gross margin % per quarter. No XBRL tag exists for it — it is a ratio.

    Only emitted where both components are present for the same period, and attributed
    to the gross-profit filing. Banks report no gross profit at all, so this series is
    legitimately empty for JPM rather than zero-filled.
    """
    out = {}
    for key, gp in profit.items():
        rev = revenue.get(key)
        if rev is None or not rev.value:
            continue
        out[key] = Fact(key[0], key[1], 100.0 * gp.value / rev.value, gp.period_end,
                        gp.accn, gp.filed, derived=True)
    return out


def build_rows(ticker: str, gaap: dict) -> tuple[list[dict], dict[str, str | None]]:
    """Every financials row for one ticker, plus the tag chosen per metric.

    The chosen tag is returned rather than logged and forgotten: which us-gaap concept
    a series came from is the single most important thing to be able to audit later,
    because two tags can both be "revenue" and mean different things.
    """
    fye = fiscal_year_end_month(gaap)
    chosen = {m: select_tag(gaap, m, fye) for m in TAG_CHAINS}
    tags = {m: t for m, (t, _) in chosen.items()}
    series = {m: q for m, (_, q) in chosen.items()}
    series["gross_margin"] = gross_margin(series["revenue"], series["gross_profit"])
    series.pop("gross_profit")          # a component, not a reported metric

    rows = []
    for metric, facts in series.items():
        for f in facts.values():
            rows.append({
                "ticker": ticker, "fiscal_year": f.fiscal_year, "fiscal_qtr": f.fiscal_qtr,
                "metric": metric, "value": round(f.value, 4), "unit": UNITS[metric],
                "period_end": f.period_end, "source_accn": f.accn, "derived": f.derived,
            })
    return rows, tags


# --- persistence -------------------------------------------------------------

COLS = ["ticker", "fiscal_year", "fiscal_qtr", "metric", "value", "unit",
        "period_end", "source_accn", "source_doc_id", "derived"]


def ensure_schema(conn) -> list[str]:
    """Add the provenance columns to a `financials` table created before Phase 4.

    Idempotent, and mode-aware because SQLite has no ADD COLUMN IF NOT EXISTS. Cheaper
    and safer than asking the owner to drop and re-init a database that already holds
    3,456 chunks.
    """
    added = []
    wanted = {"period_end": ("DATE", "TEXT"), "source_accn": ("TEXT", "TEXT"),
              "derived": ("BOOLEAN DEFAULT FALSE", "INTEGER DEFAULT 0")}
    cur = conn.cursor()
    if storage_mode() == "local":
        existing = {r[1] for r in cur.execute("PRAGMA table_info(financials)").fetchall()}
    else:
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'financials'")
        existing = {r[0] for r in cur.fetchall()}
    for col, (pg_type, lite_type) in wanted.items():
        if col not in existing:
            ddl = lite_type if storage_mode() == "local" else pg_type
            cur.execute(f"ALTER TABLE financials ADD COLUMN {col} {ddl}")
            added.append(col)
    conn.commit()
    return added


def known_doc_ids(conn) -> set[str]:
    cur = conn.cursor()
    cur.execute("SELECT doc_id FROM documents")
    return {r[0] for r in cur.fetchall()}


def write_financials(conn, rows: list[dict]) -> int:
    """Replace each (ticker, metric) series wholesale, in one transaction.

    Upsert alone is not enough here, and the reason is worth keeping. Upsert makes
    *re-running* idempotent, but it cannot retire a row the new run no longer produces
    — and `select_tag` can legitimately change its mind. When JPM's revenue tag was
    corrected from `Revenues` (FY2008-FY2014) to `RevenuesNetOfInterestExpense`
    (FY2014-FY2026), the upsert left both eras in the table: one continuous-looking
    revenue series with a silent definitional break in 2014, which is exactly the
    failure the tag chain exists to prevent. Delete-then-insert per series makes this
    load authoritative for what it produces, because companyfacts always yields the
    complete history — there is no incremental case to preserve.

    `source_doc_id` is a FK into `documents`, which holds only the filings Phase 1
    ingested — roughly 5 of the ~70 accessions companyfacts cites. Setting it blindly
    would violate the constraint, so it is populated only when we actually hold that
    filing, and `source_accn` carries the true provenance in every row regardless.
    """
    if not rows:
        return 0
    ph = "?" if storage_mode() == "local" else "%s"
    known = known_doc_ids(conn)
    cur = conn.cursor()

    for ticker, metric in sorted({(r["ticker"], r["metric"]) for r in rows}):
        cur.execute(f"DELETE FROM financials WHERE ticker = {ph} AND metric = {ph}",
                    (ticker, metric))

    sql = (f"INSERT INTO financials ({', '.join(COLS)}) "
           f"VALUES ({', '.join([ph] * len(COLS))})")
    params = [
        (r["ticker"], r["fiscal_year"], r["fiscal_qtr"], r["metric"], r["value"], r["unit"],
         r["period_end"], r["source_accn"],
         r["source_accn"] if r["source_accn"] in known else None, r["derived"])
        for r in rows
    ]
    cur.executemany(sql, params)
    conn.commit()
    return len(params)


def purge_ticker(conn, ticker: str) -> int:
    """Drop every financials row for a ticker — for a metric a filer has stopped
    reporting entirely, which delete-then-insert can't reach (it writes no rows)."""
    ph = "?" if storage_mode() == "local" else "%s"
    cur = conn.cursor()
    cur.execute(f"DELETE FROM financials WHERE ticker = {ph}", (ticker,))
    conn.commit()
    return cur.rowcount


def load_ticker(conn, ticker: str, cik: str, refresh: bool = False,
                session=None, limiter=None) -> tuple[int, dict]:
    """Fetch, parse and persist one ticker.

    Returns (rows written, per-metric row counts, chosen us-gaap tag per metric).
    """
    data = fetch_companyfacts(ticker, cik, refresh, session, limiter)
    rows, tags = build_rows(ticker, data["facts"]["us-gaap"])
    n = write_financials(conn, rows)
    return n, dict(Counter(r["metric"] for r in rows)), tags


def load_all(refresh: bool = False) -> int:
    from src.ingestion.edgar import sec_session
    session, limiter = sec_session()
    conn = get_conn()
    total = 0
    try:
        ensure_schema(conn)
        for ticker, info in CFG["tickers"].items():
            n, counts, tags = load_ticker(conn, ticker, info["cik"], refresh, session, limiter)
            total += n
            print(f"  {ticker:5s} {n:4d} rows  {counts}")
            print(f"        tags: {tags}")
    finally:
        conn.close()
    return total
