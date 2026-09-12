"""Canned, parameterized queries over `financials`. The LLM never writes SQL in v1.

The router (Phase 6) picks a query *name* from `QUERIES` and supplies typed params;
the SQL lives here, fully parameterized. Text-to-SQL is a documented v2 item and is
deferred for two reasons worth stating plainly: injection risk (a generated string
reaching the DB is an attack surface no prompt can close), and eval complexity (a
wrong-but-plausible query returns a number, not an error, so you cannot tell a
correct answer from a lucky one without a second grader).

The correctness trap this module exists to handle: **the series has holes.** Q4 EPS is
never derived (diluted share counts move, so annual EPS is not the sum of quarters —
see financials.ADDITIVE), and a filer can simply not report a metric. `ORDER BY
fiscal_year DESC, fiscal_qtr DESC LIMIT 4` then returns four rows that *look*
contiguous but skip the gap, and any growth computed by pairing row N with row N+1
silently compares Q3 to Q1. So every growth figure here is computed against the
*named* prior period — `prev_quarter` / `year_ago_quarter` — and is None when that
period is genuinely absent, rather than against whatever row happened to come next.

Values come back as float. Postgres NUMERIC yields Decimal and SQLite REAL yields
float; normalizing at the boundary means downstream arithmetic can't raise
TypeError on one backend and work on the other.
"""
from dataclasses import dataclass
from datetime import date

from src.indexing.store import as_date, placeholder

METRICS = ("revenue", "eps_diluted", "operating_income", "gross_margin")

_SELECT = ("SELECT ticker, metric, fiscal_year, fiscal_qtr, value, unit, period_end, "
           "source_accn, source_doc_id, derived FROM financials")


@dataclass
class MetricPoint:
    ticker: str
    metric: str
    fiscal_year: int
    fiscal_qtr: int
    value: float
    unit: str
    period_end: date | None
    source_accn: str | None
    source_doc_id: str | None
    derived: bool

    @property
    def period(self) -> str:
        return f"FY{self.fiscal_year}" + (f"Q{self.fiscal_qtr}" if self.fiscal_qtr else "")

    @property
    def key(self) -> tuple[int, int]:
        return (self.fiscal_year, self.fiscal_qtr)

    def format(self) -> str:
        """Human-readable value, unit-aware. Used by the generation layer."""
        if self.unit == "usd":
            return f"${self.value / 1e9:,.2f}B"
        if self.unit == "usd_per_share":
            return f"${self.value:,.2f}"
        if self.unit == "pct":
            return f"{self.value:.1f}%"
        return f"{self.value:,.2f}"


@dataclass
class GrowthPoint:
    """A period paired with its named comparison period.

    `prior` is None when that period is absent from the table — the caller must render
    that as "not comparable", never as 0% or as growth against a different quarter.
    """
    point: MetricPoint
    prior: MetricPoint | None
    basis: str                      # QoQ | YoY

    @property
    def pct_change(self) -> float | None:
        if self.prior is None or not self.prior.value:
            return None
        return 100.0 * (self.point.value - self.prior.value) / abs(self.prior.value)

    @property
    def comparable(self) -> bool:
        return self.pct_change is not None

    def format(self) -> str:
        if not self.comparable:
            return f"{self.point.period}: {self.point.format()} ({self.basis} n/a — no prior period)"
        return (f"{self.point.period}: {self.point.format()} "
                f"({self.basis} {self.pct_change:+.1f}% vs {self.prior.period})")


# --- period arithmetic -------------------------------------------------------

def prev_quarter(fy: int, q: int) -> tuple[int, int]:
    return (fy - 1, 4) if q == 1 else (fy, q - 1)


def year_ago_quarter(fy: int, q: int) -> tuple[int, int]:
    return (fy - 1, q)


# --- row plumbing ------------------------------------------------------------

def _row_to_point(row) -> MetricPoint:
    return MetricPoint(
        ticker=row[0], metric=row[1], fiscal_year=int(row[2]), fiscal_qtr=int(row[3]),
        value=float(row[4]), unit=row[5], period_end=as_date(row[6]),
        source_accn=row[7], source_doc_id=row[8], derived=bool(row[9]),
    )


def _fetch(conn, where: str, params: tuple, order: str = "", limit: int | None = None):
    sql = f"{_SELECT} WHERE {where} {order}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"          # int() — never a caller string
    cur = conn.cursor()
    cur.execute(sql, params)
    return [_row_to_point(r) for r in cur.fetchall()]


# --- canned queries ----------------------------------------------------------

def metric_series(conn, ticker: str, metric: str, n_quarters: int = 4) -> list[MetricPoint]:
    """Last N *quarterly* points for one metric, newest first.

    Full-year rows (fiscal_qtr = 0) are excluded: mixing an annual total into a
    quarterly series is the same category of error as mixing a YTD XBRL fact into one.
    """
    ph = placeholder()
    return _fetch(conn,
                  f"ticker = {ph} AND metric = {ph} AND fiscal_qtr > 0",
                  (ticker, metric),
                  "ORDER BY fiscal_year DESC, fiscal_qtr DESC",
                  n_quarters)


def revenue_last_n_quarters(conn, ticker: str, n_quarters: int = 4) -> list[MetricPoint]:
    """The Definition-of-Done query: 'how has revenue changed over the last 4 quarters'."""
    return metric_series(conn, ticker, "revenue", n_quarters)


def latest_metric(conn, ticker: str, metric: str) -> MetricPoint | None:
    points = metric_series(conn, ticker, metric, 1)
    return points[0] if points else None


def _point_at(conn, ticker: str, metric: str, key: tuple[int, int]) -> MetricPoint | None:
    ph = placeholder()
    rows = _fetch(conn, f"ticker = {ph} AND metric = {ph} AND fiscal_year = {ph} "
                        f"AND fiscal_qtr = {ph}", (ticker, metric, key[0], key[1]))
    return rows[0] if rows else None


def _growth(conn, ticker: str, metric: str, n_quarters: int, basis: str,
            shift) -> list[GrowthPoint]:
    """Pair each of the last N points with its *named* comparison period."""
    out = []
    for p in metric_series(conn, ticker, metric, n_quarters):
        prior = _point_at(conn, ticker, metric, shift(p.fiscal_year, p.fiscal_qtr))
        out.append(GrowthPoint(point=p, prior=prior, basis=basis))
    return out


def qoq_growth(conn, ticker: str, metric: str = "revenue",
               n_quarters: int = 4) -> list[GrowthPoint]:
    return _growth(conn, ticker, metric, n_quarters, "QoQ", prev_quarter)


def yoy_growth(conn, ticker: str, metric: str = "revenue",
               n_quarters: int = 4) -> list[GrowthPoint]:
    """Year-over-year. The meaningful one for seasonal businesses — MSFT's Q2 is a
    December quarter and JPM's Q4 carries year-end items, so QoQ on either compares
    two structurally different quarters."""
    return _growth(conn, ticker, metric, n_quarters, "YoY", year_ago_quarter)


def margin_trend(conn, ticker: str, n_quarters: int = 8) -> list[MetricPoint]:
    """Gross margin over time. Empty for filers that report no gross profit (banks) —
    callers must say 'not reported', not 'zero'. See available_metrics()."""
    return metric_series(conn, ticker, "gross_margin", n_quarters)


def compare_tickers(conn, tickers: list[str], metric: str = "revenue",
                    n_quarters: int = 1) -> dict[str, list[MetricPoint]]:
    """Same metric across tickers, for CROSS_SOURCE questions.

    Caveat the answer layer must surface: fiscal calendars differ, so NVDA's FY2026Q3
    and MSFT's FY2026Q3 are not the same three months. `period_end` is carried on every
    point precisely so a comparison can be dated honestly.
    """
    return {t: metric_series(conn, t, metric, n_quarters) for t in tickers}


def available_metrics(conn, ticker: str) -> list[str]:
    """Which metrics this filer actually reports.

    JPM reports no gross margin and no operating income — a bank has no cost of
    revenue. The router asks this before promising a number, so an unreportable metric
    becomes an explicit 'not reported by this filer' instead of an empty result the
    generator might read as zero.
    """
    ph = placeholder()
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT metric FROM financials WHERE ticker = {ph}", (ticker,))
    return sorted(r[0] for r in cur.fetchall())


def coverage(conn) -> list[dict]:
    """Per ticker/metric coverage summary — powers the smoke test and the UI sidebar."""
    cur = conn.cursor()
    cur.execute("SELECT ticker, metric, count(*), min(fiscal_year), max(fiscal_year), "
                "sum(CASE WHEN derived THEN 1 ELSE 0 END) FROM financials "
                "GROUP BY ticker, metric ORDER BY ticker, metric")
    return [{"ticker": r[0], "metric": r[1], "n": int(r[2]), "from_fy": int(r[3]),
             "to_fy": int(r[4]), "derived": int(r[5])} for r in cur.fetchall()]


# --- registry the router selects from ----------------------------------------
# Name -> (callable, required params, description). The LLM picks a name and fills
# params; it never sees or writes SQL. Keeping the specs as data makes the router's
# choice testable and lets the UI list what the NUMERIC route can actually answer.

QUERIES: dict[str, dict] = {
    "metric_series": {
        "fn": metric_series, "params": {"ticker": "str", "metric": METRICS,
                                        "n_quarters": "int"},
        "description": "Last N quarters of one metric, newest first.",
    },
    "revenue_last_n_quarters": {
        "fn": revenue_last_n_quarters, "params": {"ticker": "str", "n_quarters": "int"},
        "description": "Revenue for the last N quarters.",
    },
    "latest_metric": {
        "fn": latest_metric, "params": {"ticker": "str", "metric": METRICS},
        "description": "Most recent reported value of one metric.",
    },
    "qoq_growth": {
        "fn": qoq_growth, "params": {"ticker": "str", "metric": METRICS, "n_quarters": "int"},
        "description": "Quarter-over-quarter % change against the named prior quarter.",
    },
    "yoy_growth": {
        "fn": yoy_growth, "params": {"ticker": "str", "metric": METRICS, "n_quarters": "int"},
        "description": "Year-over-year % change against the same quarter last year.",
    },
    "margin_trend": {
        "fn": margin_trend, "params": {"ticker": "str", "n_quarters": "int"},
        "description": "Gross margin % over the last N quarters.",
    },
    "compare_tickers": {
        "fn": compare_tickers, "params": {"tickers": "list[str]", "metric": METRICS,
                                          "n_quarters": "int"},
        "description": "One metric across several tickers (note: fiscal calendars differ).",
    },
    "available_metrics": {
        "fn": available_metrics, "params": {"ticker": "str"},
        "description": "Which metrics this filer reports at all.",
    },
}


def run(conn, name: str, **params):
    """Execute a canned query by name. Raises on an unknown name or an unknown param,
    so a router hallucinating either fails loudly instead of silently doing nothing."""
    if name not in QUERIES:
        raise ValueError(f"unknown query {name!r}; expected one of {sorted(QUERIES)}")
    spec = QUERIES[name]
    unexpected = set(params) - set(spec["params"])
    if unexpected:
        raise ValueError(f"{name}: unexpected param(s) {sorted(unexpected)}; "
                         f"accepts {sorted(spec['params'])}")
    return spec["fn"](conn, **params)
