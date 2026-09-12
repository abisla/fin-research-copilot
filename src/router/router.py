"""Query router: rules first, LLM fallback.

Routes: NEWS | FILING_RAG | NUMERIC | MIXED | CROSS_SOURCE

Rules catch the cheap, unambiguous cases (fast, free, deterministic, testable).
LLM fallback handles genuinely ambiguous phrasing.
Interview q: "why not LLM for everything?" -> latency + cost + non-determinism
in eval; rules give you a testable contract for the 80% case.

**The routes are defined by which store answers the question**, not by topic — that
is what keeps them decidable rather than a matter of taste:

* `NUMERIC`      — `financials` alone. "What was NVDA revenue last quarter."
* `FILING_RAG`   — filing chunks alone. "What risk factors does MSFT disclose."
* `NEWS`         — `news_articles` alone. "What happened to JPM last week."
* `MIXED`        — numbers *and* narrative for ONE ticker. The Definition-of-Done
                   query, "how has revenue changed over four quarters and why",
                   is this: SQL gives the change, filings give the why.
* `CROSS_SOURCE` — spans more than one ticker, or explicitly asks one source to be
                   reconciled against another ("does the news match the 10-K").

`classify()` is deliberately pure — no DB, no network, no clock beyond `now` — so the
routing contract is unit-testable without infrastructure. Logging is a separate seam
(`log_decision`), and `route_and_log()` is what the answer pipeline calls, so every
decision a user actually triggers lands in `routing_log`.
"""
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from src.common.config import CFG
from src.common.models import Filters
from src.indexing.store import placeholder

ROUTES = ("NEWS", "FILING_RAG", "NUMERIC", "MIXED", "CROSS_SOURCE")

NUMERIC_PAT = re.compile(
    r"\b(revenue|sales|eps|earnings per share|margin|operating income|gross profit|"
    r"growth|grew|declined|how much|how many|what was|qoq|yoy|quarter over quarter|"
    r"year over year)\b", re.I)
# Explicit news vocabulary ONLY. Recency ("last week", "latest") is deliberately not
# here — it comes from detect_window, because a recency phrase alone does not mean news:
# "what was NVDA revenue last quarter" is a NUMERIC question wearing a date phrase.
NEWS_PAT = re.compile(
    r"\b(news|headlines?|announced?|announcement|press release|what happened|"
    r"happening|going on|developments?|coverage)\b", re.I)
# "annual report" / "quarterly filing" were missing here, which cost temp-05 in the
# Phase 7 eval: "Microsoft's *latest* annual report" matched the recency phrase, nothing
# marked it as a filing question, and it routed to NEWS. Same class as the "last
# quarter" bug in DECISIONS #26 — a document noun has to outrank a recency adjective.
FILING_PAT = re.compile(
    r"\b(10-?[kq]|8-?k|filings?|filed|annual report|quarterly report|"
    r"quarterly filing|annual filing|risk factors?|md&a|management (said|discuss)|"
    r"earnings call|prepared remarks|commentary|disclosed?|outlook|guidance|"
    r"strategy|competition|segments?)\b", re.I)
# "and why", "explain", "drivers" — the tell that a number alone will not answer it.
NARRATIVE_PAT = re.compile(
    r"\b(why|explain|reason|driver|drivers|cause|caused|because|how come|"
    r"what drove|attribut\w+|commentary|context)\b", re.I)
RECONCILE_PAT = re.compile(
    r"\b(match|consistent with|compared to what|reconcile|agree with|contradict|"
    r"versus what|against what|say the same)\b", re.I)
COMPARE_PAT = re.compile(r"\b(compare|versus|vs\.?|against|between|which of)\b", re.I)

METRIC_WORDS = {
    "revenue": "revenue", "sales": "revenue", "top line": "revenue",
    "eps": "eps_diluted", "earnings per share": "eps_diluted",
    "operating income": "operating_income", "operating profit": "operating_income",
    "gross margin": "gross_margin", "margin": "gross_margin",
}

RELATIVE_DAYS = {
    "today": 1, "yesterday": 2, "this week": 7, "last week": 7, "past week": 7,
    "last 7 days": 7, "last month": 30, "past month": 30, "last quarter": 120,
    "recently": 30, "recent": 30, "latest": 30,
}
NDAYS_PAT = re.compile(r"\b(?:last|past)\s+(\d+)\s+(day|week|month)s?\b", re.I)


@dataclass
class RouteDecision:
    query: str
    route: str
    method: str                                  # rule | llm | llm_invalid
    tickers: list[str] = field(default_factory=list)
    filters: Filters = field(default_factory=Filters)
    reasons: list[str] = field(default_factory=list)   # which rules fired — shown in the UI
    query_plan: tuple[str, dict] | None = None        # canned query + params for NUMERIC
    news_signal: bool = False                         # asked for news, vs merely multi-ticker
    latency_ms: int = 0

    def describe(self) -> str:
        why = "; ".join(self.reasons) or "no rule matched"
        return (f"{self.route} via {self.method} ({self.latency_ms}ms) — "
                f"tickers={self.tickers or 'any'}; {why}")


def detect_tickers(query: str) -> list[str]:
    """Ticker symbols and company aliases, in config order so output is stable.

    Aliases come from the news config rather than a second list here — one place to
    add a company. Word-boundary matched so "JPM" does not fire inside "JPMorgan's"
    twice and, more importantly, so "AI" style substrings never match a ticker.
    """
    found = []
    aliases = CFG["news"].get("aliases", {})
    for ticker in CFG["tickers"]:
        names = [ticker.lower()] + [a.lower() for a in aliases.get(ticker, [])]
        if any(re.search(rf"\b{re.escape(n)}\b", query, re.I) for n in names):
            found.append(ticker)
    return found


def detect_window(query: str, now: datetime | None = None) -> tuple[date | None, str | None]:
    """Relative date phrase -> a concrete `date_from`. Returns (date, phrase).

    CLAUDE.md principle #4: news queries filter by date in SQL *before* similarity, so
    the phrase has to become a hard bound here rather than a hint in the prompt. An
    explicit "last 30 days" beats the vaguer table entries, so it is checked first.
    """
    now = now or datetime.now(timezone.utc)
    m = NDAYS_PAT.search(query)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        days = n * {"day": 1, "week": 7, "month": 30}[unit]
        return (now - timedelta(days=days)).date(), m.group(0)
    for phrase, days in RELATIVE_DAYS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", query, re.I):
            return (now - timedelta(days=days)).date(), phrase
    return None, None


def detect_metric(query: str) -> str | None:
    """Longest alias wins, so "gross margin" is not swallowed by "margin"."""
    low = query.lower()
    hits = [(len(word), metric) for word, metric in METRIC_WORDS.items() if word in low]
    return max(hits)[1] if hits else None


def plan_query(query: str, tickers: list[str]) -> tuple[str, dict] | None:
    """Pick a canned query from `queries.QUERIES` plus its params.

    The LLM never writes SQL (DECISIONS #20). It does not pick the query either in v1
    — these rules do — so a NUMERIC answer is reproducible and the failure mode is a
    missing plan rather than a plausible wrong one.
    """
    if not tickers:
        return None
    metric = detect_metric(query) or "revenue"
    n = 4
    m = re.search(r"\b(?:last|past)\s+(\d+)\s+quarters?\b", query, re.I)
    if m:
        n = int(m.group(1))

    if len(tickers) > 1:
        return "compare_tickers", {"tickers": tickers, "metric": metric, "n_quarters": n}
    ticker = tickers[0]
    if re.search(r"\b(yoy|year over year|year-over-year|vs last year)\b", query, re.I):
        return "yoy_growth", {"ticker": ticker, "metric": metric, "n_quarters": n}
    if re.search(r"\b(qoq|quarter over quarter|quarter-over-quarter|sequential)\b", query, re.I):
        return "qoq_growth", {"ticker": ticker, "metric": metric, "n_quarters": n}
    if metric == "gross_margin":
        return "margin_trend", {"ticker": ticker, "n_quarters": max(n, 8)}
    if re.search(r"\b(latest|most recent|last reported)\b", query, re.I):
        return "latest_metric", {"ticker": ticker, "metric": metric}
    if re.search(r"\b(chang\w+|trend|over the last|history)\b", query, re.I):
        return "metric_series", {"ticker": ticker, "metric": metric, "n_quarters": n}
    return "latest_metric", {"ticker": ticker, "metric": metric}


def _rule_route(query: str, tickers: list[str], window: date | None) -> tuple[str | None, list[str]]:
    """The decidable cases. Returns (route or None, reasons)."""
    reasons = []
    numeric = bool(NUMERIC_PAT.search(query))
    # Note: a recency window is NOT folded in here. It is weaker evidence than a
    # news keyword and is weighed separately below, so "revenue last quarter"
    # stays NUMERIC.
    news = bool(NEWS_PAT.search(query))
    filing = bool(FILING_PAT.search(query))
    narrative = bool(NARRATIVE_PAT.search(query))

    if numeric:
        reasons.append("numeric keyword")
    if news:
        reasons.append("news keyword")
    if window is not None:
        reasons.append("recency phrase")
    if filing:
        reasons.append("filing keyword")
    if narrative:
        reasons.append("narrative 'why'")

    # Reconciling one source against another is cross-source regardless of ticker count.
    if RECONCILE_PAT.search(query) and (news or filing):
        reasons.append("reconcile two sources")
        return "CROSS_SOURCE", reasons
    if len(tickers) > 1 and (COMPARE_PAT.search(query) or numeric or filing or news):
        reasons.append(f"{len(tickers)} tickers")
        return "CROSS_SOURCE", reasons
    # A number plus a "why" needs both stores; this is the Definition-of-Done query.
    if numeric and (narrative or filing):
        return "MIXED", reasons
    # Numeric beats a bare recency phrase: "revenue last quarter" names a fiscal period,
    # not a news window. Explicit news vocabulary is what overrides it.
    if numeric and not news:
        return "NUMERIC", reasons
    if news or (window is not None and not filing):
        return "NEWS", reasons
    if filing:
        return "FILING_RAG", reasons
    return None, reasons


# Few-shot, and NEWS is described restrictively on purpose. Measured on 7 ambiguous
# queries llama3.1:8b scored 2/7 with a bare label list, answering NEWS to everything
# including "NVDA Blackwell" — classic position bias toward the first option offered.
# Naming FILING_RAG as the explicit home for broad/vague company questions, plus one
# example per route, took the same 7 cases to 7/7. The fallback only ever sees queries
# no rule matched, which are exactly the vague ones, so this margin is the whole value
# of the fallback.
LLM_SYSTEM = """You classify a financial research question by WHICH DATA STORE answers it.
Reply with exactly one label from this list and nothing else:
NUMERIC FILING_RAG NEWS MIXED CROSS_SOURCE

NUMERIC      - a reported figure from a financials table (revenue, EPS, margin).
FILING_RAG   - narrative the company writes about itself in SEC filings: strategy,
               risk factors, products, competition, segments. This is also the right
               answer for a BROAD or VAGUE question about one company.
NEWS         - ONLY when the question asks about recent events, announcements, or
               what is happening now.
MIXED        - needs a figure AND filing narrative for ONE company.
CROSS_SOURCE - spans more than one company, or reconciles two different sources.

Examples:
Q: What was NVDA revenue last quarter
A: NUMERIC
Q: Tell me about Microsoft
A: FILING_RAG
Q: NVDA Blackwell
A: FILING_RAG
Q: Is JPM a good investment
A: FILING_RAG
Q: What happened to NVDA this week
A: NEWS
Q: Compare NVDA and MSFT margins
A: CROSS_SOURCE
Q: Why did revenue grow
A: MIXED"""


def _llm_route(query: str) -> tuple[str, str]:
    """LLM fallback. Returns (route, method); never raises and never invents a route."""
    from src.generation.llm import LLMUnavailable, complete
    try:
        raw = complete(LLM_SYSTEM, query, temperature=0.0).strip().upper()
    except LLMUnavailable:
        return "FILING_RAG", "llm_unavailable"
    # Earliest occurrence in the *response*, not first in ROUTES: a model that answers
    # "FILING_RAG, though it borders on NEWS" must resolve to FILING_RAG, and scanning
    # in ROUTES order would silently return whichever label happens to be listed first.
    hits = [(raw.find(r), r) for r in ROUTES if r in raw]
    if hits:
        return min(hits)[1], "llm"
    return "FILING_RAG", "llm_invalid"


def classify(query: str, now: datetime | None = None, use_llm: bool = True) -> RouteDecision:
    """Route a query. Pure: no DB, no clock beyond `now`, so it is testable directly.

    The LLM is consulted only when no rule fires at all. A wrong-but-confident rule is
    worse than a slow LLM call, so the rules are written to abstain (return None)
    rather than guess when the signals conflict.
    """
    started = time.perf_counter()
    tickers = detect_tickers(query)
    date_from, phrase = detect_window(query, now)
    route, reasons = _rule_route(query, tickers, date_from)
    method = "rule"

    if route is None:
        route, method = _llm_route(query) if use_llm else ("FILING_RAG", "no_rule_default")
        reasons.append(f"fallback -> {method}")
    if phrase:
        reasons.append(f"window '{phrase}' -> since {date_from}")

    # Only NEWS-ish routes get a hard date bound; a 10-K is not stale at 8 days old.
    filters = Filters(
        tickers=tickers or None,
        date_from=date_from if route in ("NEWS", "CROSS_SOURCE") else None,
    )
    # CROSS_SOURCE covers both "compare two tickers" and "check the news against the
    # 10-K". Only the second wants news evidence; without this flag a pure numeric
    # comparison drags in eight unrelated articles and buries the figures it asked for.
    decision = RouteDecision(
        query=query, route=route, method=method, tickers=tickers, filters=filters,
        reasons=reasons, news_signal=bool(NEWS_PAT.search(query)) or date_from is not None,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    if route in ("NUMERIC", "MIXED", "CROSS_SOURCE"):
        decision.query_plan = plan_query(query, tickers)
    return decision


def log_decision(conn, decision: RouteDecision) -> None:
    """Append to `routing_log`. Never lets a logging failure break an answer."""
    ph = placeholder()
    try:
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO routing_log (query, route, method, latency_ms) "
            f"VALUES ({ph}, {ph}, {ph}, {ph})",
            (decision.query, decision.route, decision.method, decision.latency_ms))
        conn.commit()
    except Exception:
        conn.rollback()


def route_and_log(conn, query: str, now: datetime | None = None,
                  use_llm: bool = True) -> RouteDecision:
    """What the answer pipeline calls, so every user-triggered decision is logged."""
    decision = classify(query, now=now, use_llm=use_llm)
    log_decision(conn, decision)
    return decision


def recent_decisions(conn, limit: int = 20) -> list[dict]:
    """Read back the log — powers the smoke test and the Phase 8 sidebar."""
    cur = conn.cursor()
    cur.execute("SELECT query, route, method, latency_ms FROM routing_log "
                f"ORDER BY id DESC LIMIT {int(limit)}")
    return [{"query": r[0], "route": r[1], "method": r[2], "latency_ms": r[3]}
            for r in cur.fetchall()]
