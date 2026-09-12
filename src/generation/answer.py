"""Answer assembly: route -> gather evidence -> generate with [n] citations -> verify.

Three stores answer questions here and they have nothing in common — filing chunks
carry a `chunk_id`, news carries an `article_id`, financials carry no id at all, just a
(ticker, metric, period) coordinate. Rather than teach the prompt three citation
formats, everything is normalized to `Evidence` first, numbered `[1]..[n]` once, and
verified once. That is what makes the citation post-check a single function instead of
three, and what lets `Answer.citations` be a list of stable keys the eval harness can
compare against `expected_chunk_ids`.

**The post-check is the point.** An LLM will happily write `[7]` when six chunks were
supplied, and a reader who trusts citations is exactly the reader that mistake fools
(FC-6). So every `[n]` is checked against the context that was actually sent: valid
ones are resolved to their evidence key, invalid ones are stripped from the prose and
recorded on the Answer. Stripping rather than only flagging is deliberate — a dangling
citation left in the text still reads as sourced to a human skimming the answer.

Evidence assembly is *hard-filter-first* (CLAUDE.md #4): the router's ticker and date
bounds become SQL/payload predicates before anything is embedded or ranked.
"""
import re
from dataclasses import dataclass

from src.common.config import CFG
from src.common.models import Answer, Filters
from src.generation.llm import LLMUnavailable, available, complete
from src.generation.prompts import ANSWER_SYSTEM, INSUFFICIENT
from src.router.router import RouteDecision, route_and_log

CITATION_RE = re.compile(r"\[(\d+)\]")
NEWS_BODY_CHARS = 1200
FILING_BODY_CHARS = 1800


@dataclass
class Evidence:
    """One numbered piece of context, whatever store it came from."""
    key: str                    # chunk_id | article_id | financials:TICKER:metric
    kind: str                   # filing | news | financial
    label: str                  # provenance line the model and the UI both see
    text: str
    url: str | None = None

    def render(self, n: int) -> str:
        return f"[{n}] ({self.label})\n{self.text}"


def build_context(evidence: list[Evidence]) -> str:
    """Number every item [1]..[n]. The numbering here is the only thing the model may
    cite, and it is the same numbering the post-check validates against."""
    return "\n\n".join(e.render(i) for i, e in enumerate(evidence, start=1))


# --- evidence gathering ------------------------------------------------------

def filing_evidence(retriever, query: str, tickers: list[str],
                    top_k: int | None = None, mode: str = "rerank") -> list[Evidence]:
    """Filing chunks via the Phase 3 stack.

    Only the ticker filter is carried over from the router, never its date window: a
    risk factor disclosed in last year's 10-K is not stale the way a news article is,
    and applying a 7-day bound to filings would empty the context on every NEWS-ish
    phrasing of a filing question.
    """
    filters = Filters(tickers=tickers or None)
    hits = retriever.search(query, mode=mode,
                            top_k=top_k or CFG["retrieval"]["final_top_k"],
                            filters=filters)
    out = []
    for h in hits:
        m = h.meta
        bits = [m.ticker, m.doc_type, m.fiscal_period or "", m.section or ""]
        label = " · ".join(b for b in bits if b)
        if m.filing_date:
            label += f" · filed {m.filing_date}"
        # Two chunks of the same filing otherwise render an identical provenance line,
        # so a reader checking [2] against [7] cannot tell which passage was cited.
        # The chunk_id tail is the only thing that distinguishes them.
        label += f" · #{m.chunk_id.rsplit('::', 1)[-1]}"
        out.append(Evidence(key=m.chunk_id, kind="filing", label=label,
                            text=h.text[:FILING_BODY_CHARS], url=m.source_url))
    return out


def news_evidence(conn, tickers: list[str], date_from, limit: int = 8) -> list[Evidence]:
    """Recent, non-duplicate articles, via the Phase 5 event loader.

    Reuses `brief.load_events` rather than re-writing the window SQL, so the dedup and
    clustering semantics stay defined in exactly one place. Articles are labelled
    FULL TEXT / HEADLINE ONLY for the same reason the brief does it (FC-6): most news
    reaches us as a Google redirect with no recoverable body, and a headline that is
    not marked as the evidence ceiling invites the model to invent the rest.
    """
    from datetime import datetime, timezone
    from datetime import time as dtime

    from src.news.brief import load_events

    since = None
    if date_from:
        since = datetime.combine(date_from, dtime.min, tzinfo=timezone.utc)

    articles = []
    for ticker in (tickers or list(CFG["tickers"])):
        for event in load_events(conn, ticker, since=since):
            articles.extend((ticker, a) for a in event.live)
    articles.sort(key=lambda ta: ta[1].published_at, reverse=True)

    out = []
    for ticker, a in articles[:limit]:
        if a.headline_only:
            body = "HEADLINE ONLY — no body text available. The headline is the only evidence."
        else:
            body = " ".join(a.text.split())[:NEWS_BODY_CHARS]
        label = f"{ticker} · news · {a.source or 'unknown'} · {a.published_at.date()}"
        out.append(Evidence(key=a.article_id, kind="news", label=label,
                            text=f"Headline: {a.headline}\n{body}", url=a.url))
    return out


def financial_evidence(conn, plan: tuple[str, dict] | None) -> list[Evidence]:
    """Run the canned query the router picked and render its rows as one context item.

    One item per result set, not per quarter: the answer cites "the revenue series",
    and four separate numbered items for four quarters would push the filing chunks
    out of a small context for no gain in traceability.
    """
    if not plan:
        return []
    from src.structured import queries as q

    name, params = plan
    try:
        rows = q.run(conn, name, **params)
    except ValueError:
        return []                       # unknown query/param — fail closed, not loudly wrong
    if not rows:
        return []
    if not isinstance(rows, list):
        rows = [rows]

    lines, ticker = [], params.get("ticker") or ",".join(params.get("tickers", []))
    for row in rows:
        lines.append(row.format() if hasattr(row, "format") else str(row))
    metric = params.get("metric", "")
    return [Evidence(
        key=f"financials:{ticker}:{metric or name}",
        kind="financial",
        label=f"{ticker} · financials · {name}({metric})" if metric
              else f"{ticker} · financials · {name}",
        text="\n".join(lines))]


def gather(conn, decision: RouteDecision, retriever=None,
           mode: str = "rerank") -> list[Evidence]:
    """Assemble the context for a route. The route decides which stores are consulted."""
    route = decision.route
    evidence: list[Evidence] = []

    if route in ("NUMERIC", "MIXED", "CROSS_SOURCE"):
        evidence += financial_evidence(conn, decision.query_plan)
    # NEWS always; CROSS_SOURCE only when the query actually asked about news, so
    # "compare NVDA and MSFT revenue" stays a numbers-and-filings answer.
    if route == "NEWS" or (route == "CROSS_SOURCE" and decision.news_signal):
        evidence += news_evidence(conn, decision.tickers, decision.filters.date_from)
    if route in ("FILING_RAG", "MIXED", "CROSS_SOURCE") and retriever is not None:
        evidence += filing_evidence(retriever, decision.query, decision.tickers, mode=mode)
    return evidence


# --- citation verification ---------------------------------------------------

def check_citations(text: str, n_context: int) -> tuple[str, list[int], list[int]]:
    """Verify every [n] against the context actually sent.

    Returns (cleaned_text, cited, hallucinated). Invalid citations are removed from the
    prose, because a dangling [7] still reads as evidence to someone skimming; keeping
    them only in a warning field would leave the misleading artifact in the text a user
    copies out.
    """
    cited, hallucinated = [], []
    for raw in CITATION_RE.findall(text):
        n = int(raw)
        (cited if 1 <= n <= n_context else hallucinated).append(n)

    cleaned = text
    if hallucinated:
        cleaned = CITATION_RE.sub(
            lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_context else "", text)
        cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned, sorted(set(cited)), sorted(set(hallucinated))


def _insufficient(reason: str, decision: RouteDecision, evidence: list[Evidence]) -> Answer:
    return Answer(text=f"{INSUFFICIENT}: {reason}", citations=[], route=decision.route,
                  insufficient_evidence=True, evidence=evidence)


def generate(query: str, evidence: list[Evidence], decision: RouteDecision) -> Answer:
    """Prompt, generate, then verify. No evidence means no LLM call at all."""
    if not evidence:
        return _insufficient(
            f"no {decision.route.lower().replace('_', ' ')} evidence matched "
            f"{decision.tickers or 'this query'}", decision, evidence)
    if not available():
        return _insufficient("no LLM backend reachable to compose an answer",
                             decision, evidence)

    user = (f"Question: {query}\n\nNumbered context ({len(evidence)} items):\n"
            f"{build_context(evidence)}")
    try:
        raw = complete(ANSWER_SYSTEM, user)
    except LLMUnavailable as e:
        return _insufficient(f"LLM backend failed mid-request ({e})", decision, evidence)

    cleaned, cited, hallucinated = check_citations(raw, len(evidence))
    return Answer(
        text=cleaned,
        citations=[evidence[n - 1].key for n in cited],
        route=decision.route,
        insufficient_evidence=cleaned.upper().startswith(INSUFFICIENT),
        hallucinated_citations=hallucinated,
        evidence=evidence,
    )


def ask(query: str, conn=None, retriever=None, mode: str = "rerank",
        use_llm: bool = True) -> Answer:
    """End-to-end: route (logged) -> gather -> generate -> verify.

    Opens and closes its own handles when none are passed, so a one-off CLI call works;
    pass a long-lived `Retriever` when asking many questions (the eval harness does).
    """
    from src.common.db import get_conn
    from src.retrieval.hybrid import Retriever

    own_conn = conn is None
    own_retriever = retriever is None
    conn = conn or get_conn()
    try:
        decision = route_and_log(conn, query, use_llm=use_llm)
        if retriever is None and decision.route in ("FILING_RAG", "MIXED", "CROSS_SOURCE"):
            retriever = Retriever(conn=conn)
        evidence = gather(conn, decision, retriever, mode=mode)
        answer = generate(query, evidence, decision)
        answer.decision = decision
        return answer
    finally:
        if own_retriever and retriever is not None:
            retriever._owns["conn"] = False     # the conn is ours to close, not its
            retriever.close()
        if own_conn:
            conn.close()
