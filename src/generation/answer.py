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
from dataclasses import dataclass, field
from functools import lru_cache

from src.common.config import CFG
from src.common.models import Answer, Filters
from src.generation.llm import LLMUnavailable, available, complete
from src.generation.prompts import ANSWER_SYSTEM, INSUFFICIENT
from src.router.router import RouteDecision, route_and_log

# Matches [3] and the grouped form [2, 4, 6] that llama3.1 emits despite the prompt.
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cite_numbers(text: str) -> list[int]:
    return [int(n) for g in CITATION_RE.findall(text) for n in re.findall(r"\d+", g)]
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
    score: float | None = None   # retriever score, when the item came from a ranked list
    retriever: str | None = None # which arm produced it — the UI shows dense/rrf/cross-encoder

    def render(self, n: int) -> str:
        return f"[{n}] ({self.label})\n{self.text}"


def build_context(evidence: list[Evidence]) -> str:
    """Number every item [1]..[n]. The numbering here is the only thing the model may
    cite, and it is the same numbering the post-check validates against."""
    return "\n\n".join(e.render(i) for i, e in enumerate(evidence, start=1))


# --- evidence gathering ------------------------------------------------------

def filing_evidence(retriever, query: str, tickers: list[str],
                    top_k: int | None = None, mode: str = "rerank",
                    overrides: Filters | None = None) -> list[Evidence]:
    """Filing chunks via the Phase 3 stack.

    Only the ticker filter is carried over from the router, never its date window: a
    risk factor disclosed in last year's 10-K is not stale the way a news article is,
    and applying a 7-day bound to filings would empty the context on every NEWS-ish
    phrasing of a filing question.

    `overrides` is the one exception, and it exists for the Phase 8 sidebar: a filter a
    human set explicitly outranks one the router inferred, including a date bound. It
    is passed in rather than read from the decision so that the inference path stays
    exactly what Phase 7 measured.
    """
    filters = overrides or Filters(tickers=tickers or None)
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
                            text=h.text[:FILING_BODY_CHARS], url=m.source_url,
                            score=h.score, retriever=h.retriever))
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


def gather(conn, decision: RouteDecision, retriever=None, mode: str = "rerank",
           overrides: Filters | None = None, top_k: int | None = None) -> list[Evidence]:
    """Assemble the context for a route. The route decides which stores are consulted.

    `overrides` carries explicitly-set filters (the Streamlit sidebar) into both the
    filing search and the news window; None means the router's own inference stands,
    which is the path the evals exercise.
    """
    route = decision.route
    evidence: list[Evidence] = []
    date_from = (overrides.date_from if overrides and overrides.date_from
                 else decision.filters.date_from)
    tickers = (overrides.tickers if overrides and overrides.tickers else decision.tickers)

    if route in ("NUMERIC", "MIXED", "CROSS_SOURCE"):
        evidence += financial_evidence(conn, decision.query_plan)
    # NEWS always; CROSS_SOURCE only when the query actually asked about news, so
    # "compare NVDA and MSFT revenue" stays a numbers-and-filings answer.
    if route == "NEWS" or (route == "CROSS_SOURCE" and decision.news_signal):
        evidence += news_evidence(conn, tickers, date_from)
    if route in ("FILING_RAG", "MIXED", "CROSS_SOURCE") and retriever is not None:
        evidence += filing_evidence(retriever, decision.query, tickers, mode=mode,
                                    overrides=overrides, top_k=top_k)
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
    for n in cite_numbers(text):
        (cited if 1 <= n <= n_context else hallucinated).append(n)

    cleaned = text
    if hallucinated:
        def keep_valid(m):
            ok = [n for n in re.findall(r"\d+", m.group(1)) if 1 <= int(n) <= n_context]
            return f"[{', '.join(ok)}]" if ok else ""
        cleaned = CITATION_RE.sub(keep_valid, text)
        cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned, sorted(set(cited)), sorted(set(hallucinated))



# --- post-generation validation chain ----------------------------------------
#
# `check_citations` proves a [n] *exists*. It says nothing about whether the sentence
# it is attached to is true of item n, and FC-14 is what that gap looks like: the answer
# cited [1] next to "$44.1B" and "$39.3B" while item [1] held 96.22 / 81.61 / 68.13 /
# 57.01. Every citation resolved; every number was invented.
#
# Each validator is deterministic — no second LLM call — so its verdict is reproducible
# and the gate can assert on it. They run cheapest-and-most-decisive first and stop at
# the first failure. On failure the answer is *withheld*, not regenerated: a retry loop
# multiplies latency and gives a model that just invented a number another draw at
# inventing one. The contract is CLAUDE.md #5 — cite or say insufficient evidence.
#
# Known cost, stated: these are lexical/numeric checks. They catch a figure that is not
# in the evidence; they cannot catch a negation flip or a paraphrase that inverts a
# claim. That is an NLI/judge problem and is left as a v2 item, not papered over.

@dataclass
class Verdict:
    name: str
    passed: bool
    detail: str = ""
    items: list[str] = field(default_factory=list)   # the offending figures / sentences


def _vcfg() -> dict:
    v = CFG.get("generation", {}).get("validation", {}) or {}
    return {"enabled": v.get("enabled", True),
            "min_citation_coverage": v.get("min_citation_coverage", 0.8),
            "min_grounding_overlap": v.get("min_grounding_overlap", 0.5)}


_SCALE = {"trillion": 1e12, "t": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9,
          "million": 1e6, "mm": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
# A number with no unit in a filing table ("44,062") is in whatever the table header
# says, which is not in the chunk. So a bare evidence figure may stand for any of these.
_BARE_SCALES = (1.0, 1e3, 1e6, 1e9)

_FIG_RE = re.compile(
    r"(?<![A-Za-z0-9_.,/])(?:(?P<cur>[$€£])\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suf>%|\s?percent\b|\s?(?:trillion|billion|million|thousand)\b"
    r"|\s?(?:bps|basis points?)\b|(?:bn|MM|[BMKT])\b)?", re.I)
# "Item 2.02", "Section 7" are labels, not figures.
_LABEL_BEFORE = re.compile(r"(?:item|section|note|exhibit|part|table|figure|rule|regulation)\s+$", re.I)
_MARKUP_RE = re.compile(r"[*_`#]")


@dataclass
class Figure:
    raw: str
    value: float          # in the figure's own unit, scale applied
    half: float           # half a unit of the last printed digit — the rounding slack
    bare: bool            # no currency, no %, no scale: its scale is unknowable
    unscaled: bool        # no scale suffix (may still carry "$"): a filing table cell like
                          # "$ 71" is in millions, and the "$" says nothing about that
    checkable: bool       # looks like a financial figure rather than a count or a label


def extract_figures(text: str) -> list[Figure]:
    out = []
    for m in _FIG_RE.finditer(text):
        if _LABEL_BEFORE.search(text[:m.start()]):
            continue
        num, suf, cur = m.group("num"), (m.group("suf") or "").strip().lower(), m.group("cur")
        digits = num.replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        scale = _SCALE.get(suf, 1.0)
        out.append(Figure(
            raw=m.group(0).strip(), value=float(digits) * scale,
            half=0.5 * 10 ** -decimals * scale,
            bare=not (cur or suf), unscaled=not suf or suf in ("%", "percent"),
            checkable=bool(cur or suf or "." in num or "," in num or len(digits) >= 5)))
    return out


@lru_cache(maxsize=1024)
def _evidence_values(text: str) -> tuple[tuple[float, float], ...]:
    vals = []
    for f in extract_figures(text):
        for sc in (_BARE_SCALES if f.unscaled else (1.0,)):
            vals.append((f.value * sc, f.half * sc))
    return tuple(vals)


def _supported(fig: Figure, pool: tuple[tuple[float, float], ...]) -> bool:
    """Could `fig` be a rounding of some figure in `pool`? Compatible when the gap is
    within the coarser of the two roundings, so $96.2B matches 96.22B and $96,221 million
    matches $96.22B, while $44.1B matches nothing near 96.22B."""
    # A bare figure ("44,062") has no stated scale, so it may stand for any of them —
    # the same latitude the evidence side gets. Currency/percent/scaled figures do not.
    for sc in (_BARE_SCALES if fig.bare else (1.0,)):
        value, half = fig.value * sc, fig.half * sc
        for v, h in pool:
            if abs(value - v) <= max(half, h) + 1e-9 * max(abs(v), 1.0):
                return True
    return False


def _prose(text: str) -> str:
    return _MARKUP_RE.sub("", CITATION_RE.sub(" ", text))


def validate_numeric(text: str, evidence: list[Evidence]) -> Verdict:
    """Every figure in the prose must appear somewhere in the numbered evidence.

    Fails closed on derived figures too: "up 41.5%" computed by the model from two
    evidence numbers is rejected unless the evidence states it. LLM arithmetic is
    unreliable, and a growth rate that is not in the SQL output cannot be checked.
    """
    pools = [_evidence_values(e.text) for e in evidence]
    bad = [f.raw for f in extract_figures(_prose(text))
           if f.checkable and not any(_supported(f, p) for p in pools)]
    if bad:
        return Verdict("numeric_consistency", False,
                       f"{len(bad)} figure(s) not in the evidence: {', '.join(bad)}", bad)
    return Verdict("numeric_consistency", True)


_HEADER_RE = re.compile(r"^(?:facts|interpretation)\s*:?$", re.I)
# Sentences that assert absence rather than a fact: there is nothing to cite. The second
# line duplicates answer_eval's refusal pattern on purpose — generation must not import
# from evals — so keep the two in step.
_NO_INTERP_RE = re.compile(
    r"(?:context|evidence|sources?)\s+(?:does not|doesn't|provides? no|contains? no|is insufficient)"
    r"|no interpretation|cannot be inferred|not supported by"
    r"|no mention of|did not (?:explicitly )?(?:mention|state|provide|specify|discuss)"
    r"|not (?:mentioned|provided|specified|discussed) in the (?:provided )?(?:context|documents)"
    r"|no information (?:about|on|regarding)|couldn'?t find|could not find|\bunclear\b"
    r"|(?:\bno\b|\bnot\b|n't)\b.{0,60}\b(?:in|within) the (?:provided )?(?:context|documents|sources)", re.I)
_STOP = frozenset("""about above after also among because been before being between both could
does each from have into more most much only other over same should since some such than that
their them then there these they this those through under until very were what when where which
while whose will with within would your""".split())
_MIN_WORDS = 4


@dataclass
class Unit:
    text: str            # sentence without citation markers
    cites: list[int]
    section: str         # "" | "facts" | "interpretation"


def claim_units(text: str) -> list[Unit]:
    """Split prose into the sentences a reader would expect to be individually sourced.

    Bullets are separate units. A citation trailing the full stop ("... rose. [1]") is
    moved back inside the sentence it belongs to, otherwise the next split strands it.
    Headers, very short fragments and "the context supports no interpretation" lines
    are not claims.
    """
    units, section = [], ""
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        plain = _MARKUP_RE.sub("", line).strip()
        if not plain:
            continue
        if _HEADER_RE.match(plain):
            section = plain.rstrip(":").lower()
            continue
        if plain.endswith(":"):
            continue                                  # a lead-in, not a claim
        line = re.sub(r"([.!?])\s*((?:\[\d+\]\s*)+)", lambda m: " " + m.group(2) + m.group(1), line)
        for sent in re.split(r"(?<=[.!?])(?<!\b[A-Z]\.)(?<!\bInc\.)(?<!\bCorp\.)(?<!\bCo\.)(?<!\bLtd\.)(?<!\bNo\.)"
                              r"\s+(?=[A-Z$\"(\[])", line):
            cites = cite_numbers(sent)
            body = _prose(sent).strip()
            if len(re.findall(r"\w+", body)) < _MIN_WORDS or _NO_INTERP_RE.search(body):
                continue
            units.append(Unit(body, cites, section))
    return units


def validate_coverage(text: str, evidence: list[Evidence]) -> Verdict:
    """Every claim-bearing sentence should carry a citation; any that states a figure must.

    Two thresholds because the failures differ. An uncited *figure* is checkable and
    wrong to ship. An uncited connective sentence is a style defect, tolerated up to
    `min_citation_coverage`. Zero citations overall lands under any sane threshold, so
    `Answer.uncited` becomes a withheld answer rather than a warning.
    """
    units = claim_units(text)
    if not units:
        return Verdict("citation_coverage", True)
    uncited = [u for u in units if not u.cites]
    fig_uncited = [u.text for u in uncited if any(f.checkable for f in extract_figures(u.text))]
    coverage = 1 - len(uncited) / len(units)
    floor = _vcfg()["min_citation_coverage"]
    if fig_uncited:
        return Verdict("citation_coverage", False,
                       f"{len(fig_uncited)} sentence(s) state a figure with no citation", fig_uncited)
    if coverage < floor:
        return Verdict("citation_coverage", False,
                       f"only {coverage:.0%} of {len(units)} claims cited (need {floor:.0%})",
                       [u.text for u in uncited])
    return Verdict("citation_coverage", True, f"{coverage:.0%} of {len(units)} claims cited")


def _stems(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z][a-z'-]{3,}", text.lower()):
        if w in _STOP:
            continue
        for suf in ("ing", "ed", "es", "s", "ly"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[:-len(suf)]
                break
        out.add(w)
    return out


def validate_grounding(text: str, evidence: list[Evidence]) -> Verdict:
    """Each cited sentence must be supported by the item(s) it cites, not by the context
    at large. Two tests: its figures appear in a cited item (a right number pinned to
    the wrong [n] is a misattribution), and its content words mostly appear there.

    Lexical overlap is a deliberately blunt instrument: it flags a sentence about
    something the cited text never mentions, and passes one that inverts the cited text.
    """
    floor = _vcfg()["min_grounding_overlap"]
    bad = []
    for u in claim_units(text):
        cited = [evidence[n - 1] for n in u.cites if 1 <= n <= len(evidence)]
        if not cited:
            continue                                   # coverage's job, not grounding's
        pools = tuple(v for e in cited for v in _evidence_values(e.text))
        miss = [f.raw for f in extract_figures(u.text) if f.checkable and not _supported(f, pools)]
        if miss:
            bad.append(f"{u.text[:90]} [figure {', '.join(miss)} not in cited item(s)]")
            continue
        if all(e.kind == "financial" for e in cited):
            continue        # rows of numbers carry no prose to overlap with; figures were checked
        words = _stems(u.text)
        if not words:
            continue
        have = set().union(*(_stems(f'{e.label} {e.text}') for e in cited))
        overlap = len(words & have) / len(words)
        if overlap < floor:
            bad.append(f"{u.text[:90]} [overlap {overlap:.0%}]")
    if bad:
        return Verdict("grounding", False, f"{len(bad)} sentence(s) not supported by their citation", bad)
    return Verdict("grounding", True)


VALIDATORS = (("numeric_consistency", validate_numeric),
              ("citation_coverage", validate_coverage),
              ("grounding", validate_grounding))


def run_chain(text: str, evidence: list[Evidence], refusal: bool = False) -> list[Verdict]:
    """Run validators in order, stopping at the first failure.

    A refusal is only checked for figures: it makes no claims to cover or ground, but
    "INSUFFICIENT EVIDENCE: ... though revenue was about $44B" is FC-12's smuggling
    pattern and the numeric check is what catches the number.
    """
    verdicts = []
    for name, fn in VALIDATORS:
        if refusal and name != "numeric_consistency":
            continue
        v = fn(text, evidence)
        verdicts.append(v)
        if not v.passed:
            break
    return verdicts


def _insufficient(reason: str, decision: RouteDecision, evidence: list[Evidence]) -> Answer:
    return Answer(text=f"{INSUFFICIENT}: {reason}", citations=[], route=decision.route,
                  insufficient_evidence=True, evidence=evidence)


def generate(query: str, evidence: list[Evidence], decision: RouteDecision,
             validate: bool | None = None) -> Answer:
    """Prompt, generate, verify citations, then run the validation chain.

    No evidence means no LLM call at all. `validate` overrides `generation.validation.
    enabled` so the eval harness can measure the chain's effect as an A/B arm.
    """
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
    refusal = cleaned.upper().startswith(INSUFFICIENT)

    verdicts: list[Verdict] = []
    if _vcfg()["enabled"] if validate is None else validate:
        verdicts = run_chain(cleaned, evidence, refusal=refusal)
        failed = next((v for v in verdicts if not v.passed), None)
        if failed:
            # The reason names the check, never the offending figures: repeating an
            # unsupported "$44.1B" in the refusal would publish it anyway. The figures
            # live on `validation` and `rejected_text` for the audit trail and the UI.
            ans = _insufficient(
                f"the generated answer failed the {failed.name} check and was withheld",
                decision, evidence)
            ans.validation, ans.rejected_text = verdicts, cleaned
            ans.hallucinated_citations = hallucinated
            return ans

    return Answer(
        text=cleaned,
        citations=[evidence[n - 1].key for n in cited],
        route=decision.route,
        insufficient_evidence=refusal,
        hallucinated_citations=hallucinated,
        evidence=evidence,
        validation=verdicts,
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
