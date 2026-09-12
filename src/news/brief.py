"""Weekly intelligence brief: SQL window -> events -> ranked -> summarized -> assembled.

The pipeline is hard-filter-first, exactly as CLAUDE.md principle #4 requires: ticker
and date range are a SQL `WHERE`, not a similarity search. Nothing is embedded at
brief time — clustering already happened at ingest, and `event_id` is a column.

**Ranking is where the news pipeline earns its keep.** The naive signal, article
count, is actively wrong on this corpus: the single largest JPM cluster in a live
window was 18 articles from *one* source (MarketBeat's automated 13F filing posts,
"$JPM Shares Acquired by <fund> LLC" repeated with different fund names). Volume says
that's the week's biggest story; it isn't news at all. Distinct-source count is the
signal that survives syndication and bot spam, because getting ten different
newsrooms to cover something is the actual evidence of importance. So sources
dominate the score and article count only breaks ties — see `score_event`.

Summarization degrades rather than fails. With an LLM reachable it writes the
summary and the bull/bear read; without one it falls back to an extractive summary
built from the headlines themselves, clearly marked. A brief you can't produce
because Ollama isn't running is worse than a plainer brief that is honest about how
it was made.
"""
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from src.common.config import CFG
from src.news.ingest import PUBLISHER_SUFFIX_RE
from src.generation.llm import LLMUnavailable, available, complete
from src.generation.prompts import EVENT_SUMMARY_SYSTEM
from src.indexing.store import as_date, placeholder


def display_headline(headline: str) -> str:
    """Drop Google's " - Publisher" suffix for rendering only.

    The stored headline stays exactly as published — the brief cites sources on their
    own line, so repeating the publisher inside every title is noise, but rewriting
    what the DB holds would make the citation disagree with the record.
    """
    return PUBLISHER_SUFFIX_RE.sub("", headline).strip()


@dataclass
class Article:
    article_id: str
    headline: str
    source: str | None
    published_at: datetime
    url: str
    topic: str | None
    is_duplicate: bool
    text: str | None = None         # None = headline-only (Google redirect or blocked)

    @property
    def headline_only(self) -> bool:
        return not (self.text or "").strip()


@dataclass
class Event:
    event_id: str
    ticker: str
    articles: list[Article] = field(default_factory=list)
    summary: str = ""
    summary_method: str = ""        # llm | extractive

    @property
    def primary(self) -> Article:
        """Earliest non-duplicate article — the one that broke the story."""
        live = [a for a in self.articles if not a.is_duplicate] or self.articles
        return min(live, key=lambda a: a.published_at)

    @property
    def live(self) -> list[Article]:
        return [a for a in self.articles if not a.is_duplicate]

    @property
    def sources(self) -> list[str]:
        return sorted({a.source for a in self.live if a.source})

    @property
    def n_duplicates(self) -> int:
        return sum(1 for a in self.articles if a.is_duplicate)

    @property
    def topic(self) -> str | None:
        """Modal topic across the cluster."""
        tags = [a.topic for a in self.live if a.topic]
        return max(set(tags), key=tags.count) if tags else None

    @property
    def span(self) -> tuple[date, date]:
        days = [a.published_at.date() for a in self.live]
        return min(days), max(days)


VOLUME_CAP = 3          # articles beyond this add nothing to the score


def score_event(event: Event) -> float:
    """Rank score: source diversity first, volume as a strict tie-break.

    `n_sources + 0.25 * min(n_articles, 3)`. The cap is the whole point. Uncapped,
    volume still wins on this corpus: MarketBeat's 18 automated 13F posts about JPM
    scored 5.5 and outranked a genuine story ("India lifts ban on JPMorgan unit",
    2 sources) at 2.5. Capping the volume term at 0.75 keeps it strictly below the
    value of a single additional source, so one more newsroom covering a story always
    beats any amount of repetition by one publisher.

    That makes the guarantee real rather than aspirational: **n articles from one
    source can never outrank n+1 sources, for any n.** Deliberately not a learned
    ranker — it has to be explainable in one line, and whether source-diversity
    ranking beats count ranking on human-judged importance is a Phase 7 measurement,
    not a tuned constant.
    """
    return len(event.sources) + 0.25 * min(len(event.live), VOLUME_CAP)


def load_events(conn, ticker: str, days: int | None = None,
                since: datetime | None = None) -> list[Event]:
    """Clustered articles for one ticker in the window. Hard SQL filter, no vectors."""
    days = days or CFG["news"]["lookback_days"]
    since = since or datetime.now(timezone.utc) - timedelta(days=days)
    ph = placeholder()
    cur = conn.cursor()
    cur.execute(
        "SELECT article_id, ticker, event_id, headline, source, published_at, url, "
        f"topic, is_duplicate, text FROM news_articles WHERE ticker = {ph} "
        f"AND published_at >= {ph} ORDER BY published_at",
        (ticker, since.isoformat() if ph == "?" else since))

    events: dict[str, Event] = {}
    for row in cur.fetchall():
        eid = row[2] or f"unclustered-{row[0]}"
        published = row[5]
        if isinstance(published, str):
            published = datetime.fromisoformat(published)
        events.setdefault(eid, Event(event_id=eid, ticker=row[1])).articles.append(
            Article(article_id=row[0], headline=row[3], source=row[4],
                    published_at=published, url=row[6], topic=row[7],
                    is_duplicate=bool(row[8]), text=row[9]))
    return [e for e in events.values() if e.live]


def rank_events(events: list[Event], top_n: int = 5) -> list[Event]:
    return sorted(events, key=score_event, reverse=True)[:top_n]


BODY_EXCERPT_CHARS = 1500


def _numbered_context(event: Event) -> str:
    """Number the articles and label each one with how much evidence it carries.

    Until now this passed headlines only, so body text that ingestion had gone to the
    trouble of extracting never reached the model — and, worse, an article with no body
    was indistinguishable from one with a body. The model saw a bare headline and
    filled the gap from parametric knowledge about the company (FC-3). The label is
    what the prompt's evidence rule keys on, so it is not decoration.
    """
    lines = []
    for i, a in enumerate(event.live, start=1):
        head = f"[{i}] ({a.published_at.date()}, {a.source or 'unknown'})"
        if a.headline_only:
            lines.append(f"{head} HEADLINE ONLY — no body text available.\n"
                         f"    Headline: {a.headline}")
        else:
            body = " ".join(a.text.split())[:BODY_EXCERPT_CHARS]
            lines.append(f"{head} FULL TEXT\n    Headline: {a.headline}\n"
                         f"    Body: {body}")
    return "\n".join(lines)


def summarize_event(event: Event, ticker: str, use_llm: bool = True) -> Event:
    """Attach a summary. Falls back to extractive when no LLM backend is reachable."""
    if use_llm and available():
        n_full = sum(1 for a in event.live if not a.headline_only)
        user = (f"Ticker: {ticker}\nArticles about one event "
                f"({n_full} with full text, {len(event.live) - n_full} headline-only):\n"
                f"{_numbered_context(event)}\n\n"
                "Summarize this event. Give bull and bear implications only where a "
                "FULL TEXT article supports them; otherwise use the exact sentence the "
                "evidence rule specifies. Cite article numbers inline like [1].")
        try:
            event.summary = complete(EVENT_SUMMARY_SYSTEM, user)
            event.summary_method = "llm"
            return event
        except LLMUnavailable:
            pass
    event.summary = _extractive_summary(event)
    event.summary_method = "extractive"
    return event


def _extractive_summary(event: Event) -> str:
    """Deterministic fallback: the earliest headline plus who else covered it.

    Makes no claim the headlines don't make, and never invents a bull/bear read — an
    unsupported implication is precisely the hallucination the citation contract
    elsewhere in this build exists to prevent.
    """
    first, last = event.span
    window = f"{first}" if first == last else f"{first} to {last}"
    others = [display_headline(a.headline)
              for a in sorted(event.live, key=lambda a: a.published_at)[1:3]]
    lines = [f"{display_headline(event.primary.headline)} "
             f"({event.primary.source or 'unknown'}, {window})."]
    if others:
        lines.append("Also reported: " + "; ".join(others) + ".")
    lines.append(f"Covered by {len(event.sources)} source(s): {', '.join(event.sources[:6])}"
                 + (f"; {event.n_duplicates} near-duplicate repost(s) suppressed."
                    if event.n_duplicates else "."))
    return " ".join(lines)


def render_brief(ticker: str, events: list[Event], days: int,
                 generated: datetime | None = None) -> str:
    """Markdown brief with dated, linked sources under every event."""
    generated = generated or datetime.now(timezone.utc)
    company = CFG["tickers"].get(ticker, {}).get("company", ticker)
    out = [f"# {ticker} — {company}", "",
           f"Weekly intelligence brief · last {days} days · "
           f"generated {generated.date()}", ""]
    if not events:
        out += ["_No qualifying news in this window._", ""]
        return "\n".join(out)

    methods = {e.summary_method for e in events if e.summary_method}
    if methods == {"extractive"}:
        out += ["> **Note:** no LLM backend reachable — summaries are extractive "
                "(headline-derived, no interpretation). Start Ollama for bull/bear reads.", ""]

    for i, event in enumerate(events, start=1):
        first, last = event.span
        window = f"{first}" if first == last else f"{first}–{last}"
        tag = f" · _{event.topic}_" if event.topic else ""
        out += [f"## {i}. {display_headline(event.primary.headline)}",
                f"*{window} · {len(event.live)} article(s) · "
                f"{len(event.sources)} source(s)*{tag}", "",
                event.summary, "", "**Sources**"]
        for a in sorted(event.live, key=lambda a: a.published_at):
            out.append(f"- [{display_headline(a.headline)}]({a.url}) — {a.source or 'unknown'}, "
                       f"{a.published_at.date()}")
        if event.n_duplicates:
            out.append(f"- _({event.n_duplicates} near-duplicate repost(s) suppressed)_")
        out.append("")
    return "\n".join(out)


def weekly_brief(conn, ticker: str, days: int | None = None, top_n: int = 5,
                 use_llm: bool = True) -> str:
    """End-to-end brief for one ticker."""
    days = days or CFG["news"]["lookback_days"]
    events = rank_events(load_events(conn, ticker, days), top_n)
    for event in events:
        summarize_event(event, ticker, use_llm)
    return render_brief(ticker, events, days)
