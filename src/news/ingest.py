"""News ingestion: RSS -> relevance filter -> dedup -> event clusters -> `news_articles`.

RSS-first, because it is free, has no API key, and is the only source that stays
legal without a licensing conversation. Three feed kinds, each with a different
failure mode that shapes the pipeline (all verified live — DECISIONS #21):

* **Google News search RSS** — broadest coverage (100 items/ticker), but every `link`
  is an opaque `news.google.com/rss/articles/CBM...` redirect that resolves to a
  JavaScript interstitial, not the article. So the real publisher URL is *not
  recoverable* and the body cannot be fetched. These items are headline-only. They
  are still the most valuable feed for *event detection*, because the publisher name
  does come through (`<source>`), which is exactly what source-diversity ranking needs.
* **Yahoo Finance ticker RSS** — direct publisher URLs whose bodies trafilatura can
  usually extract, but the feed leaks generic syndicated finance content: a third of
  the NVDA feed was unrelated ("Worried About RMDs in Retirement?"). Hence the alias
  relevance filter.
* **Company IR feeds** — authoritative and fully extractable, but only where a company
  publishes one. JPM has no working public RSS, so it runs on the two above.

Because Google redirect URLs are unique per item even for the same story, URL dedup
alone cannot collapse syndication. That is the whole reason for the second,
embedding-based dedup pass.
"""
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import feedparser

from src.common.config import CFG
from src.common.db import get_conn, storage_mode
from src.indexing.store import placeholder

GOOGLE_HOST = "news.google.com"


@dataclass
class NewsItem:
    article_id: str
    ticker: str
    headline: str
    source: str | None
    published_at: datetime
    url: str
    text: str | None = None
    topic: str | None = None
    event_id: str | None = None
    is_duplicate: bool = False
    feed: str = ""                      # which feed produced it — diagnostics only
    duplicate_of: str | None = field(default=None)


def article_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def is_google_redirect(url: str) -> bool:
    return GOOGLE_HOST in url


def to_utc(entry) -> datetime | None:
    """feedparser gives a UTC struct_time; the column is NOT NULL so undated items drop."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime(*parsed[:6], tzinfo=timezone.utc)


def entry_source(entry, fallback: str) -> str:
    """Publisher name. Google nests it under <source>; other feeds don't carry one, so
    the publisher is the feed's own host."""
    src = entry.get("source")
    if isinstance(src, dict) and src.get("title"):
        return src["title"]
    return fallback


def is_relevant(headline: str, ticker: str) -> bool:
    """Alias match on the headline.

    Needed because the ticker-scoped feeds are not actually ticker-scoped — Yahoo's
    NVDA feed carries general market articles. Matching on the headline (not the body)
    keeps the filter cheap and keeps it honest: an article that never names the company
    in its headline is not ticker news for a weekly brief.
    """
    low = headline.lower()
    return any(a in low for a in CFG["news"]["aliases"].get(ticker, [ticker.lower()]))


def classify_topic(headline: str) -> str | None:
    """First-match rule-based tag. Deterministic and auditable; no LLM in the loop."""
    low = headline.lower()
    for topic, keywords in CFG["news"]["topics"].items():
        if any(k in low for k in keywords):
            return topic
    return None


def fetch_feed(url: str, limit: int | None = None) -> list:
    parsed = feedparser.parse(url)
    entries = parsed.entries or []
    return entries[: limit or CFG["news"]["max_per_feed"]]


def feed_urls(ticker: str, days: int) -> dict[str, str]:
    """Every feed URL for one ticker, keyed by feed name."""
    news = CFG["news"]
    query = news["queries"].get(ticker, ticker).replace(" ", "+")
    urls = {
        "google_news": news["feeds"]["google_news"].format(query=query, days=days),
        "yahoo": news["feeds"]["yahoo"].format(ticker=ticker),
    }
    ir = news.get("ir_feeds", {}).get(ticker)
    if ir:
        urls["ir"] = ir
    return urls


def collect(ticker: str, days: int | None = None) -> list[NewsItem]:
    """Every in-window, relevant, URL-unique item for one ticker across its feeds."""
    days = days or CFG["news"]["lookback_days"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    seen: dict[str, NewsItem] = {}

    for feed_name, url in feed_urls(ticker, days).items():
        for entry in fetch_feed(url):
            link = (entry.get("link") or "").strip()
            headline = (entry.get("title") or "").strip()
            published = to_utc(entry)
            if not link or not headline or published is None or published < cutoff:
                continue
            # IR feeds are first-party by definition — everything on NVIDIA's newsroom
            # is NVIDIA news even when the headline never says "NVIDIA".
            if feed_name != "ir" and not is_relevant(headline, ticker):
                continue
            aid = article_id(link)
            if aid in seen:                     # exact-URL dedup, pass 1
                continue
            seen[aid] = NewsItem(
                article_id=aid, ticker=ticker, headline=headline,
                source=entry_source(entry, feed_name), published_at=published,
                url=link, topic=classify_topic(headline), feed=feed_name,
            )
    return sorted(seen.values(), key=lambda i: i.published_at)


def fetch_text(item: NewsItem) -> str | None:
    """Article body via trafilatura. None for Google redirects and for publishers that
    block extraction — both are normal, and the brief must work on headlines alone."""
    if is_google_redirect(item.url):
        return None
    import trafilatura
    downloaded = trafilatura.fetch_url(item.url)
    if not downloaded:
        return None
    return trafilatura.extract(downloaded) or None


def hydrate_text(items: list[NewsItem]) -> int:
    """Fetch bodies for the items that can have one. Returns how many succeeded."""
    if not CFG["news"].get("fetch_text", True):
        return 0
    n = 0
    for item in items:
        if item.is_duplicate or is_google_redirect(item.url):
            continue
        try:
            item.text = fetch_text(item)
        except Exception:
            item.text = None                    # a dead link must not kill the run
        n += bool(item.text)
    return n


# --- dedup and clustering ----------------------------------------------------

# Google appends " - Publisher" to every title; tickers arrive as "(NASDAQ:NVDA)".
PUBLISHER_SUFFIX_RE = re.compile(r"\s+-\s+[^-]{2,30}$")
EXCHANGE_RE = re.compile(r"\((?:NASDAQ|NYSE)[:\s][^)]*\)", re.I)


def normalize_headline(headline: str, ticker: str) -> str:
    """Strip the parts every headline about one company shares, before embedding.

    Measured on 67 live NVDA headlines: raw mean pairwise cosine is 0.714 and 690 pairs
    clear the 0.75 clustering threshold; with the company name, exchange tag and
    publisher suffix removed it is 0.588 and 51 pairs. The company name alone
    contributes ~0.13 of *constant* similarity to every pair, which is enough to push
    unrelated stories over the cut — that is why the configured thresholds produced one
    39-article NVDA "event" containing four unrelated opinion pieces.

    Only the vectors use this. The stored headline stays verbatim, because the brief
    has to show what was actually published.
    """
    text = PUBLISHER_SUFFIX_RE.sub("", headline)
    text = EXCHANGE_RE.sub(" ", text)
    for alias in CFG["news"]["aliases"].get(ticker, [ticker.lower()]) + [f"${ticker}"]:
        text = re.sub(rf"\b{re.escape(alias)}\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" :-\u2013\u2014")
    return text or headline          # never hand an empty string to the encoder


def _headline_vectors(items: list[NewsItem]):
    from src.common.embeddings import embed_passages
    return embed_passages([normalize_headline(i.headline, i.ticker) for i in items])


def mark_near_duplicates(items: list[NewsItem], threshold: float | None = None) -> int:
    """Flag syndicated re-posts of a story already seen. Earliest copy wins.

    Exact-URL dedup cannot do this job: Google News assigns a *unique* redirect URL to
    every item, so twenty outlets running the same wire story arrive as twenty distinct
    URLs with near-identical headlines. Left alone they inflate that story's article
    count and hand it the top slot in the brief purely on volume.

    Items are processed oldest-first and compared against the kept set, so the survivor
    is the earliest publication — the one that broke the story. Duplicates are kept in
    the table (flagged) rather than dropped, because the duplicate *count* is itself a
    signal, and because dropping them would make the dedup unauditable.
    """
    threshold = threshold or CFG["news"]["dedup_headline_sim"]
    if len(items) < 2:
        return 0
    items.sort(key=lambda i: i.published_at)
    vecs = _headline_vectors(items)             # L2-normalized, so dot == cosine
    kept: list[int] = []
    n_dupes = 0
    for i in range(len(items)):
        best_j, best_sim = None, 0.0
        for j in kept:
            sim = float(vecs[i] @ vecs[j])
            if sim > best_sim:
                best_j, best_sim = j, sim
        if best_j is not None and best_sim >= threshold:
            items[i].is_duplicate = True
            items[i].duplicate_of = items[best_j].article_id
            n_dupes += 1
        else:
            kept.append(i)
    return n_dupes


def cluster_events(items: list[NewsItem], threshold: float | None = None) -> int:
    """Assign `event_id` by agglomerative clustering over headline embeddings.

    Config gives a *similarity* threshold (0.75); sklearn wants a distance, and cosine
    distance is 1 - cosine similarity, so the cut is at 0.25. Average linkage, because
    a story's coverage is a loose cloud of paraphrases: single linkage chains unrelated
    stories together through one ambiguous headline, complete linkage splits genuine
    coverage of one event into several.

    Only non-duplicate items are clustered — syndicated copies would otherwise dominate
    a cluster's geometry and pull its centroid onto the wire copy's phrasing.
    """
    threshold = threshold or CFG["news"]["event_cluster_threshold"]
    live = [i for i in items if not i.is_duplicate]
    if not live:
        return 0
    if len(live) == 1:
        live[0].event_id = _event_id(live[0].ticker, [live[0]])
        return 1

    from sklearn.cluster import AgglomerativeClustering
    vecs = _headline_vectors(live)
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.0 - threshold,
        metric="cosine", linkage="average",
    ).fit_predict(vecs)

    groups: dict[int, list[NewsItem]] = {}
    for item, label in zip(live, labels):
        groups.setdefault(int(label), []).append(item)
    for members in groups.values():
        eid = _event_id(members[0].ticker, members)
        for m in members:
            m.event_id = eid
    # Duplicates inherit their original's event so the brief can count them.
    by_id = {i.article_id: i for i in items}
    for item in items:
        if item.is_duplicate and item.duplicate_of in by_id:
            item.event_id = by_id[item.duplicate_of].event_id
    return len(groups)


def _event_id(ticker: str, members: list[NewsItem]) -> str:
    """Stable while the cluster's earliest member is stable — re-running ingestion on
    an unchanged window reproduces the same ids instead of churning them."""
    earliest = min(members, key=lambda m: m.published_at)
    digest = hashlib.sha1(f"{ticker}:{earliest.article_id}".encode()).hexdigest()[:12]
    return f"evt-{ticker}-{digest}"


# --- persistence -------------------------------------------------------------

COLS = ["article_id", "ticker", "headline", "source", "published_at", "url",
        "text", "topic", "event_id", "is_duplicate"]


def write_articles(conn, items: list[NewsItem]) -> int:
    """Upsert on article_id = sha1(url), so re-running a window restates rather than
    duplicating. Clustering is recomputed each run, so event_id is always overwritten."""
    if not items:
        return 0
    ph = placeholder()
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in COLS if c != "article_id")
    sql = (f"INSERT INTO news_articles ({', '.join(COLS)}) "
           f"VALUES ({', '.join([ph] * len(COLS))}) "
           f"ON CONFLICT (article_id) DO UPDATE SET {updates}")
    rows = [
        (i.article_id, i.ticker, i.headline, i.source,
         i.published_at.isoformat() if storage_mode() == "local" else i.published_at,
         i.url, i.text, i.topic, i.event_id, i.is_duplicate)
        for i in items
    ]
    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def ingest_ticker(conn, ticker: str, days: int | None = None,
                  with_text: bool = True) -> dict:
    """Full pipeline for one ticker. Returns a stats dict for the CLI and the gate."""
    items = collect(ticker, days)
    n_dupes = mark_near_duplicates(items)
    n_events = cluster_events(items)
    n_text = hydrate_text(items) if with_text else 0
    written = write_articles(conn, items)
    return {"ticker": ticker, "collected": len(items), "duplicates": n_dupes,
            "events": n_events, "with_text": n_text, "written": written,
            "feeds": {f: sum(1 for i in items if i.feed == f) for f in
                      {i.feed for i in items}}}


def ingest_all(days: int | None = None, with_text: bool = True) -> list[dict]:
    conn = get_conn()
    try:
        return [ingest_ticker(conn, t, days, with_text) for t in CFG["tickers"]]
    finally:
        conn.close()
