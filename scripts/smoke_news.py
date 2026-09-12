"""Phase 5 gate: news dedup, clustering, ranking and brief assembly.

Dedup/clustering/ranking are asserted on hand-built items, not on live RSS, because
those properties must hold exactly and a feed's contents change hourly. The SQL
window filter and the rendered brief are then exercised against the real
`news_articles` table.

Read-only with respect to the news table.

    python scripts/smoke_news.py
"""
from datetime import datetime, timedelta, timezone

from src.common.config import CFG
from src.common.db import get_conn
from src.generation import llm
from src.news import brief, ingest

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def item(key, headline, source, hours, ticker="NVDA"):
    url = f"https://example.com/{key}"
    return ingest.NewsItem(
        article_id=ingest.article_id(url), ticker=ticker, headline=headline,
        source=source, published_at=NOW + timedelta(hours=hours), url=url,
        topic=ingest.classify_topic(headline), feed="test")


def main():
    # --- feed wiring -----------------------------------------------------
    urls = ingest.feed_urls("NVDA", 7)
    assert "NVIDIA" in urls["google_news"] and "when:7d" in urls["google_news"]
    assert "s=NVDA" in urls["yahoo"] and "ir" in urls           # NVDA has an IR feed
    assert "ir" not in ingest.feed_urls("JPM", 7), "JPM has no working IR feed"
    print(f"feeds ok   - NVDA: {sorted(urls)}; JPM: {sorted(ingest.feed_urls('JPM', 7))}")

    # --- relevance filter: the Yahoo feed leaks unrelated finance content --
    assert ingest.is_relevant("Nvidia beats on Q3 revenue", "NVDA")
    assert ingest.is_relevant("NVDA hits record high", "NVDA")
    assert not ingest.is_relevant("Worried About RMDs in Retirement?", "NVDA")
    assert ingest.is_relevant("JP Morgan names new CFO", "JPM")
    print("filter ok  - alias match keeps ticker news, drops generic finance syndication")

    # --- ids and the Google redirect problem ------------------------------
    assert ingest.article_id("https://a.com/x") == ingest.article_id("https://a.com/x")
    assert ingest.is_google_redirect("https://news.google.com/rss/articles/CBMiabc")
    assert not ingest.is_google_redirect("https://www.fool.com/investing/x")
    google = ingest.NewsItem(
        article_id="x", ticker="NVDA", headline="h", source="s",
        published_at=NOW, url="https://news.google.com/rss/articles/CBMiabc")
    assert ingest.fetch_text(google) is None, \
        "Google redirects must short-circuit before any network call"
    print("ids ok     - sha1(url) stable; Google redirects detected (headline-only)")

    # --- headline normalization ------------------------------------------
    norm = ingest.normalize_headline(
        "Nvidia's $279 Billion Bet Changes Everything (NASDAQ:NVDA) - Seeking Alpha", "NVDA")
    assert "Nvidia" not in norm and "NASDAQ" not in norm and "Seeking Alpha" not in norm
    assert "279 Billion Bet" in norm, norm
    assert ingest.normalize_headline("Nvidia", "NVDA") == "Nvidia", "must never return empty"
    print(f"normalize ok - {norm!r}")

    # --- dedup: syndication arrives as distinct URLs ----------------------
    items = [
        item("a", "Nvidia announces Blackwell Ultra at GTC", "Reuters", 0),
        item("b", "Nvidia announces Blackwell Ultra at GTC", "Yahoo Finance", 3),   # exact re-post
        item("c", "Nvidia unveils Blackwell Ultra accelerator at GTC", "CNBC", 5),  # paraphrase
        item("d", "JPMorgan raises dividend after stress test", "Bloomberg", 2),
    ]
    n_dupes = ingest.mark_near_duplicates(items)
    by_key = {i.url.rsplit("/", 1)[-1]: i for i in items}
    assert n_dupes >= 1, "identical syndicated headline must be flagged"
    assert by_key["b"].is_duplicate, "later identical copy must be the duplicate"
    assert not by_key["a"].is_duplicate, "earliest copy must survive"
    assert by_key["b"].duplicate_of == by_key["a"].article_id
    assert not by_key["d"].is_duplicate, "unrelated story must not be deduped"
    print(f"dedup ok   - {n_dupes} near-dupe(s); earliest kept, unrelated story untouched")

    # --- clustering -------------------------------------------------------
    n_events = ingest.cluster_events(items)
    assert n_events >= 2, f"expected the JPM story to separate, got {n_events} cluster(s)"
    assert by_key["a"].event_id and by_key["a"].event_id == by_key["c"].event_id, \
        "paraphrases of one event must share an event_id"
    assert by_key["d"].event_id != by_key["a"].event_id, "unrelated story must not merge"
    assert by_key["b"].event_id == by_key["a"].event_id, "duplicate must inherit its event"
    assert ingest.cluster_events(list(items)) == n_events, "clustering must be deterministic"
    print(f"cluster ok - {n_events} events; paraphrase merged, duplicate inherited, "
          f"unrelated separate")

    # --- ranking: the property that keeps bot spam out of the brief -------
    spam = brief.Event(event_id="spam", ticker="JPM", articles=[
        brief.Article(f"s{i}", f"JPM Shares Acquired by Fund {i} LLC", "MarketBeat",
                      NOW, f"https://e.com/s{i}", "deal", False) for i in range(18)])
    real = brief.Event(event_id="real", ticker="JPM", articles=[
        brief.Article(f"r{i}", "India lifts ban on JPMorgan unit", src, NOW,
                      f"https://e.com/r{i}", None, False)
        for i, src in enumerate(("Reuters", "Bloomberg"))])
    assert brief.score_event(real) > brief.score_event(spam), (
        f"2 sources ({brief.score_event(real):.2f}) must outrank 18 articles from 1 "
        f"source ({brief.score_event(spam):.2f})")
    assert brief.rank_events([spam, real], 2)[0] is real
    print(f"rank ok    - 2 sources={brief.score_event(real):.2f} beats "
          f"18 articles/1 source={brief.score_event(spam):.2f} (volume capped at "
          f"{brief.VOLUME_CAP})")

    # --- extractive summary makes no unsupported claim --------------------
    summarized = brief.summarize_event(real, "JPM", use_llm=False)
    assert summarized.summary_method == "extractive"
    assert "Reuters" in summarized.summary and "India lifts ban" in summarized.summary
    for invented in ("bull", "bear", "should buy", "will rise"):
        assert invented not in summarized.summary.lower(), \
            f"extractive fallback must not invent {invented!r}"
    print("summary ok - extractive fallback cites sources, invents no implication")

    # --- LLM seam degrades, never crashes ---------------------------------
    if llm.available():
        print(f"llm ok     - backend {llm.backend()!r} reachable, summaries will be generated")
    else:
        try:
            llm.complete("s", "u")
            raise AssertionError("complete() must raise when the backend is down")
        except llm.LLMUnavailable:
            pass
        print(f"llm ok     - backend {llm.backend()!r} down; raises LLMUnavailable, "
              f"brief falls back")

    # --- real corpus: SQL window filter + rendered brief ------------------
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*), count(DISTINCT event_id), count(DISTINCT ticker) "
                    "FROM news_articles")
        n_art, n_ev, n_tick = cur.fetchone()
        assert n_art and n_ev, "news_articles is empty — run scripts/ingest_news.py first"
        print(f"\ncorpus: {n_art} articles, {n_ev} events, {n_tick} tickers")

        days = CFG["news"]["lookback_days"]
        for ticker in CFG["tickers"]:
            events = brief.load_events(conn, ticker, days)
            assert events, f"no {ticker} events in the last {days} days"
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            for e in events:
                assert all(a.ticker if hasattr(a, "ticker") else True for a in e.articles)
                assert all(a.published_at >= cutoff for a in e.live), \
                    f"{ticker}: SQL window filter leaked an out-of-range article"
                assert all(not a.is_duplicate for a in e.live)
            ranked = brief.rank_events(events, 3)
            assert [brief.score_event(e) for e in ranked] == sorted(
                [brief.score_event(e) for e in ranked], reverse=True)
            top = ranked[0]
            print(f"  {ticker:5s} {len(events):3d} events; top: {len(top.live)} article(s) / "
                  f"{len(top.sources)} source(s) — {top.primary.headline[:52]}")

        text = brief.weekly_brief(conn, "NVDA", days, top_n=3, use_llm=False)
        assert text.startswith("# NVDA") and "Weekly intelligence brief" in text
        assert "**Sources**" in text and "http" in text, "brief must link its sources"
        assert "extractive" in text, "extractive briefs must say so"
        assert str(datetime.now(timezone.utc).year) in text, "brief must be dated"
        print(f"\nbrief ok   - {len(text.splitlines())} lines, sources linked and dated")
    finally:
        conn.close()

    print("\nPHASE 5 NEWS PIPELINE PASSED")


if __name__ == "__main__":
    main()
