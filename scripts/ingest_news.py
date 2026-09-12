"""Phase 5 CLI: RSS -> relevance filter -> dedup -> event clusters -> `news_articles`.

Idempotent: upserts on article_id = sha1(url). Clustering is recomputed over the
whole window each run, so event assignments always reflect the current corpus.

    python scripts/ingest_news.py                  # every ticker, config lookback
    python scripts/ingest_news.py --days 14
    python scripts/ingest_news.py --ticker NVDA --no-text
"""
import argparse

from src.common.config import CFG
from src.common.db import get_conn
from src.news import ingest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", action="append", help="limit to ticker(s); repeatable")
    ap.add_argument("--days", type=int, help="lookback window (default: config)")
    ap.add_argument("--no-text", action="store_true",
                    help="skip trafilatura body fetch (much faster)")
    args = ap.parse_args()

    conn = get_conn()
    try:
        totals = {"collected": 0, "duplicates": 0, "events": 0, "with_text": 0}
        for ticker in (args.ticker or CFG["tickers"]):
            s = ingest.ingest_ticker(conn, ticker, args.days, with_text=not args.no_text)
            for k in totals:
                totals[k] += s[k]
            print(f"  {ticker:5s} {s['collected']:3d} articles  {s['duplicates']:2d} near-dupes  "
                  f"{s['events']:3d} events  {s['with_text']:3d} with body text  "
                  f"feeds={s['feeds']}")
        print(f"\n{totals['collected']} articles, {totals['duplicates']} near-duplicates "
              f"suppressed, {totals['events']} events, {totals['with_text']} with body text")

        cur = conn.cursor()
        cur.execute("SELECT ticker, count(*), count(DISTINCT event_id), count(DISTINCT source) "
                    "FROM news_articles GROUP BY ticker ORDER BY ticker")
        print(f"\n{'ticker':7s} {'articles':>8s} {'events':>7s} {'sources':>8s}")
        for t, n, e, s in cur.fetchall():
            print(f"{t:7s} {n:8d} {e:7d} {s:8d}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
