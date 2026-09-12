"""Phase 5 CLI: weekly intelligence brief from clustered news.

    python scripts/news_brief.py --ticker NVDA
    python scripts/news_brief.py --all --days 7 --out data/briefs
    python scripts/news_brief.py --ticker JPM --no-llm     # force extractive summaries

Summaries use the configured LLM backend when reachable and fall back to extractive
(headline-derived, no interpretation) when it isn't — the brief always renders.
"""
import argparse
from pathlib import Path

from src.common.config import CFG
from src.common.db import get_conn
from src.generation.llm import available, backend
from src.news.brief import weekly_brief

ROOT = Path(__file__).parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", action="append", help="ticker(s); repeatable")
    ap.add_argument("--all", action="store_true", help="every ticker in config.yaml")
    ap.add_argument("--days", type=int, help="lookback window (default: config)")
    ap.add_argument("--top", type=int, default=5, help="events per brief (default 5)")
    ap.add_argument("--no-llm", action="store_true", help="force extractive summaries")
    ap.add_argument("--out", help="directory to write <TICKER>.md into")
    args = ap.parse_args()

    tickers = list(CFG["tickers"]) if args.all else (args.ticker or ["NVDA"])
    use_llm = not args.no_llm
    if use_llm and not available():
        print(f"note: LLM backend {backend()!r} unreachable — extractive summaries\n")

    conn = get_conn()
    try:
        for ticker in tickers:
            text = weekly_brief(conn, ticker, args.days, args.top, use_llm)
            if args.out:
                path = ROOT / args.out
                path.mkdir(parents=True, exist_ok=True)
                (path / f"{ticker}.md").write_text(text)
                print(f"wrote {path / f'{ticker}.md'}")
            else:
                print(text)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
