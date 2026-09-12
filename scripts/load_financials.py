"""Phase 4 CLI: SEC companyfacts XBRL -> `financials`, 8-K outlook blocks -> `guidance`.

Idempotent — upserts on (ticker, fiscal_year, fiscal_qtr, metric), so re-running
restates values rather than duplicating rows. The companyfacts JSON (4-9 MB/ticker)
is cached under data/raw/{ticker}/companyfacts.json; `--refresh` re-pulls it.

    python scripts/load_financials.py            # load every ticker in config.yaml
    python scripts/load_financials.py --refresh  # re-pull companyfacts from SEC first
    python scripts/load_financials.py --ticker NVDA
    python scripts/load_financials.py --skip-guidance

Guidance is not in XBRL — it is prose in the EX-99.1 earnings release, so it is
extracted from the chunks Phase 2 already indexed. Run scripts/build_index.py first
or that half is a no-op.
"""
import argparse

from src.common.config import CFG
from src.common.db import get_conn
from src.structured import financials, guidance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", action="append", help="limit to ticker(s); repeatable")
    ap.add_argument("--refresh", action="store_true", help="re-fetch companyfacts from SEC")
    ap.add_argument("--skip-guidance", action="store_true", help="XBRL metrics only")
    args = ap.parse_args()

    tickers = {t: CFG["tickers"][t] for t in (args.ticker or CFG["tickers"])}
    from src.ingestion.edgar import sec_session
    session, limiter = sec_session()

    conn = get_conn()
    try:
        added = financials.ensure_schema(conn)
        if added:
            print(f"migrated `financials`: added {', '.join(added)}")
        total = 0
        for ticker, info in tickers.items():
            n, counts, tags = financials.load_ticker(conn, ticker, info["cik"], args.refresh,
                                                     session, limiter)
            total += n
            print(f"  {ticker:5s} {n:4d} rows  {counts}")
            for metric, tag in tags.items():
                print(f"        {metric:18s} <- {tag or 'NOT REPORTED by this filer'}")
        print(f"\n{total} financial rows upserted")

        if not args.skip_guidance:
            n_guide = sum(guidance.load_all(conn, t) for t in tickers)
            print(f"{n_guide} guidance rows extracted from ingested 8-K earnings releases")

        cur = conn.cursor()
        cur.execute("SELECT ticker, metric, count(*), min(fiscal_year), max(fiscal_year) "
                    "FROM financials GROUP BY 1,2 ORDER BY 1,2")
        print(f"\n{'ticker':7s} {'metric':18s} {'rows':>5s}  years")
        for t, m, n, lo, hi in cur.fetchall():
            print(f"{t:7s} {m:18s} {n:5d}  FY{lo}-FY{hi}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
