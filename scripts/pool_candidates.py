"""Pool retriever candidates for manual relevance judging (TREC-style).

For each retrieval-graded question, take the top-N from *every* retrieval mode and
union them. Judging only the pool is standard practice, and its bias is worth stating:
a chunk no retriever surfaced can never be judged relevant, so recall is measured
against the pool rather than against the corpus. Pooling all four arms (rather than
the one being tuned) is what keeps the comparison between arms fair — no arm gets to
define the ground truth it is then scored against.

    python scripts/pool_candidates.py --out /tmp/pool.txt --depth 8
"""
import argparse
import sys

from src.common.db import get_conn
from src.common.models import Filters
from src.retrieval.hybrid import MODES, Retriever

sys.path.insert(0, "evals")
from questions_draft import QUESTIONS  # noqa: E402

# insufficient_evidence questions are graded on the refusal, not on chunks.
POOLED_ROUTES = {"FILING_RAG", "MIXED", "CROSS_SOURCE"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--only", help="comma-separated question ids")
    args = ap.parse_args()

    wanted = set(args.only.split(",")) if args.only else None
    conn = get_conn()
    retr = Retriever(conn=conn)
    lines = []
    try:
        for qid, category, route, query, tickers in QUESTIONS:
            if category == "insufficient_evidence" or route not in POOLED_ROUTES:
                continue
            if wanted and qid not in wanted:
                continue
            filters = Filters(tickers=tickers or None)
            pool: dict[str, dict] = {}
            for mode in MODES:
                for rank, ch in enumerate(retr.search(query, mode=mode, top_k=args.depth,
                                                      filters=filters), start=1):
                    e = pool.setdefault(ch.meta.chunk_id, {"ch": ch, "ranks": {}})
                    e["ranks"][mode] = rank

            lines.append(f"\n{'='*100}\n{qid} [{category}] {query}\n{'='*100}")
            ordered = sorted(pool.values(),
                             key=lambda e: min(e["ranks"].values()))
            for e in ordered:
                ch, m = e["ch"], e["ch"].meta
                ranks = " ".join(f"{k}={v}" for k, v in sorted(e["ranks"].items()))
                body = " ".join((ch.text or "").split())[:260]
                lines.append(f"\n  {m.chunk_id}")
                lines.append(f"    {m.ticker} {m.doc_type} {m.fiscal_period} "
                             f"| {m.section} | filed {m.filing_date} | {ranks}")
                lines.append(f"    {body}")
    finally:
        retr._owns["conn"] = False
        retr.close()
        conn.close()

    text = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
