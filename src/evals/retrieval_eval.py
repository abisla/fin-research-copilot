"""Retrieval evaluation: dense vs BM25 vs hybrid vs hybrid+rerank.

Four arms, one question set, identical filters and identical candidate widths — the
`Retriever` facade exists so the only thing that varies between arms is the ranking
strategy (DECISIONS #17). Anything else and the comparison measures plumbing.

**Metrics, and why three of them.**

* `Recall@5`  — share of the relevant set found in the top 5. On this corpus the
  relevant sets run 2-30 chunks, so Recall@5 is *capped* at `5/|relevant|` and a
  question with 30 relevant chunks can never score above 0.17. Reported as-is rather
  than normalized, with the ceiling printed alongside, because silently rescaling a
  metric to look better is how eval tables start lying.
* `Precision@5` — share of the top 5 that is relevant. Unaffected by set size, so it is
  the honest cross-question comparison.
* `MRR` — reciprocal rank of the *first* relevant chunk. This is the one that matters
  for generation: the answer only needs one good chunk near the top, and a reranker
  that moves a relevant chunk from rank 4 to rank 1 shows up here and nowhere else.
* `Hit@5` — did any relevant chunk appear at all. The floor below which the generator
  cannot possibly answer.

    python -m src.evals.retrieval_eval
    python -m src.evals.retrieval_eval --modes dense,rerank --limit 5
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from src.common.db import get_conn
from src.common.models import Filters
from src.indexing.store import placeholder
from src.retrieval.hybrid import MODES, Retriever

ROOT = Path(__file__).parents[2]
QUESTIONS = ROOT / "evals/questions.jsonl"
RESULTS_MD = ROOT / "evals/results.md"
K = 5


def load_questions(path=QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score(retrieved_ids: list[str], relevant: set[str], current: set[str] | None = None) -> dict:
    """Recall@5, Precision@5, MRR and Hit@5 for one ranked list.

    `current` is the subset of the relevant set that lives in the company's *current*
    filing. It is scored separately rather than folded into relevance, because the
    first attempt did fold it in and the result was misleading: four of five "misses"
    on sem-01 were the identical export-control paragraph retrieved from a 10-Q instead
    of the 10-K, which is a recency preference problem, not a retrieval-quality one.
    Conflating them made every arm look broken and hid which defect was which.
    """
    top = retrieved_ids[:K]
    hits = [cid for cid in top if cid in relevant]
    rr = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            rr = 1.0 / rank
            break
    fresh = [cid for cid in hits if current and cid in current]
    return {
        "recall_at_5": len(hits) / len(relevant) if relevant else 0.0,
        "precision_at_5": len(hits) / len(top) if top else 0.0,
        "mrr": rr,
        "hit_at_5": bool(hits),
        "ceiling": min(K, len(relevant)) / len(relevant) if relevant else 0.0,
        # Of the relevant chunks retrieved, what share came from the current filing.
        "freshness": len(fresh) / len(hits) if hits else 0.0,
    }


def evaluate(modes: list[str] | None = None, limit: int | None = None) -> dict:
    modes = modes or list(MODES)
    questions = [q for q in load_questions() if q.get("expected_chunk_ids")]
    if limit:
        questions = questions[:limit]

    conn = get_conn()
    retr = Retriever(conn=conn)
    rows: list[dict] = []
    try:
        for q in questions:
            relevant = set(q["expected_chunk_ids"])
            current = set(q.get("expected_current_ids") or [])
            filters = Filters(tickers=q["tickers"] or None)
            for mode in modes:
                started = time.perf_counter()
                hits = retr.search(q["query"], mode=mode, top_k=K, filters=filters)
                elapsed = (time.perf_counter() - started) * 1000
                s = score([h.meta.chunk_id for h in hits], relevant, current)
                rows.append({"question_id": q["id"], "category": q["category"],
                             "retriever": mode, "n_relevant": len(relevant),
                             "latency_ms": elapsed, **s})
    finally:
        retr._owns["conn"] = False
        retr.close()
        conn.close()
    return {"rows": rows, "modes": modes, "n_questions": len(questions)}


def aggregate(rows: list[dict], key: str = "retriever") -> dict[str, dict]:
    out: dict[str, dict] = {}
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    for name, group in groups.items():
        out[name] = {
            "n": len(group),
            "recall_at_5": statistics.mean(r["recall_at_5"] for r in group),
            "precision_at_5": statistics.mean(r["precision_at_5"] for r in group),
            "mrr": statistics.mean(r["mrr"] for r in group),
            "hit_at_5": statistics.mean(float(r["hit_at_5"]) for r in group),
            "ceiling": statistics.mean(r["ceiling"] for r in group),
            "freshness": statistics.mean(r["freshness"] for r in group),
            "latency_ms": statistics.median(r["latency_ms"] for r in group),
        }
    return out


def write_db(conn, run_id: str, rows: list[dict]) -> int:
    ph = placeholder()
    cur = conn.cursor()
    cur.executemany(
        f"INSERT INTO eval_results (run_id, question_id, category, retriever, "
        f"recall_at_5, precision_at_5, hit_at_5, mrr) "
        f"VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}) "
        f"ON CONFLICT (run_id, question_id, retriever) DO UPDATE SET "
        f"recall_at_5 = EXCLUDED.recall_at_5, precision_at_5 = EXCLUDED.precision_at_5, "
        f"hit_at_5 = EXCLUDED.hit_at_5, mrr = EXCLUDED.mrr",
        [(run_id, r["question_id"], r["category"], r["retriever"],
          r["recall_at_5"], r["precision_at_5"], r["hit_at_5"], r["mrr"]) for r in rows])
    conn.commit()
    return len(rows)


def _table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def render_markdown(result: dict, run_id: str) -> str:
    rows = result["rows"]
    overall = aggregate(rows)
    order = [m for m in MODES if m in overall]

    out = [f"# Retrieval evaluation\n",
           f"Run `{run_id}` · {result['n_questions']} retrieval-graded questions · "
           f"top-{K}\n",
           "Ground truth is built by `scripts/build_questions.py` from explicit, "
           "re-runnable relevance rules; see that module for the method and its bias.\n",
           "## Overall\n"]
    body = []
    for m in order:
        a = overall[m]
        body.append([f"`{m}`", f"{a['recall_at_5']:.3f}", f"{a['precision_at_5']:.3f}",
                     f"{a['mrr']:.3f}", f"{a['hit_at_5']:.3f}", f"{a['freshness']:.3f}",
                     f"{a['latency_ms']:.0f}"])
    out.append(_table(["retriever", "Recall@5", "Precision@5", "MRR", "Hit@5",
                       "Freshness", "median ms"], body))
    rel_sizes = sorted({r["n_relevant"] for r in rows})
    out.append(f"\nRecall@5 ceiling on this question set is "
               f"**{overall[order[0]]['ceiling']:.3f}** — relevant sets run "
               f"{rel_sizes[0]}-{rel_sizes[-1]} chunks, so recall is bounded by "
               f"`5/|relevant|`. Reported unnormalized on purpose. Precision@5 and MRR "
               f"are unaffected by set size and are the fair cross-question comparison; "
               f"**Freshness** is the share of retrieved relevant chunks that came from "
               f"the company's current filing (FC-10).\n")

    out.append("## By category (MRR)\n")
    cats = sorted({r["category"] for r in rows})
    body = []
    for cat in cats:
        sub = [r for r in rows if r["category"] == cat]
        agg = aggregate(sub)
        body.append([cat] + [f"{agg[m]['mrr']:.3f}" if m in agg else "—" for m in order])
    out.append(_table(["category"] + [f"`{m}`" for m in order], body))

    out.append("\n## By category (Precision@5)\n")
    body = []
    for cat in cats:
        sub = [r for r in rows if r["category"] == cat]
        agg = aggregate(sub)
        body.append([cat] + [f"{agg[m]['precision_at_5']:.3f}" if m in agg else "—"
                             for m in order])
    out.append(_table(["category"] + [f"`{m}`" for m in order], body))

    out.append("\n## Per question (MRR)\n")
    body = []
    for qid in sorted({r["question_id"] for r in rows},
                      key=lambda x: (x.split("-")[0], x)):
        sub = {r["retriever"]: r for r in rows if r["question_id"] == qid}
        n_rel = next(iter(sub.values()))["n_relevant"]
        body.append([qid, str(n_rel)] +
                    [f"{sub[m]['mrr']:.2f}" if m in sub else "—" for m in order])
    out.append(_table(["question", "|rel|"] + [f"`{m}`" for m in order], body))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", help="comma-separated subset of " + ",".join(MODES))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    modes = args.modes.split(",") if args.modes else list(MODES)
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%S")
    result = evaluate(modes, args.limit)

    overall = aggregate(result["rows"])
    print(f"\nrun {run_id} — {result['n_questions']} questions, top-{K}\n")
    print(f"{'retriever':10s} {'Recall@5':>9s} {'Prec@5':>8s} {'MRR':>7s} "
          f"{'Hit@5':>7s} {'Fresh':>7s} {'ms':>6s}")
    for m in [x for x in MODES if x in overall]:
        a = overall[m]
        print(f"{m:10s} {a['recall_at_5']:9.3f} {a['precision_at_5']:8.3f} "
              f"{a['mrr']:7.3f} {a['hit_at_5']:7.3f} {a['freshness']:7.3f} "
              f"{a['latency_ms']:6.0f}")

    if not args.no_write:
        conn = get_conn()
        try:
            n = write_db(conn, run_id, result["rows"])
        finally:
            conn.close()
        RESULTS_MD.write_text(render_markdown(result, run_id))
        print(f"\nwrote {n} rows to eval_results and {RESULTS_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
