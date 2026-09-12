"""Answer evaluation: LLM-as-judge over groundedness, citation correctness, relevance.

The judge gets a **separate prompt from the generator** and never sees the question's
expected answer — only the question, the numbered context that was actually sent, and
the answer. Sharing a prompt would measure self-consistency rather than quality.

**What the judge is and is not trusted with.** It scores three things 1-5 and lists any
claim it cannot map to the context. It is *not* trusted with citation validity: whether
`[7]` exists when six items were supplied is decidable, so `answer.check_citations`
decides it and the judge's opinion is ignored. Same for the insufficient-evidence set —
whether the answer refused is a string test, not a judgement call. A judge is for the
part that genuinely needs reading; everything decidable is decided in code, because a
judge that can be wrong about arithmetic will eventually be wrong about arithmetic.

FC-6 is the reason groundedness is scored against the *context*, not against the world:
a claim can be true about NVIDIA and still be a hallucination here if nothing in the
supplied chunks supports it.

    python -m src.evals.answer_eval
    python -m src.evals.answer_eval --limit 5 --out evals/answer_eval.json
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

from src.common.db import get_conn
from src.generation import answer as A
from src.generation.llm import LLMUnavailable, available, complete
from src.generation.prompts import INSUFFICIENT
from src.indexing.store import placeholder
from src.retrieval.hybrid import Retriever
from src.router import router as R

ROOT = Path(__file__).parents[2]
QUESTIONS = ROOT / "evals/questions.jsonl"
OUT_JSON = ROOT / "evals/answer_eval.json"

JUDGE_SYSTEM = """You grade a financial assistant's answer against the context it was given.
You are grading faithfulness to that context, NOT whether the claim is true in the world:
a statement that is factually correct about the company is still UNSUPPORTED here if the
numbered context does not contain it.

Score each 1-5 (5 is best):
- groundedness: every claim traceable to the context. 5 = all claims supported;
  1 = mostly invented.
- citation_correctness: claims carry [n] markers that point at the item actually
  containing them. 5 = every claim correctly cited; 1 = citations absent or misattributed.
- relevance: the answer addresses the question asked. 5 = fully; 1 = off-topic.

Also list every claim you cannot map to the context, verbatim and short.

An answer that says "INSUFFICIENT EVIDENCE" when the context genuinely lacks the answer
is CORRECT: score groundedness 5 and relevance 5. Refusing is not a failure.

Reply with JSON only, no prose:
{"groundedness": n, "citation_correctness": n, "relevance": n,
 "unsupported_claims": ["..."], "note": "one sentence"}"""

JSON_RE = re.compile(r"\{.*\}", re.S)

# A refusal in prose rather than in the contract's words. The first version of this eval
# tested only for the INSUFFICIENT EVIDENCE prefix and therefore scored ie-06 — "There is
# no mention of NVIDIA's acquisition of Intel in the provided context" — as a failure to
# refuse, which is the opposite of what happened. Both are now measured: `refused` is the
# behaviour (did it decline), `used_sentinel` is contract compliance (did it decline in
# the words the prompt specified). Conflating them punishes the right behaviour for the
# wrong reason.
REFUSAL_RE = re.compile(
    r"insufficient evidence|no mention of|does not (?:mention|contain|discuss|provide|specify)|"
    r"not (?:mentioned|provided|specified|discussed|contained|available) in the (?:provided )?context|"
    r"no information (?:about|on|regarding)|context does not", re.I)


def judge(question: str, context: str, answer_text: str) -> dict:
    user = (f"QUESTION:\n{question}\n\nNUMBERED CONTEXT:\n{context}\n\n"
            f"ANSWER:\n{answer_text}")
    try:
        raw = complete(JUDGE_SYSTEM, user, temperature=0.0)
    except LLMUnavailable as e:
        return {"error": f"judge unavailable: {e}"}
    m = JSON_RE.search(raw)
    if not m:
        return {"error": "judge returned no JSON", "raw": raw[:300]}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"error": "judge JSON did not parse", "raw": raw[:300]}
    out = {}
    for k in ("groundedness", "citation_correctness", "relevance"):
        try:
            out[k] = max(1, min(5, int(data.get(k, 0))))
        except (TypeError, ValueError):
            out[k] = None
    claims = data.get("unsupported_claims") or []
    out["unsupported_claims"] = [str(c) for c in claims if str(c).strip()][:10]
    out["note"] = str(data.get("note", ""))[:300]
    return out


def evaluate(limit: int | None = None, mode: str = "rerank") -> list[dict]:
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines()
                 if line.strip()]
    if limit:
        questions = questions[:limit]

    conn = get_conn()
    retr = Retriever(conn=conn)
    results = []
    try:
        for q in questions:
            started = time.perf_counter()
            decision = R.route_and_log(conn, q["query"], use_llm=True)
            evidence = A.gather(conn, decision, retr, mode=mode)
            ans = A.generate(q["query"], evidence, decision)
            elapsed = (time.perf_counter() - started) * 1000

            # Decidable in code — the judge is not consulted on any of these.
            expects_refusal = q["category"] == "insufficient_evidence"
            used_sentinel = ans.text.upper().startswith(INSUFFICIENT)
            refused = used_sentinel or bool(REFUSAL_RE.search(ans.text[:400]))
            rec = {
                "id": q["id"], "category": q["category"], "query": q["query"],
                "route": ans.route, "expected_route": q["route"],
                "route_correct": ans.route == q["route"],
                "n_evidence": len(evidence),
                "n_citations": len(ans.citations),
                "hallucinated_citations": ans.hallucinated_citations,
                "citations_valid": not ans.hallucinated_citations,
                "uncited": ans.uncited,
                "refused": refused,
                "used_sentinel": used_sentinel,
                "refusal_correct": refused == expects_refusal,
                "sentinel_correct": used_sentinel == expects_refusal,
                "latency_ms": round(elapsed),
                "answer": ans.text[:1500],
            }
            if evidence and not expects_refusal:
                rec["judge"] = judge(q["query"], A.build_context(evidence), ans.text)
            elif expects_refusal:
                # Still judged: a refusal that smuggles in claims is the failure mode.
                rec["judge"] = judge(q["query"], A.build_context(evidence) or "(none)",
                                     ans.text)
            results.append(rec)
            j = rec.get("judge", {})
            print(f"  {q['id']:8s} {ans.route:13s} ev={len(evidence):2d} "
                  f"cit={len(ans.citations):2d} "
                  f"g={j.get('groundedness', '-')} c={j.get('citation_correctness', '-')} "
                  f"r={j.get('relevance', '-')} "
                  f"{'REFUSED' if refused else ''}")
    finally:
        retr._owns["conn"] = False
        retr.close()
        conn.close()
    return results


def summarize(results: list[dict]) -> dict:
    def mean(key):
        vals = [r["judge"][key] for r in results
                if isinstance(r.get("judge"), dict) and r["judge"].get(key) is not None]
        return statistics.mean(vals) if vals else None

    hallucinated = [r for r in results
                    if isinstance(r.get("judge"), dict) and r["judge"].get("unsupported_claims")]
    return {
        "n": len(results),
        "groundedness": mean("groundedness"),
        "citation_correctness": mean("citation_correctness"),
        "relevance": mean("relevance"),
        "route_accuracy": statistics.mean(float(r["route_correct"]) for r in results),
        "citations_valid_rate": statistics.mean(float(r["citations_valid"]) for r in results),
        "uncited_rate": statistics.mean(float(r["uncited"]) for r in results),
        "refusal_accuracy": statistics.mean(float(r["refusal_correct"]) for r in results),
        "sentinel_accuracy": statistics.mean(float(r.get("sentinel_correct",
                                                        r["refusal_correct"]))
                                             for r in results),
        "answers_with_unsupported_claims": len(hallucinated),
        "hallucination_rate": len(hallucinated) / len(results) if results else 0.0,
    }


def write_db(conn, run_id: str, results: list[dict]) -> int:
    ph = placeholder()
    rows = []
    for r in results:
        j = r.get("judge") or {}
        rows.append((run_id, r["id"], r["category"], "answer",
                     j.get("groundedness"), bool(r["citations_valid"]),
                     bool(j.get("unsupported_claims"))))
    cur = conn.cursor()
    cur.executemany(
        f"INSERT INTO eval_results (run_id, question_id, category, retriever, "
        f"groundedness, citation_ok, hallucinated) "
        f"VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}) "
        f"ON CONFLICT (run_id, question_id, retriever) DO UPDATE SET "
        f"groundedness = EXCLUDED.groundedness, citation_ok = EXCLUDED.citation_ok, "
        f"hallucinated = EXCLUDED.hallucinated", rows)
    conn.commit()
    return len(rows)


ANSWER_MARKER = "<!-- answer-eval -->"


def append_results_md(summary: dict, results: list[dict], run_id: str,
                      path: Path = ROOT / "evals/results.md") -> None:
    """Append (or replace) the answer-quality section of results.md.

    One results file is the reader's entry point, so the answer numbers live next to the
    retrieval numbers. The marker makes the section idempotent — a re-run replaces it
    instead of stacking another copy.
    """
    rows = []
    cats = sorted({r["category"] for r in results})
    for cat in cats:
        sub = [r for r in results if r["category"] == cat]
        def m(key):
            vals = [r["judge"][key] for r in sub
                    if isinstance(r.get("judge"), dict) and r["judge"].get(key) is not None]
            return f"{statistics.mean(vals):.2f}" if vals else "—"
        rows.append([cat, str(len(sub)), m("groundedness"), m("citation_correctness"),
                     m("relevance"),
                     f"{statistics.mean(float(r['route_correct']) for r in sub):.2f}",
                     f"{statistics.mean(float(r['citations_valid']) for r in sub):.2f}"])

    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    section = f"""{ANSWER_MARKER}
## Answer quality (LLM-as-judge)

Run `{run_id}` · {summary['n']} questions · judge prompt is separate from the generator
and never sees the expected answer.

| overall | value |
|---|---|
| groundedness | {summary['groundedness']:.2f} / 5 |
| citation correctness (judge) | {summary['citation_correctness']:.2f} / 5 |
| relevance | {summary['relevance']:.2f} / 5 |
| route accuracy | {summary['route_accuracy']:.2f} |
| citations valid (checked in code) | {summary['citations_valid_rate']:.2f} |
| answers citing nothing | {summary['uncited_rate']:.2f} |
| refusal accuracy (declined when it should) | {summary['refusal_accuracy']:.2f} |
| sentinel accuracy (declined *in the contract's words*) | {summary['sentinel_accuracy']:.2f} |
| answers with >=1 unsupported claim | {summary['hallucination_rate']:.2f} |

| category | n | ground. | cite (judge) | relev. | route | cites valid |
|---|---|---|---|---|---|---|
{body}

Citation validity is decided by `answer.check_citations`, not by the judge — whether
`[7]` exists when six items were supplied is decidable, and the judge is demonstrably
unreliable on it (it scored citation_correctness 5/5 on an answer carrying zero
citations). The judge is used only for the part that needs reading.
"""
    text = path.read_text() if path.exists() else ""
    if ANSWER_MARKER in text:
        text = text.split(ANSWER_MARKER)[0].rstrip() + "\n\n"
    path.write_text(text.rstrip() + "\n\n" + section)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mode", default="rerank")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    if not available():
        raise SystemExit("no LLM backend reachable — start Ollama")

    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%S")
    print(f"run {run_id} — judging with a separate prompt\n")
    results = evaluate(args.limit, args.mode)
    summary = summarize(results)

    Path(args.out).write_text(json.dumps(
        {"run_id": run_id, "summary": summary, "results": results}, indent=2))

    conn = get_conn()
    try:
        n = write_db(conn, run_id, results)
    finally:
        conn.close()

    print(f"\n--- summary ({summary['n']} questions) ---")
    for k in ("groundedness", "citation_correctness", "relevance"):
        v = summary[k]
        print(f"  {k:22s} {v:.2f} / 5" if v is not None else f"  {k:22s} n/a")
    for k in ("route_accuracy", "citations_valid_rate", "uncited_rate",
              "refusal_accuracy", "sentinel_accuracy", "hallucination_rate"):
        print(f"  {k:22s} {summary[k]:.2f}")
    append_results_md(summary, results, run_id)
    print(f"\nwrote {args.out}, {n} rows to eval_results, and the answer section "
          f"of evals/results.md")


if __name__ == "__main__":
    main()
