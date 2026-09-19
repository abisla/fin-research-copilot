"""Before/after: the answer pipeline with the validation chain off vs on.

Evidence is gathered ONCE per question and shared, so the only thing that differs
between arms is generation + the chain — retrieval noise cannot leak into the delta.
Each question is then generated twice (validate=False, validate=True). Generation is
stochastic (temperature 0.1), so the two arms are independent draws; to separate the
chain's effect from sampling noise the OFF-arm texts are also scored by the chain
after the fact ("paired"), which is deterministic on identical text.

Metrics are all decided in code; the LLM judge is deliberately not used (#30).

    python -m src.evals.validation_eval
"""
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from src.common.db import get_conn
from src.evals.answer_eval import REFUSAL_RE
from src.generation import answer as A
from src.generation.prompts import INSUFFICIENT
from src.retrieval.hybrid import Retriever
from src.router import router as R

ROOT = Path(__file__).parents[2]
QUESTIONS = ROOT / "evals/questions.jsonl"
OUT_JSON = ROOT / "evals/validation_eval.json"
PARTIAL = ROOT / "evals/validation_eval.partial.json"   # checkpoint; deleted on success
RESULTS_MD = ROOT / "evals/results.md"
MARKER = "<!-- validation-eval -->"


def score(ans, evidence) -> dict:
    """Code-decided properties of the answer *as shown to the user*."""
    text = ans.text
    refused = text.upper().startswith(INSUFFICIENT) or bool(REFUSAL_RE.search(text[:400]))
    units = A.claim_units(text)
    fig_uncited = [u for u in units if not u.cites and
                   any(f.checkable for f in A.extract_figures(u.text))]
    bad_figs = A.validate_numeric(text, evidence).items
    return {
        "refused": refused,
        "withheld": bool(ans.rejected_text),
        "withheld_by": next((v.name for v in ans.validation if not v.passed), None),
        "n_citations": len(ans.citations),
        "hallucinated_citations": len(ans.hallucinated_citations),
        # Contract (CLAUDE.md #5): cite, or say insufficient evidence. A refusal complies.
        "compliant": refused or (bool(ans.citations) and not ans.hallucinated_citations
                                 and not fig_uncited),
        "unsupported_figures": bad_figs,
    }


def _gen(*args, **kw):
    """One retry on a transport timeout: `llm.complete` lets TimeoutError escape rather
    than raising LLMUnavailable, and a stalled Ollama call must not void a 40-question
    run. A second failure is recorded by the caller's crash, not swallowed."""
    for attempt in range(3):
        try:
            return A.generate(*args, **kw)
        except (TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(10)


def run() -> list[dict]:
    questions = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    conn = get_conn()
    retr = Retriever(conn=conn)
    out = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else []
    done = {r["id"] for r in out}
    try:
        for q in questions:
            if q["id"] in done:
                continue
            decision = R.classify(q["query"], use_llm=True)
            evidence = A.gather(conn, decision, retr)
            t0 = time.perf_counter()
            off = _gen(q["query"], evidence, decision, validate=False)
            t_off = time.perf_counter() - t0
            on = _gen(q["query"], evidence, decision, validate=True)
            # Paired control: the chain applied to the OFF arm's own text.
            refusal = off.text.upper().startswith(INSUFFICIENT)
            post = A.run_chain(off.text, evidence, refusal=refusal) if evidence else []
            post_fail = next((v.name for v in post if not v.passed), None)
            rec = {"id": q["id"], "category": q["category"], "query": q["query"],
                   "route": decision.route, "n_evidence": len(evidence),
                   "off": score(off, evidence), "on": score(on, evidence),
                   "paired_would_withhold": post_fail,
                   "off_text": off.text[:1500], "on_text": on.text[:1500],
                   "rejected_text": (on.rejected_text or "")[:1500],
                   "rejected_detail": [v.detail for v in on.validation if not v.passed]}
            out.append(rec)
            PARTIAL.write_text(json.dumps(out))
            print(f"  {q['id']:8s} off:{'REF' if rec['off']['refused'] else 'ans'} "
                  f"on:{'WITHHELD ' + rec['on']['withheld_by'] if rec['on']['withheld'] else ('REF' if rec['on']['refused'] else 'ans')}"
                  f"  ({t_off:.0f}s)", flush=True)
    finally:
        retr._owns["conn"] = False
        retr.close()
        conn.close()
    return out


def _rate(rows, pred) -> float:
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else 0.0


def summarize(res: list[dict]) -> dict:
    ie = [r for r in res if r["category"] == "insufficient_evidence"]
    ans = [r for r in res if r["category"] != "insufficient_evidence"]
    s = {"n": len(res)}
    for arm in ("off", "on"):
        s[arm] = {
            "compliance": _rate(res, lambda r: r[arm]["compliant"]),
            "compliance_n": sum(r[arm]["compliant"] for r in res),
            "unsupported_rate": _rate(res, lambda r: bool(r[arm]["unsupported_figures"])),
            "unsupported_n": sum(bool(r[arm]["unsupported_figures"]) for r in res),
            "refusal_rate": _rate(res, lambda r: r[arm]["refused"]),
            "refusal_n": sum(r[arm]["refused"] for r in res),
            "false_refusal_n": sum(r[arm]["refused"] for r in ans),
            "correct_refusal_n": sum(r[arm]["refused"] for r in ie),
            "withheld_n": sum(r[arm]["withheld"] for r in res),
            "withheld_by": dict(Counter(r[arm]["withheld_by"] for r in res if r[arm]["withheld"])),
        }
    s["n_answerable"], s["n_ie"] = len(ans), len(ie)
    s["paired_withhold_n"] = sum(bool(r["paired_would_withhold"]) for r in res)
    s["paired_by"] = dict(Counter(r["paired_would_withhold"] for r in res if r["paired_would_withhold"]))
    s["paired_withhold_by_cat"] = dict(Counter(r["category"] for r in res if r["paired_would_withhold"]))
    return s


def write_md(s: dict, res: list[dict], run_id: str) -> None:
    o, n = s["off"], s["on"]
    N = s["n"]
    sec = f"""{MARKER}
## Validation chain: before / after

Run `{run_id}` · {N} questions ({s['n_answerable']} answerable, {s['n_ie']} insufficient-evidence) ·
retrieval evidence shared between arms, generation run twice (`validate=False` / `validate=True`).

> **Development numbers.** The 80% citation-coverage and 50% grounding-overlap thresholds
> were tuned on this same 40-question set (and the validator's regexes were fixed against
> its failures), so these figures are optimistic about how the chain behaves on unseen
> questions and are not a held-out estimate. Generation is stochastic (temperature 0.1):
> the two arms are independent draws, so small differences are within sampling noise.
> The "paired" row applies the chain to the off-arm's own text and is the noise-free
> read of what the chain rejects.

| metric | validation off | validation on |
|---|---|---|
| citation-contract compliance (answers that cite validly, or decline) | {o['compliance_n']}/{N} ({o['compliance']:.2f}) | {n['compliance_n']}/{N} ({n['compliance']:.2f}) |
| unsupported-figure rate (answers showing a figure not in the evidence) | {o['unsupported_n']}/{N} ({o['unsupported_rate']:.2f}) | {n['unsupported_n']}/{N} ({n['unsupported_rate']:.2f}) |
| refusal rate, all questions | {o['refusal_n']}/{N} ({o['refusal_rate']:.2f}) | {n['refusal_n']}/{N} ({n['refusal_rate']:.2f}) |
| refusals on *answerable* questions ({s['n_answerable']}) | {o['false_refusal_n']} | {n['false_refusal_n']} |
| refusals on insufficient-evidence questions ({s['n_ie']}) | {o['correct_refusal_n']} | {n['correct_refusal_n']} |
| answers withheld by the chain | — | {n['withheld_n']}/{N} {n['withheld_by'] or ''} |
| paired: off-arm answers the chain would withhold | {s['paired_withhold_n']}/{N} {s['paired_by'] or ''} | — |

How to read it:
- **Compliance is close to true by construction** in the on arm: the chain's coverage check
  is most of the definition, so 100% there is the mechanism working, not independent
  evidence. The independent signals are the off-arm number and the withheld count.
- **Unsupported figures** counts answers whose *shown* text has a checkable figure absent
  from every evidence item (`validate_numeric`). A withheld answer shows a refusal, so it
  scores 0 — the price is the withheld count and the extra refusals on answerable questions.
- Refusal = the sentinel or the eval's refusal pattern in the first 400 characters. The
  chain converts a bad answer into a sentinel refusal, so **refusals on answerable
  questions is the cost column**: each is an answer the system had (possibly a good one)
  and did not give.
- Not measured here: whether withheld answers were *right*. The chain checks that text is
  supported by evidence, not that the evidence answers the question (stale-period figures
  pass, see FC-14).
"""
    md = RESULTS_MD.read_text() if RESULTS_MD.exists() else ""
    if MARKER in md:
        md = md.split(MARKER)[0].rstrip() + "\n\n"
    else:
        md = md.rstrip() + "\n\n"
    RESULTS_MD.write_text(md + sec)


def main() -> None:
    from datetime import datetime, timezone
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    res = run()
    PARTIAL.unlink(missing_ok=True)
    s = summarize(res)
    OUT_JSON.write_text(json.dumps({"run_id": run_id, "summary": s, "results": res}, indent=1))
    write_md(s, res, run_id)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
