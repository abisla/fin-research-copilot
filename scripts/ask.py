"""Phase 6 CLI: ask a question end-to-end (route -> retrieve -> generate -> verify).

    python scripts/ask.py "How has NVDA revenue changed over the last four quarters and why"
    python scripts/ask.py "What risk factors does MSFT disclose" --show-context
    python scripts/ask.py "Give me JPM news from last week" --mode hybrid

Prints the routing decision and the evidence list alongside the answer, because the
point of the demo is that every claim can be traced back to the chunk it came from.
"""
import argparse

from src.common.db import get_conn
from src.generation.answer import ask
from src.retrieval.hybrid import MODES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="+")
    ap.add_argument("--mode", default="rerank", choices=MODES,
                    help="retrieval strategy for filing evidence (default rerank)")
    ap.add_argument("--no-llm-route", action="store_true",
                    help="rules only; never fall back to the LLM classifier")
    ap.add_argument("--show-context", action="store_true",
                    help="print the full text of every evidence item")
    args = ap.parse_args()
    question = " ".join(args.question)

    conn = get_conn()
    try:
        answer = ask(question, conn=conn, mode=args.mode, use_llm=not args.no_llm_route)
    finally:
        conn.close()

    print(f"\nQ: {question}")
    print(f"route: {answer.decision.describe()}")
    if answer.decision.query_plan:
        name, params = answer.decision.query_plan
        print(f"plan:  {name}({', '.join(f'{k}={v}' for k, v in params.items())})")

    print(f"\nevidence ({len(answer.evidence)} items):")
    for i, e in enumerate(answer.evidence, start=1):
        mark = "*" if e.key in answer.citations else " "
        print(f" {mark}[{i}] {e.kind:9s} {e.label}")
        if args.show_context:
            print(f"      {e.text[:400]}...")
    print("  (* = cited in the answer)")

    print(f"\n{answer.text}\n")
    if answer.hallucinated_citations:
        print(f"WARNING: stripped hallucinated citation(s) {answer.hallucinated_citations} "
              f"— cited beyond the {len(answer.evidence)} items supplied")
    if answer.uncited:
        print("WARNING: answer cites nothing and did not declare insufficient evidence")


if __name__ == "__main__":
    main()
