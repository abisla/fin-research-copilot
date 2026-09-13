"""Gate for Phase 8: the Streamlit app renders, routes, retrieves and cites — headless.

    python scripts/smoke_app.py

Uses `streamlit.testing.v1.AppTest`, which executes `app.py` exactly as a browser
session would, minus the browser. That matters more than it sounds: every other phase
gate here asserts on library functions, and a demo UI is the one component whose
failure modes (a widget key collision, a stale session_state, a handle opened per
rerun) only appear when the script is actually run as a Streamlit script.

What is asserted is the contract the demo exists to show, not the styling: the route
is displayed, evidence is numbered, the numbering the UI shows is the numbering the
citation check validated, and no unhandled exception reaches the page.
"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app.py")   # resolved against this file, not cwd
TIMEOUT = 600           # a cold run loads bge-small, the cross-encoder and the LLM


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"{label:12s} {'ok' if ok else 'FAIL'}   - {detail}")
    return ok


def run(question: str | None = None, **widgets) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    if question is not None:
        at.text_input(key="question").set_value(question)
    for key, value in widgets.items():
        at.session_state[key] = value
    if question is not None or widgets:
        at.run()
    return at


def main() -> int:
    passed = True

    # 1. Cold render: no exception, the question box exists, one example per route.
    at = run()
    passed &= check("render", not at.exception,
                    f"{len(at.tabs)} tabs, {len(at.button)} buttons, "
                    f"no exception on cold start")
    if at.exception:
        for e in at.exception:
            print(e.stack_trace or e.message)
        return 1

    routes = {b.label for b in at.button}
    passed &= check("examples", {"NUMERIC", "NEWS", "FILING_RAG", "MIXED",
                                 "CROSS_SOURCE"} <= routes,
                    "one seeded example per route")

    # 2. The default (MIXED) question routes and renders its decision.
    metrics = {m.label: m.value for m in at.metric}
    passed &= check("route", metrics.get("Route") == "MIXED",
                    f"{metrics.get('Route')} via {metrics.get('Decided by')} "
                    f"in {metrics.get('End to end')}")

    # 3. Evidence is rendered, numbered, and marked cited / not cited — the whole
    #    point of the page. Expander labels carry [n], the score and the mark.
    labels = [e.label for e in at.expander]
    numbered = [lab for lab in labels if lab.lstrip("📄📰🔢• ").startswith("**[")]
    passed &= check("evidence", len(numbered) >= 2,
                    f"{len(numbered)} numbered items, e.g. "
                    f"{numbered[0][:70] if numbered else '—'}")
    passed &= check("cited", any("✅ cited" in lab for lab in numbered),
                    "at least one evidence item is marked as cited in the answer")
    passed &= check("scores", any("score" in lab for lab in numbered),
                    "retrieved chunks show their retriever score")

    # 4. A NEWS question reaches the news store, not the filing index.
    at = run("Give me all important NVDA news from last week")
    metrics = {m.label: m.value for m in at.metric}
    news_items = [e.label for e in at.expander if "📰" in e.label]
    passed &= check("news", metrics.get("Route") == "NEWS" and bool(news_items),
                    f"route {metrics.get('Route')}, {len(news_items)} article(s) "
                    f"as evidence")

    # 5. A sidebar filter is an override, not a suggestion: the question names NVDA,
    #    the sidebar says MSFT, and MSFT must win — the filter is a hard predicate
    #    pushed into retrieval, not a hint added to the prompt.
    at = run("What risk factors does NVDA disclose about competition",
             f_tickers=["MSFT"])
    filing = [e.label for e in at.expander if "📄" in e.label]
    passed &= check("filters", bool(filing) and all("MSFT" in lab for lab in filing),
                    f"question said NVDA, sidebar said MSFT -> {len(filing)} filing "
                    f"chunk(s), all MSFT")

    print("\n" + ("PHASE 8 DEMO APP PASSED" if passed else "PHASE 8 DEMO APP FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
