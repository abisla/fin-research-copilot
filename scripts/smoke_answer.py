"""Phase 6 gate: routing contract, evidence assembly, citation verification.

Routing is asserted on hand-written queries rather than on whatever the corpus happens
to contain, because the route is a *contract*: "revenue and why" must reach MIXED
whether or not NVDA filed this week. `classify()` is pure, so all of that runs without
a database.

The citation post-check is asserted on synthetic model output, not on generated text —
the property is "a [7] against 6 items is caught and removed", and waiting for the LLM
to actually emit one would make the gate flaky in the direction that hides bugs.

Then the real stores: every route is run end-to-end against Postgres, and the
Definition-of-Done query must produce evidence from both SQL and the filing index.

    python scripts/smoke_answer.py
    python scripts/smoke_answer.py --no-llm      # contract only, no generation
"""
import argparse
from datetime import datetime, timezone

from src.common.db import get_conn
from src.generation import answer as A
from src.generation.llm import available, backend
from src.generation.prompts import INSUFFICIENT
from src.router import router as R

NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)

# (query, expected route) — the contract the rules must hold without an LLM.
ROUTING_CASES = [
    ("What was NVDA revenue last quarter",                              "NUMERIC"),
    ("How much did Microsoft earn per share in fiscal 2027",            "NUMERIC"),
    ("How has NVDA revenue changed over the last four quarters and why", "MIXED"),
    ("Why did JPMorgan revenue decline",                                "MIXED"),
    ("What risk factors does MSFT disclose in its 10-K",                "FILING_RAG"),
    ("What does NVIDIA say about competition in its filings",           "FILING_RAG"),
    ("Give me all important NVDA news from last week",                  "NEWS"),
    ("What happened to JPM in the past 3 days",                         "NEWS"),
    ("Compare NVDA and MSFT revenue growth",                            "CROSS_SOURCE"),
    ("Does the recent NVDA news match what they said in the 10-K",      "CROSS_SOURCE"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-llm", action="store_true", help="skip generation")
    args = ap.parse_args()

    # --- routing contract: pure, no DB, no LLM ---------------------------
    for query, expected in ROUTING_CASES:
        d = R.classify(query, now=NOW, use_llm=False)
        assert d.route == expected, f"{query!r} -> {d.route}, expected {expected} ({d.reasons})"
        assert d.method == "rule", f"{query!r} needed a fallback: {d.method}"
    print(f"routing ok   - {len(ROUTING_CASES)}/{len(ROUTING_CASES)} cases by rule, no LLM")

    # --- ticker + window extraction --------------------------------------
    assert R.detect_tickers("compare NVDA and Microsoft") == ["NVDA", "MSFT"], "config order"
    assert R.detect_tickers("JP Morgan raised its dividend") == ["JPM"], "alias match"
    assert R.detect_tickers("what about AI stocks") == [], "must not match substrings"
    since, _phrase = R.detect_window("news from the last 3 days", now=NOW)
    assert since == datetime(2026, 9, 9, tzinfo=timezone.utc).date(), since
    assert R.detect_window("what is the 10-K risk section", now=NOW) == (None, None)
    print(f"extract ok   - tickers in config order; 'last 3 days' -> since {since}")

    # --- date bound is applied to news, never to filings ------------------
    news = R.classify("NVDA news last week", now=NOW, use_llm=False)
    filing = R.classify("NVDA risk factors in its 10-K", now=NOW, use_llm=False)
    assert news.filters.date_from is not None, "NEWS must carry a hard date bound"
    assert filing.filters.date_from is None, "a 10-K is not stale at 8 days old"
    print(f"filters ok   - NEWS since {news.filters.date_from}; FILING_RAG unbounded")

    # --- the router picks a canned query, never writes SQL ----------------
    from src.structured.queries import QUERIES
    plan = R.classify("How has NVDA revenue changed over the last four quarters and why",
                      now=NOW, use_llm=False).query_plan
    assert plan and plan[0] in QUERIES, f"plan {plan} is not a canned query"
    assert plan == ("metric_series", {"ticker": "NVDA", "metric": "revenue",
                                      "n_quarters": 4}), plan
    margin = R.classify("show me MSFT gross margin trend", now=NOW, use_llm=False).query_plan
    assert margin[0] == "margin_trend", margin
    yoy = R.classify("NVDA revenue growth year over year", now=NOW, use_llm=False).query_plan
    assert yoy[0] == "yoy_growth", yoy
    print(f"plan ok      - {plan[0]}/{margin[0]}/{yoy[0]} selected from {len(QUERIES)} canned queries")

    # --- citation post-check: synthetic output, exact properties ----------
    clean, cited, halluc = A.check_citations("Revenue rose [1] and margin fell [3].", 3)
    assert cited == [1, 3] and halluc == [], (cited, halluc)
    clean, cited, halluc = A.check_citations("Rose [1] sharply [7] per the filing [2].", 3)
    assert cited == [1, 2] and halluc == [7], (cited, halluc)
    assert "[7]" not in clean, f"hallucinated citation left in the prose: {clean!r}"
    assert "[1]" in clean and "[2]" in clean, "valid citations must survive"
    clean, cited, halluc = A.check_citations("No citations at all.", 3)
    assert cited == [] and halluc == []
    print(f"citations ok - [7]/3 stripped and flagged; valid ones preserved -> {clean!r}")

    # --- context numbering matches what the checker validates -------------
    ev = [A.Evidence(key=f"k{i}", kind="filing", label=f"L{i}", text=f"T{i}")
          for i in range(3)]
    ctx = A.build_context(ev)
    assert ctx.startswith("[1] (L0)") and "[3] (L2)" in ctx and "[4]" not in ctx
    print(f"context ok   - {len(ev)} items numbered [1]..[{len(ev)}]")

    conn = get_conn()
    try:
        # --- every route assembles real evidence -------------------------
        print()
        for query, expected in ROUTING_CASES:
            d = R.route_and_log(conn, query, now=NOW, use_llm=False)
            retr = None
            if d.route in ("FILING_RAG", "MIXED", "CROSS_SOURCE"):
                from src.retrieval.hybrid import Retriever
                retr = Retriever(conn=conn)
            ev = A.gather(conn, d, retr)
            kinds = sorted({e.kind for e in ev})
            assert ev, f"{query!r} ({d.route}) assembled no evidence"
            assert len({e.key for e in ev}) == len(ev), f"{query!r} duplicate evidence keys"
            print(f"  {d.route:13s} {len(ev):2d} items {kinds}  {query[:44]}")
            if retr:
                retr._owns["conn"] = False
                retr.close()

        # --- CROSS_SOURCE consults news only when asked ------------------
        from src.retrieval.hybrid import Retriever as _R
        cmp_d = R.classify("Compare NVDA and MSFT revenue growth", now=NOW, use_llm=False)
        rec_d = R.classify("Does the recent NVDA news match what they said in the 10-K",
                           now=NOW, use_llm=False)
        assert not cmp_d.news_signal and rec_d.news_signal, "news signal misread"
        r1 = _R(conn=conn)
        cmp_kinds = {e.kind for e in A.gather(conn, cmp_d, r1)}
        rec_kinds = {e.kind for e in A.gather(conn, rec_d, r1)}
        r1._owns["conn"] = False
        r1.close()
        assert "news" not in cmp_kinds, f"numeric comparison pulled news: {cmp_kinds}"
        assert "news" in rec_kinds, f"reconciliation needs news: {rec_kinds}"
        print(f"\nscope ok     - compare={sorted(cmp_kinds)}; reconcile={sorted(rec_kinds)}")

        # --- MIXED must genuinely span both stores -----------------------
        dod = "How has NVDA revenue changed over the last four quarters and why"
        d = R.route_and_log(conn, dod, now=NOW, use_llm=False)
        from src.retrieval.hybrid import Retriever
        retr = Retriever(conn=conn)
        ev = A.gather(conn, d, retr)
        retr._owns["conn"] = False
        retr.close()
        kinds = {e.kind for e in ev}
        assert {"financial", "filing"} <= kinds, f"Definition-of-Done needs both, got {kinds}"
        print(f"\nmixed ok     - Definition-of-Done query spans {sorted(kinds)}")

        # --- insufficient evidence: no context, no LLM call --------------
        empty = R.classify("What was TSLA revenue", now=NOW, use_llm=False)
        ans = A.generate("What was TSLA revenue", [], empty)
        assert ans.insufficient_evidence and ans.text.startswith(INSUFFICIENT), ans.text
        assert not ans.citations and not ans.uncited
        print(f"empty ok     - no evidence -> {ans.text[:60]!r}")

        # --- routing_log records every decision --------------------------
        logged = R.recent_decisions(conn, limit=5)
        assert logged and all(r["route"] in R.ROUTES for r in logged), logged
        assert all(r["method"] for r in logged)
        print(f"log ok       - routing_log holds {len(logged)} recent decision(s); "
              f"newest {logged[0]['route']} via {logged[0]['method']}")

        # --- LLM fallback: only vague queries reach it -------------------
        if not (args.no_llm or not available()):
            # A bare label list made llama3.1:8b answer NEWS to all of these
            # (position bias); the few-shot prompt is what fixed it. See DECISIONS #26.
            for query, expected in [("Tell me about Microsoft", "FILING_RAG"),
                                    ("NVDA Blackwell", "FILING_RAG"),
                                    ("Is JPM a good investment", "FILING_RAG"),
                                    ("Anything new with Microsoft lately", "NEWS")]:
                d = R.classify(query, now=NOW, use_llm=True)
                assert d.method in ("llm", "rule"), d.method
                assert d.route == expected, f"{query!r} -> {d.route} via {d.method}"
            assert R.classify("Tell me about Microsoft", now=NOW,
                              use_llm=True).method == "llm", "should not be rule-routed"
            print("\nfallback ok  - 4/4 vague queries classified, none defaulted to NEWS")

        # --- generation, when a backend is up ----------------------------
        if args.no_llm or not available():
            print(f"\nllm skipped  - backend {backend()!r} "
                  f"{'disabled by flag' if args.no_llm else 'unreachable'}")
        else:
            retr = Retriever(conn=conn)
            ans = A.ask(dod, conn=conn, retriever=retr)
            retr._owns["conn"] = False
            retr.close()
            assert ans.route == "MIXED", ans.route
            assert not ans.hallucinated_citations, \
                f"cited beyond context: {ans.hallucinated_citations}"
            assert not ans.uncited, "answer cited nothing and claimed no insufficiency"
            assert all(k in {e.key for e in ans.evidence} for k in ans.citations)
            assert "Facts" in ans.text and "Interpretation" in ans.text, \
                "a research question must be split into Facts / Interpretation"
            print(f"\nanswer ok    - {len(ans.citations)} citation(s) resolved to evidence "
                  f"keys, 0 hallucinated, Facts/Interpretation present")
    finally:
        conn.close()

    print("\nPHASE 6 ROUTER + GENERATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
