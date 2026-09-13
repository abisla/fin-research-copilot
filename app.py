"""Phase 8 demo UI: one question box, and everything the system did to answer it.

    streamlit run app.py

The point of this page is not that it answers questions — `scripts/ask.py` already
does. It is that every stage is visible at once: which store the router picked and
why, which chunks came back with what metadata and what score, which of them the
answer actually cited, and which citations were stripped as invented. A demo that
shows only the final answer is indistinguishable from one that makes it up.

Three things are deliberate here:

* **Heavy handles are cached, the DB connection is not.** The Qdrant client, the BM25
  index and the encoders cost seconds to load and are shared across reruns via
  `st.cache_resource`; a connection is cheap and a stale one in a failed transaction
  is a support ticket, so each run opens and closes its own.
* **Sidebar filters are overrides, not suggestions.** They are pushed into retrieval
  as hard predicates (`Filters`), the same object the eval harness uses — a filter a
  human set outranks one the router inferred.
* **The retrieval-mode toggle is the interview feature.** `dense | bm25 | hybrid |
  rerank` are the four arms Phase 7 measured, and the A/B tab runs all four on the
  same query so the tradeoff can be shown live rather than asserted.
"""
import time
from datetime import date, timedelta

import streamlit as st

from src.common.config import CFG
from src.common.db import get_conn, get_qdrant, storage_mode
from src.common.models import Filters
from src.generation import answer as A
from src.generation.llm import available, backend
from src.retrieval.hybrid import MODES, Retriever
from src.router.router import recent_decisions, route_and_log

st.set_page_config(page_title="Financial Research Copilot", page_icon="📊", layout="wide")

# (button label, question) — the five routes, one example each, so a demo can walk
# the router through every branch without typing.
EXAMPLES = [
    ("MIXED", "How has NVDA revenue changed over the last four quarters and why"),
    ("NEWS", "Give me all important NVDA news from last week"),
    ("FILING_RAG", "What risk factors does MSFT disclose about competition"),
    ("CROSS_SOURCE", "Compare NVDA and MSFT revenue growth"),
    ("NUMERIC", "What was NVDA's most recent quarterly revenue"),
]

MODE_HELP = {
    "dense": "bge-small query vector → Qdrant HNSW. The baseline the others must beat.",
    "bm25": "BM25Okapi over the same chunks. Wins on rare tokens: tickers, 'Blackwell', 'CET1'.",
    "hybrid": "RRF (k=60) over dense top-30 + BM25 top-30, with each arm's rank-1 anchored.",
    "rerank": "The fused pool re-scored by ms-marco-MiniLM cross-encoder. Default.",
}
KIND_ICON = {"filing": "📄", "news": "📰", "financial": "🔢"}


# --- cached handles ----------------------------------------------------------

@st.cache_resource(show_spinner="Loading Qdrant client and BM25 index…")
def _handles():
    """Query-time handles that are expensive once and free thereafter.

    The DB connection is deliberately *not* cached — see the module docstring.
    """
    from src.retrieval import sparse
    return get_qdrant(), sparse.load_index()


def _retriever(conn) -> Retriever:
    """A Retriever over the cached handles. Owns nothing, so close() is a no-op."""
    client, bm25 = _handles()
    return Retriever(conn=conn, client=client, bm25=bm25)


@st.cache_data(ttl=60, show_spinner=False)
def _corpus_stats() -> dict:
    """Row counts, so an empty store is visible as an empty store rather than as a
    system that answers everything with 'insufficient evidence'."""
    stats = {}
    conn = get_conn()
    try:
        cur = conn.cursor()
        for table in ("documents", "chunks", "financials", "news_articles"):
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cur.fetchone()[0]
            except Exception:
                conn.rollback()
                stats[table] = None
    finally:
        conn.close()
    return stats


@st.cache_data(ttl=300, show_spinner=False)
def _doc_types() -> list[str]:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT doc_type FROM chunks ORDER BY doc_type")
        return [r[0] for r in cur.fetchall() if r[0]]
    except Exception:
        return ["10-K", "10-Q", "8-K"]
    finally:
        conn.close()


# --- rendering ---------------------------------------------------------------

def render_route(decision, elapsed_ms: int) -> None:
    """The routing decision, in the terms the router actually decided it in."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", decision.route)
    c2.metric("Decided by", decision.method)
    c3.metric("Route latency", f"{decision.latency_ms} ms")
    c4.metric("End to end", f"{elapsed_ms/1000:.1f} s")

    st.caption("A route names **which store answers the question**, not what it is about.")
    why = " · ".join(decision.reasons) or "no rule matched"
    st.markdown(f"**Signals:** {why}")
    if decision.tickers:
        st.markdown(f"**Tickers detected:** {', '.join(decision.tickers)}")
    if decision.query_plan:
        name, params = decision.query_plan
        args = ", ".join(f"{k}={v}" for k, v in params.items())
        st.markdown(f"**Canned SQL plan:** `{name}({args})` — the LLM never writes SQL.")


def render_evidence(evidence, cited: set[str], show_text: bool) -> None:
    """Every numbered context item, marked with whether the answer used it.

    The numbering here is the same numbering the model saw and the same one
    `check_citations` validated, so a reader can check [3] by eye.
    """
    if not evidence:
        st.info("No evidence was gathered — the answer below should say so.")
        return
    for i, e in enumerate(evidence, start=1):
        used = "✅ cited" if e.key in cited else "· not cited"
        score = f" · score {e.score:.3f} ({e.retriever})" if e.score is not None else ""
        header = f"{KIND_ICON.get(e.kind, '•')} **[{i}]** {e.label}{score} — {used}"
        with st.expander(header, expanded=show_text and i <= 3):
            st.text(e.text if show_text else e.text[:400] + ("…" if len(e.text) > 400 else ""))
            if e.url:
                st.markdown(f"[source]({e.url})")
            st.caption(f"key: `{e.key}`")


def render_answer(ans) -> None:
    """The answer, plus the two contract breaches worth shouting about."""
    if ans.insufficient_evidence:
        st.warning(ans.text)
    else:
        st.markdown(ans.text)

    if ans.hallucinated_citations:
        st.error(
            f"Stripped hallucinated citation(s) {ans.hallucinated_citations} — the model "
            f"cited beyond the {len(ans.evidence)} items supplied. The number is removed "
            f"from the prose because a dangling `[7]` still reads as sourced.")
    if ans.uncited:
        st.warning(
            "This answer cites nothing and did not declare insufficient evidence — a "
            "breach of the citation contract. Measured at 0.20 of answers across the "
            "40-question eval and left unfixed rather than tuned by feel — Phase 6 showed "
            "that tuning this prompt for coverage made it strictly worse. See "
            "evals/results.md.")


def run_ab(retriever, query: str, filters: Filters, top_k: int) -> None:
    """The same query through all four arms, side by side.

    Shared candidate widths and one shared `Filters`, so what differs between the
    columns is the retriever and nothing else — the same property that makes the
    Phase 7 A/B a comparison of retrievers rather than of plumbing.
    """
    cols = st.columns(len(MODES))
    for col, mode in zip(cols, MODES):
        with col:
            st.markdown(f"**{mode}**")
            started = time.perf_counter()
            try:
                hits = retriever.search(query, mode=mode, top_k=top_k, filters=filters)
            except Exception as exc:                     # one dead arm must not blank the page
                st.error(f"{type(exc).__name__}: {exc}")
                continue
            st.caption(f"{int((time.perf_counter() - started) * 1000)} ms")
            for rank, h in enumerate(hits, start=1):
                m = h.meta
                st.markdown(
                    f"`{rank}.` **{m.ticker} {m.doc_type}** {m.fiscal_period or ''}  \n"
                    f"<span style='font-size:0.8em;color:#888'>{(m.section or '—')[:40]} · "
                    f"score {h.score:.3f} · #{m.chunk_id.rsplit('::', 1)[-1]}</span>",
                    unsafe_allow_html=True)


# --- sidebar -----------------------------------------------------------------

def sidebar() -> dict:
    st.sidebar.title("📊 Research Copilot")
    st.sidebar.caption("Hybrid RAG over SEC filings, XBRL financials and ticker news.")

    stats = _corpus_stats()
    llm_ok = available()
    st.sidebar.markdown(
        f"**Storage:** `{storage_mode()}` · **LLM:** `{backend()}` "
        f"{'🟢' if llm_ok else '🔴'}")
    if not llm_ok:
        st.sidebar.warning(
            f"No `{backend()}` backend reachable. Retrieval and routing still work; "
            f"generation will decline rather than guess.")
    st.sidebar.caption(
        " · ".join(f"{k.split('_')[0]} {v if v is not None else '?'}"
                   for k, v in stats.items()))

    st.sidebar.subheader("Filters")
    st.sidebar.caption("Hard predicates pushed into retrieval before similarity — "
                       "an explicit filter outranks one the router inferred.")
    tickers = st.sidebar.multiselect("Ticker", list(CFG["tickers"]), default=[],
                                    key="f_tickers")
    doc_types = st.sidebar.multiselect("Document type", _doc_types(), default=[],
                                      key="f_doc_types")

    use_dates = st.sidebar.checkbox("Constrain filing date", value=False, key="f_dates")
    date_from = date_to = None
    if use_dates:
        default_from = date.today() - timedelta(days=365)
        date_from = st.sidebar.date_input("From", value=default_from, key="f_from")
        date_to = st.sidebar.date_input("To", value=date.today(), key="f_to")

    st.sidebar.subheader("Retrieval")
    mode = st.sidebar.radio("Mode", MODES, index=MODES.index("rerank"), key="mode",
                            help="Phase 7 measured all four; see README for the table.")
    st.sidebar.caption(MODE_HELP[mode])
    top_k = st.sidebar.slider("Chunks in context", 3, 15,
                              CFG["retrieval"]["final_top_k"], key="top_k")

    st.sidebar.subheader("Behaviour")
    use_llm_router = st.sidebar.checkbox(
        "LLM routing fallback", value=True, key="use_llm_router",
        help="Consulted only when no rule matches. Measured net-negative on the eval "
             "set (FC-13) — uncheck to route rules-only.")
    show_text = st.sidebar.checkbox("Show full chunk text", value=False, key="show_text")

    with st.sidebar.expander("Recent routing decisions"):
        conn = get_conn()
        try:
            rows = recent_decisions(conn, limit=10)
        except Exception:
            rows = []
        finally:
            conn.close()
        if rows:
            st.dataframe(rows, hide_index=True)
        else:
            st.caption("Nothing logged yet.")

    return {
        "filters": Filters(
            tickers=tickers or None, doc_types=doc_types or None,
            date_from=date_from, date_to=date_to),
        "mode": mode, "top_k": top_k,
        "use_llm_router": use_llm_router, "show_text": show_text,
    }


# --- main --------------------------------------------------------------------

def run_query(conn, retriever, query: str, opts: dict) -> dict:
    """Route -> gather -> generate, timed. Returns everything the tabs render.

    Kept separate from the rendering so the result can live in `st.session_state`:
    Streamlit re-executes the whole script on every widget interaction, and
    re-answering the question because someone ticked "show full chunk text" would
    burn an LLM call per click.
    """
    started = time.perf_counter()
    decision = route_and_log(conn, query, use_llm=opts["use_llm_router"])
    route_ms = int((time.perf_counter() - started) * 1000)

    # The sidebar's filters are merged with — not replaced by — the router's ticker
    # inference: an empty ticker box means "whatever the question named", not "none".
    sb = opts["filters"]
    overrides = None
    if not sb.is_empty():
        overrides = Filters(
            tickers=sb.tickers or decision.tickers or None,
            doc_types=sb.doc_types, date_from=sb.date_from, date_to=sb.date_to)

    evidence = A.gather(conn, decision, retriever, mode=opts["mode"],
                        overrides=overrides, top_k=opts["top_k"])
    ans = A.generate(query, evidence, decision)
    ans.decision = decision
    return {"query": query, "decision": decision, "evidence": evidence, "answer": ans,
            "route_ms": route_ms, "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "overrides": overrides}


def run_key(query: str, opts: dict) -> tuple:
    """What must change before the question is worth re-answering."""
    return (query, opts["mode"], opts["top_k"], opts["use_llm_router"],
            opts["filters"].describe())


def main():
    opts = sidebar()
    st.title("Financial Research Copilot")
    st.caption("Every claim cites the chunk it came from, or the answer says "
               "insufficient evidence. Citations are verified in code, not trusted.")

    if "question" not in st.session_state:
        st.session_state.question = EXAMPLES[0][1]

    st.caption("One example per route:")
    for col, (label, example) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        if col.button(label, key=f"ex_{label}", help=example):
            st.session_state.question = example

    query = st.text_input("Ask a question", key="question").strip()
    asked = st.button("Ask", type="primary")

    tab_answer, tab_ab, tab_brief = st.tabs(["Answer", "Retrieval A/B", "Weekly brief"])
    conn = get_conn()
    try:
        retriever = _retriever(conn)

        with tab_answer:
            if not query:
                st.info("Ask a question, or pick one of the route examples above.")
            else:
                key = run_key(query, opts)
                if asked or st.session_state.get("run_key") != key:
                    with st.spinner(f"Routing, retrieving ({opts['mode']}), generating…"):
                        st.session_state.result = run_query(conn, retriever, query, opts)
                    st.session_state.run_key = key
                result = st.session_state.result
                ans, evidence = result["answer"], result["evidence"]

                render_route(result["decision"], result["elapsed_ms"])
                st.subheader("Answer")
                render_answer(ans)
                st.caption(
                    f"{result['elapsed_ms']/1000:.1f}s end to end · mode `{opts['mode']}` · "
                    f"filters: {(result['overrides'] or result['decision'].filters).describe()}")

                st.subheader(f"Evidence ({len(evidence)} items)")
                st.caption("The numbering below is the numbering the model saw and the "
                           "numbering `check_citations` verified — [3] here is [3] there.")
                render_evidence(evidence, set(ans.citations), opts["show_text"])

        with tab_ab:
            st.caption(
                "The same query through all four arms, one shared filter and the same "
                "candidate widths — so what differs is the retriever, not the plumbing. "
                "Phase 7 measured these arms on 25 graded questions; see the README table.")
            if not query:
                st.info("Ask a question first.")
            elif st.toggle("Run all four modes", key="ab_on"):
                run_ab(retriever, query, opts["filters"], opts["top_k"])

        with tab_brief:
            st.caption(
                "Clustered, deduped and ranked by source diversity — n articles from one "
                "publisher can never outrank n+1 sources (FC-4). This, not the Answer tab, "
                "is the right tool for 'give me all important news'.")
            tickers = list(CFG["tickers"])
            default = tickers.index(opts["filters"].tickers[0]) if opts["filters"].tickers else 0
            ticker = st.selectbox("Ticker", tickers, index=default, key="brief_ticker")
            days = st.slider("Lookback days", 1, 30, CFG["news"]["lookback_days"],
                             key="brief_days")
            if st.toggle("Build brief", key="brief_on"):
                from src.news.brief import weekly_brief
                with st.spinner("Clustering and summarizing…"):
                    st.markdown(weekly_brief(conn, ticker, days, 5, available()))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
