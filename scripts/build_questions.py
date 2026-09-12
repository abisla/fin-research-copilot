"""Build evals/questions.jsonl — attach expected_chunk_ids to the draft questions.

**Labelling method, and why it is written as rules rather than a hand-picked list.**
Candidates were pooled from all four retrieval arms (`scripts/pool_candidates.py`) and
read. But converting those judgements into a literal list of chunk_ids produces a
ground truth nobody can check or reproduce, and this corpus makes that worse: the same
risk factor is repeated near-verbatim across a 10-K and three 10-Qs, so a hand-picked
list silently decides which *copies* count.

So each question carries an explicit relevance RULE — ticker, optional document, and
the concept that must be present — applied over the whole `chunks` table. Two
consequences worth stating plainly:

* It finds relevant chunks that **no retriever returned**, which pooling alone cannot.
  That matters here: 18 chunks mention Hopper and the pool surfaced almost none.
* It is auditable. Anyone can re-run this and get the same ground truth, and can
  disagree with a rule rather than with an opinion.

**The bias this introduces, stated rather than hidden.** Rules are lexical, so they
favour BM25 on questions whose wording overlaps the rule. Mitigation: the `semantic`
questions are deliberately paraphrased away from the vocabulary their rule matches
("depending on a small number of large customers" -> `limited number of customers`),
so lexical overlap with the *question* stays low even though the rule is lexical. The
`exact_keyword` set is the opposite on purpose — there, BM25 *should* win, and the
measurement is whether it does.

    python scripts/build_questions.py --out evals/questions.jsonl
"""
import argparse
import json
import sys

from src.common.db import get_conn
from src.structured import queries as Q

sys.path.insert(0, "evals")
from questions_draft import QUESTIONS  # noqa: E402

# Accessions of the newest filing per ticker/form — the temporal questions are graded
# on whether retrieval prefers these over older copies of similar text.
NEWEST = {
    "NVDA_10Q": "0001045810-26-000075",     # FY2026Q3, filed 2026-08-26
    "MSFT_10K": "0001193125-26-323660",     # FY2026,   filed 2026-07-29
    "JPM_10Q":  "0001628280-26-054343",     # FY2026Q2, filed 2026-08-06
}

# The current annual report per ticker. Unless a rule names a document, relevance is
# scoped to these.
#
# Why: this corpus repeats the same disclosure near-verbatim across a 10-K and three or
# four 10-Qs, so "every chunk mentioning the concept" produced relevant sets of 43-413
# chunks and a Recall@5 whose ceiling was 1%. Scoping to the current annual report both
# fixes the arithmetic and encodes the behaviour actually wanted: asked what a company
# says about a topic, cite its current disclosure, not the same words from two years ago.
# The cost is real and is reported in results.md — a correct passage retrieved from last
# year's 10-Q scores as a miss.
CURRENT_10K = {
    "NVDA": "0001045810-26-000021",         # FY2026, filed 2026-02-25
    "MSFT": "0001193125-26-323660",         # FY2026, filed 2026-07-29
    "JPM":  "0001628280-26-008131",         # FY2025, filed 2026-02-13
}

# A chunk that names a concept once in passing is not "about" it. Two occurrences is a
# crude topicality test, but it is stated and reproducible, and it removes the
# reconciliation tables and cross-reference lines that a single mention lets through.
MIN_TERM_HITS = 2

# ...except for a distinctive multi-word phrase. "limited number of customers" appears
# exactly once in the chunk that is unambiguously the answer, so a frequency test alone
# would throw away the best passage in the corpus. A phrase this specific is topical on
# one occurrence in a way that a bare word like "Azure" is not.
SPECIFIC_PHRASE_CHARS = 18

# Per-question overrides. "Hopper" is a genuinely rare token — 18 chunks corpus-wide,
# almost all single-mention, and none in the current 10-K because Blackwell superseded
# it. Scoping it like the others would have made the rule match nothing. It stays in the
# set precisely because a rare token is what sparse retrieval is supposed to win on.
MIN_HITS_OVERRIDE = {
    "kw-03": 1,
    # Density thresholds for the cross-document set: a passage that says "revenue" once
    # is not the revenue discussion. Tuned to land each relevant set in single digits.
    "x-01": 8, "x-02": 4, "x-03": 10, "x-04": 8, "x-06": 4, "temp-06": 3,
}
SCOPE_ALL = "ALL"          # accession sentinel: search the whole corpus, not one filing

# Boilerplate that retrievers surface constantly and that answers nothing: press
# release footers, section headers, forward-looking-statement disclaimers.
BOILERPLATE = [
    "About NVIDIA NVIDIA (NASDAQ",
    "For further information, contact",
    "Item 1A. Risk Factors",
    "creates platforms and tools powered by AI to deliver innovative",
    "We conducted our reviews in accordance with standards of the PCAOB",
]

# rule = (tickers, any_terms, all_terms, accession_prefix or None, sections or None)
# accession None means "the current 10-K for each ticker" (see CURRENT_10K).
RULES = {
    "sem-01": (["NVDA"], ["export control", "export licen"], [], None, None),
    "sem-02": (["MSFT"], ["highly competitive", "business model competition",
                          "compete with us"], [], None, None),
    "sem-03": (["NVDA"], ["limited number of customers", "concentration of revenue",
                          "significant portion of our revenue"], [], None, None),
    "sem-04": (["JPM"], ["regulatory capital requirement", "capital conservation buffer",
                         "minimum capital ratios", "required to hold"], [], None, None),
    "sem-05": (["NVDA"], ["depend on foundries", "third-party suppliers", "subcontractor",
                          "manufacturing capacity", "supply chain"], [], None, None),
    "sem-06": (["MSFT"], ["partnership with OpenAI", "agreement with OpenAI"], [], None, None),
    "sem-07": (["JPM"], ["stress test", "severely adverse", "capital stress",
                         "adverse economic"], [], None, None),
    "sem-08": (["NVDA"], ["demand for Blackwell", "Blackwell demand", "demand for our",
                          "exceptional demand"], [], None, None),
    "sem-09": (["MSFT"], ["cyberattack", "security incident", "security breach",
                          "cybersecurity threat"], [], None, None),
    "sem-10": (["JPM"], ["common stock repurchase", "dividend", "return capital",
                         "capital distribution"], [], None, None),

    "kw-01": (["NVDA"], ["Blackwell"], [], None, None),
    "kw-02": (["JPM"], ["CET1"], [], None, None),
    "kw-03": (["NVDA"], ["Hopper"], [], SCOPE_ALL, ["MD&A", "Business", "body"]),
    "kw-04": (["JPM"], ["Basel III"], [], None, None),
    "kw-05": (["MSFT"], ["Azure"], [], None, None),
    "kw-06": (["NVDA"], ["stock-based compensation"], [], None, None),

    # Temporal: the concept AND the newest filing. Chunks with the same text in an
    # older filing are deliberately NOT relevant — that is the property being measured.
    "temp-04": (["NVDA"], ["Blackwell"], [], NEWEST["NVDA_10Q"], None),
    "temp-05": (["MSFT"], ["segment"], [], NEWEST["MSFT_10K"], None),
    "temp-06": (["JPM"], ["net revenue"], [], NEWEST["JPM_10Q"], ["MD&A"]),

    # No section filter on these: MSFT's 10-K puts MD&A revenue discussion under a
    # heading the parser labels "Risk Factors" (same class of defect as FC-1), so
    # filtering by section would silently drop the chunks that answer the question.
    # Term density stands in for "this passage is about revenue/demand".
    "x-01": (["NVDA"], ["revenue"], [], None, None),
    "x-02": (["NVDA"], ["demand"], [], None, None),
    "x-03": (["NVDA", "MSFT"], ["revenue"], [], None, None),
    "x-04": (["NVDA"], ["revenue"], [], None, None),
    "x-05": (["NVDA"], ["export control", "export licen"], [], None, None),
    "x-06": (["NVDA", "MSFT"], ["demand"], [], None, None),
}


def matching_chunks(conn, tickers, any_terms, all_terms, accession, sections,
                    min_hits: int = MIN_TERM_HITS) -> list[str]:
    where = ["d.ticker = ANY(%s)"]
    params: list = [tickers]
    if accession == SCOPE_ALL:
        pass                                    # no document restriction
    elif accession:
        where.append("c.chunk_id LIKE %s")
        params.append(f"{accession}%")
    else:
        where.append("(" + " OR ".join(["c.chunk_id LIKE %s"] * len(tickers)) + ")")
        params += [f"{CURRENT_10K[t]}%" for t in tickers]
    if sections:
        where.append("c.section = ANY(%s)")
        params.append(sections)
    if any_terms:
        where.append("(" + " OR ".join(["c.text ILIKE %s"] * len(any_terms)) + ")")
        params += [f"%{t}%" for t in any_terms]
    for t in all_terms:
        where.append("c.text ILIKE %s")
        params.append(f"%{t}%")
    for b in BOILERPLATE:
        where.append("c.text NOT ILIKE %s")
        params.append(f"%{b}%")

    cur = conn.cursor()
    cur.execute(f"""SELECT c.chunk_id, c.text FROM chunks c JOIN documents d USING(doc_id)
                    WHERE {' AND '.join(where)} ORDER BY c.chunk_id""", params)
    rows = cur.fetchall()
    if not any_terms:
        return [r[0] for r in rows]
    keep = []
    for chunk_id, text in rows:
        low = (text or "").lower()
        hits = sum(low.count(t.lower()) for t in any_terms)
        specific = any(len(t) >= SPECIFIC_PHRASE_CHARS and t.lower() in low for t in any_terms)
        if hits >= min_hits or specific:
            keep.append(chunk_id)
    return keep


def numeric_answer(conn, qid: str) -> dict:
    """Ground the numeric questions in the financials table, so the expected answer is
    whatever the pipeline's own source of truth says — not a number typed from memory."""
    plans = {
        "num-01": ("metric_series", {"ticker": "NVDA", "metric": "revenue", "n_quarters": 4}),
        "num-02": ("margin_trend", {"ticker": "MSFT", "n_quarters": 4}),
        "num-03": ("latest_metric", {"ticker": "NVDA", "metric": "eps_diluted"}),
        "num-04": ("latest_metric", {"ticker": "JPM", "metric": "revenue"}),
        "num-05": ("yoy_growth", {"ticker": "NVDA", "metric": "revenue", "n_quarters": 4}),
        "num-06": ("latest_metric", {"ticker": "MSFT", "metric": "operating_income"}),
    }
    name, params = plans[qid]
    rows = Q.run(conn, name, **params)
    if not isinstance(rows, list):
        rows = [rows] if rows else []
    return {"expected_query": name,
            "expected_values": [r.format() for r in rows if r is not None]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="evals/questions.jsonl")
    args = ap.parse_args()

    conn = get_conn()
    out = []
    try:
        for qid, category, route, query, tickers in QUESTIONS:
            rec = {"id": qid, "category": category, "route": route, "query": query,
                   "tickers": tickers}
            if qid in RULES:
                tickers_r, any_t, all_t, accession, sections = RULES[qid]
                min_hits = MIN_HITS_OVERRIDE.get(qid, MIN_TERM_HITS)
                # Content set: the concept anywhere in the corpus. This is what
                # "did retrieval find the right passage" means.
                content = matching_chunks(conn, tickers_r, any_t, all_t,
                                          accession if accession == SCOPE_ALL else SCOPE_ALL,
                                          sections, min_hits=min_hits)
                # Recency set: the copies in the current filing. Scored separately.
                current = matching_chunks(conn, tickers_r, any_t, all_t, accession,
                                          sections, min_hits=min_hits)
                assert content, f"{qid}: relevance rule matched nothing"
                rec["expected_chunk_ids"] = content
                rec["expected_current_ids"] = current
                rec["n_relevant"] = len(content)
                rec["n_current"] = len(current)
            elif category == "numeric":
                rec.update(numeric_answer(conn, qid))
            elif category == "insufficient_evidence":
                rec["expected_answer"] = "INSUFFICIENT EVIDENCE"
            elif category == "temporal":          # the three NEWS-window ones
                rec["expected_window_days"] = 7
                rec["expected_source"] = "news_articles"
            out.append(rec)
    finally:
        conn.close()

    with open(args.out, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")

    by_cat: dict[str, int] = {}
    for r in out:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"wrote {args.out}: {len(out)} questions {by_cat}")
    graded = [r for r in out if "expected_chunk_ids" in r]
    sizes = sorted(r["n_relevant"] for r in graded)
    print(f"retrieval-graded: {len(graded)}; |content| min={sizes[0]} "
          f"median={sizes[len(sizes)//2]} max={sizes[-1]}")
    for r in graded:
        print(f"  {r['id']:8s} content={r['n_relevant']:4d} current={r['n_current']:3d}"
              f"   {r['query'][:52]}")


if __name__ == "__main__":
    main()
