"""Question set for Phase 7, before expected_chunk_ids are attached.

Kept as Python (not JSONL) only while labelling — `scripts/label_questions.py` pools
retriever candidates for each of these and emits evals/questions.jsonl. Questions are
written against content verified to exist in the corpus, but *never* by copying the
wording of the chunk that should answer them: a question phrased in the chunk's own
vocabulary measures nothing except lexical overlap, which hands the result to BM25 by
construction. The semantic set deliberately paraphrases.
"""

QUESTIONS = [
    # --- semantic (10): paraphrased, low lexical overlap with the target chunk ---
    ("sem-01", "semantic", "FILING_RAG",
     "What did NVIDIA management say about China export restrictions?", ["NVDA"]),
    ("sem-02", "semantic", "FILING_RAG",
     "How does Microsoft describe competitive pressure in cloud?", ["MSFT"]),
    ("sem-03", "semantic", "FILING_RAG",
     "What does NVIDIA say about depending on a small number of large customers?", ["NVDA"]),
    ("sem-04", "semantic", "FILING_RAG",
     "How does JPMorgan describe the amount of capital regulators require it to hold?", ["JPM"]),
    ("sem-05", "semantic", "FILING_RAG",
     "What risks does NVIDIA cite around manufacturing and its suppliers?", ["NVDA"]),
    ("sem-06", "semantic", "FILING_RAG",
     "How does Microsoft describe its artificial intelligence partnerships?", ["MSFT"]),
    ("sem-07", "semantic", "FILING_RAG",
     "What does JPMorgan say about how it would hold up in a severe economic downturn?", ["JPM"]),
    ("sem-08", "semantic", "FILING_RAG",
     "What does NVIDIA say about demand for its newest generation of chips?", ["NVDA"]),
    ("sem-09", "semantic", "FILING_RAG",
     "How does Microsoft describe the risk of a security breach?", ["MSFT"]),
    ("sem-10", "semantic", "FILING_RAG",
     "What does JPMorgan say about returning capital to its shareholders?", ["JPM"]),

    # --- exact_keyword (6): rare tokens and acronyms, where sparse should win ---
    ("kw-01", "exact_keyword", "FILING_RAG", "What is Blackwell?", ["NVDA"]),
    ("kw-02", "exact_keyword", "FILING_RAG",
     "What does JPM say about CET1 ratio requirements?", ["JPM"]),
    ("kw-03", "exact_keyword", "FILING_RAG",
     "What does NVIDIA say about the Hopper architecture?", ["NVDA"]),
    ("kw-04", "exact_keyword", "FILING_RAG",
     "What does JPMorgan disclose about Basel III?", ["JPM"]),
    ("kw-05", "exact_keyword", "FILING_RAG",
     "What does Microsoft report about Azure?", ["MSFT"]),
    ("kw-06", "exact_keyword", "FILING_RAG",
     "What does NVIDIA disclose about stock-based compensation?", ["NVDA"]),

    # --- temporal (6): 3 news-window, 3 recency-of-filing ---
    ("temp-01", "temporal", "NEWS",
     "Give me all important NVDA news from the last 7 days", ["NVDA"]),
    ("temp-02", "temporal", "NEWS",
     "What happened with JPMorgan in the past week?", ["JPM"]),
    ("temp-03", "temporal", "NEWS",
     "What news came out about Microsoft recently?", ["MSFT"]),
    ("temp-04", "temporal", "FILING_RAG",
     "What did NVIDIA say about Blackwell demand in its most recent quarterly filing?",
     ["NVDA"]),
    ("temp-05", "temporal", "FILING_RAG",
     "What does Microsoft's latest annual report say about its segments?", ["MSFT"]),
    ("temp-06", "temporal", "FILING_RAG",
     "What did JPMorgan report for net revenue in its most recent quarterly filing?",
     ["JPM"]),

    # --- numeric (6): graded on the figure, not on chunks ---
    ("num-01", "numeric", "NUMERIC", "What was NVDA revenue for the last four quarters?", ["NVDA"]),
    ("num-02", "numeric", "NUMERIC",
     "How did Microsoft's gross margin trend over the last year?", ["MSFT"]),
    ("num-03", "numeric", "NUMERIC", "What was NVDA's most recent diluted EPS?", ["NVDA"]),
    ("num-04", "numeric", "NUMERIC",
     "What was JPMorgan's revenue in the most recent quarter?", ["JPM"]),
    ("num-05", "numeric", "NUMERIC",
     "How much did NVDA revenue grow year over year?", ["NVDA"]),
    ("num-06", "numeric", "NUMERIC",
     "What is Microsoft's most recent operating income?", ["MSFT"]),

    # --- cross_document (6): must span two stores ---
    ("x-01", "cross_document", "MIXED",
     "Revenue increased last quarter — what did management say caused it?", ["NVDA"]),
    ("x-02", "cross_document", "CROSS_SOURCE",
     "Does the recent NVDA news match what management said in the filings?", ["NVDA"]),
    ("x-03", "cross_document", "CROSS_SOURCE",
     "Compare NVDA and MSFT revenue growth", ["NVDA", "MSFT"]),
    ("x-04", "cross_document", "MIXED",
     "How has NVDA revenue changed over the last four quarters and why?", ["NVDA"]),
    ("x-05", "cross_document", "MIXED",
     "What does NVIDIA say about export restrictions and how does that affect its revenue?",
     ["NVDA"]),
    ("x-06", "cross_document", "CROSS_SOURCE",
     "Do NVIDIA and Microsoft describe similar AI demand trends?", ["NVDA", "MSFT"]),

    # --- insufficient_evidence (6): the correct answer is a refusal ---
    ("ie-01", "insufficient_evidence", "FILING_RAG",
     "What is NVIDIA's market share in Brazil?", ["NVDA"]),
    ("ie-02", "insufficient_evidence", "FILING_RAG",
     "What did the CFO say about the 2030 dividend policy?", ["NVDA"]),
    ("ie-03", "insufficient_evidence", "NUMERIC",
     "What was Tesla's revenue last quarter?", []),
    ("ie-04", "insufficient_evidence", "FILING_RAG",
     "How many employees does JPMorgan plan to hire in 2031?", ["JPM"]),
    ("ie-05", "insufficient_evidence", "FILING_RAG",
     "What is Microsoft's gross margin target for fiscal 2035?", ["MSFT"]),
    ("ie-06", "insufficient_evidence", "FILING_RAG",
     "What did NVIDIA say about its acquisition of Intel?", ["NVDA"]),
]

# Categories graded on retrieved chunks; the rest are graded on the answer.
RETRIEVAL_GRADED = {"semantic", "exact_keyword", "cross_document"}
