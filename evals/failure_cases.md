# Failure cases

Real failures found while building, with the evidence that exposed them. CLAUDE.md
Phase 7 requires ≥5 documented with the fix applied; cases registered before Phase 7
are listed here as they are found, with the measurement that will confirm or refute
the fix once the eval harness exists.

Status legend: **open** = observed, not yet fixed · **fixed** = fix applied and verified ·
**accepted** = understood, deliberately not fixed in v1.

---

## FC-1 — JPM section collapse: MD&A swallows Financial Statements (open)

**Observed in** the Phase 2 chunk distribution, after `scripts/build_index.py` over all
3,456 chunks. Grouped by ticker / doc_type / section:

| ticker | doc_type | section | chunks | chars |
|---|---|---|---:|---:|
| JPM | 10-Q | MD&A | **1,195** | 2,280,493 |
| JPM | 10-Q | Financial Statements | **3** | 6,085 |
| JPM | 10-K | Financial Statements | 666 | 1,333,152 |
| JPM | 10-K | MD&A | **1** | 665 |
| NVDA | 10-Q | MD&A | 92 | 194,218 |
| NVDA | 10-Q | Financial Statements | 137 | 270,051 |
| MSFT | 10-Q | MD&A | 113 | 261,999 |
| MSFT | 10-Q | Financial Statements | 252 | 532,375 |

Two distinct defects, both from the same root cause, and both invisible in aggregate
counts — the corpus totals look healthy.

**1. JPM 10-Q: MD&A absorbs the whole document.** NVDA and MSFT file 92–113 MD&A chunks
per 10-Q set; JPM files 1,195 — 419 / 363 / 413 across its three 10-Qs, ~780K chars
each. Meanwhile "Financial Statements" is *one* chunk of ~2,030 chars per filing.
Per DECISIONS #11, JPM never restates the `Item 1.` header before the real Financial
Statements content, so the section span opened at MD&A runs to end-of-document and
the financial statements are chunked *as MD&A*.

**2. JPM 10-K: MD&A is a cross-reference stub.** One chunk, 127 tokens. Its entire text:

> Item 7. Management's Discussion and Analysis of Financial Condition and Results of
> Operations. Management's discussion and analysis […] appears on pages 46–160. Such
> information should be read in conjunction with the Consolidated Financial Statements
> and Notes thereto, which appear on pages 165–314.

JPM incorporates MD&A by reference from the Annual Report exhibit rather than inlining
it in the 10-K body. The section parsed *correctly*; there is genuinely nothing there.
The content is in a document the Phase 1 ingester never fetched (it takes the primary
10-K document only).

**Why this is a retrieval failure, not a parsing nit:**

- `section` is a filterable payload field. A query filtered to `sections=["Financial
  Statements"]` sees 3 JPM 10-Q chunks and silently misses ~2.3M chars of real
  statements sitting under the wrong label. The filter doesn't error — it returns
  confidently wrong recall, the exact post-filter failure mode DECISIONS #15 exists
  to prevent, arriving through the metadata instead of through the query path.
- JPM 10-Q MD&A alone is **34.6% of the entire corpus** (1,195 / 3,456). That skews
  BM25's IDF: banking vocabulary appears common corpus-wide, deflating its weight on
  exactly the queries where it should discriminate.
- A "what does JPM's MD&A say" query against the 10-K retrieves a pointer to page
  numbers — well-formed, on-topic, and containing no answer. The generator has no way
  to tell that from real evidence; it is a plausible source of a confident non-answer.

**Candidate fixes** (choose on Phase 7 evidence, not in advance):
1. Fall back to a Part-boundary span for 10-Q Item 1 when the Item header is missing —
   narrow, fixes defect 1, does nothing for defect 2.
2. Follow the 10-K's incorporation-by-reference to the Ex-13 Annual Report exhibit —
   fixes defect 2, and is the same class of fix as DECISIONS #10 (read the exhibit, not
   the cover document), which suggests the ingester's "primary document" assumption is
   wrong in more than one place.
3. Structural DOM parsing instead of text regex — rejected for now per DECISIONS #11.

**Phase 7 measurement.** Add JPM `Financial Statements` and JPM MD&A questions to
`evals/questions.jsonl` (cross-document and insufficient-evidence categories). Expected
now: section-filtered JPM retrieval shows near-zero recall while unfiltered retrieval
still finds the text under the MD&A label — which is the signature of a metadata defect
rather than a retrieval one, and the reason the two are measured separately.

---

## FC-2 — BM25 wins where dense cannot: rare product tokens (registered)

**Observed in** `scripts/smoke_retrieval.py` against the live corpus. Query `"Blackwell"`:
BM25's top-1 chunk ranks **22nd** on the dense side, and **8 of 30** fused candidates
are chunks the dense arm never surfaced at rank ≤30.

bge-small has no useful representation for a product codename it never saw in training;
it lands somewhere generic in embedding space. BM25's IDF makes the same token the
strongest available signal. Not a bug — this is the case hybrid retrieval exists for,
recorded as the concrete BM25-wins instance CLAUDE.md Phase 7 asks for, with the
dense-only arm's Recall@5 on exact-keyword questions as the number that proves it.

---

<!-- Remaining required cases (CLAUDE.md Phase 7): dense-wins, stale-doc-outranks-fresh,
     chunk-boundary, syndicated-news-inflation. Add as found during eval. -->
