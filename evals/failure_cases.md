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

## FC-3 — MSFT's "Business Outlook" is a pointer, not guidance (registered)

**Observed in** Phase 4 guidance extraction. NVDA's 8-K outlook blocks carry real
numbers ("Revenue is expected to be $108.0 billion, plus or minus 2%"). MSFT's
matching block says:

> Business Outlook — Microsoft will provide forward-looking guidance in connection
> with this quarterly earnings announcement on its earnings conference call and
> webcast.

The section exists, parses correctly, and contains no guidance. MSFT gives guidance
verbally on the call; the call transcript is the licensed source this build
deliberately does not scrape (README, DECISIONS #10). JPM issues no press-release
guidance at all and has zero rows.

**Why it matters:** "What is Microsoft's guidance for next quarter?" retrieves a
well-formed, on-topic, correctly-dated row containing no answer — the same shape as
FC-1's JPM 10-K MD&A stub. A generator with no groundedness check will summarize it
into a confident non-answer. Two independent sources of this failure in one corpus
suggests it is the dominant hallucination risk in the build, not an edge case.

**Phase 7 measurement.** This is an *insufficient-evidence* question, not a retrieval
one: retrieval succeeds and the correct answer is "not disclosed in the filings I
have". Added to `evals/questions.jsonl` in that category, where the judge scores
whether the generator declined rather than whether it retrieved.

---

## FC-4 — Syndicated-news inflation, and what dedup can't reach (fixed + open)

CLAUDE.md names "syndicated-news-inflation" as a required failure case. On a live
7-day window it showed up in **three distinct forms**, only one of which dedup solves.

**1. Literal syndication — fixed by URL+embedding dedup.** Google News assigns a
*unique* redirect URL to every item, so the same wire story from twenty outlets
arrives as twenty distinct URLs. Exact-URL dedup collapses none of it. The
headline-embedding pass (cosine ≥ 0.90, earliest kept) catches them: an identical
JPM headline arriving from both the Google and Yahoo feeds was correctly flagged,
with the earliest copy surviving.

**2. Company-name similarity inflation — fixed by normalization (DECISIONS #22).**
The configured 0.75 threshold merged 39 unrelated NVDA articles into one "event"
because the shared company name adds a constant ~0.13 to every pairwise cosine.
Measured: 690 spurious above-threshold pairs before normalization, 51 after.

**3. Bot-generated volume — *not* a dedup problem (open).** MarketBeat posted 18 JPM
articles in one week: "$JPM Shares Acquired by Acumen Wealth Advisors LLC",
"...by Todd Asset Management LLC", and so on — one per 13F filing. These are
genuinely *different* stories with different entities, so dedup correctly leaves them
alone, and they clustered into one 18-article event. Ranked by article count that is
the week's top story. It is not news.

Mitigated by capping volume in the ranker (DECISIONS #23) so 2 real sources outrank
18 single-source posts — but the articles are still in the corpus, still inflate
`count(*)` for JPM, and would still be retrieved by a NEWS query that filters only on
ticker and date. A source-quality prior or a template-detection pass is the real fix;
neither is in v1.

**Phase 7 measurement.** Precision@5 on NEWS-route questions, scored on whether the
returned events are things a human analyst would call news. The prediction is that JPM
scores worst of the three tickers, because it has no IR feed (DECISIONS #21) and the
highest bot-post share.

---

## FC-5 — Headline clustering yields topics, not events (open)

**Observed** after the FC-4 normalization fix. The resulting clusters are coherent but
they are *themes*, not discrete events: NVDA's top cluster is five outlets publishing
price predictions, JPM's is generic "is JPM a buy" commentary pages. A real discrete
event ("India lifts ban on JPMorgan unit, broker in market-manipulation case") appears
in only 2 articles and ranks second.

**Cause:** these feeds are dominated by commentary rather than reporting, and 181 of
206 articles are headline-only (Google redirects, DECISIONS #21), so clustering has
roughly ten words per article to work with. Ten words about one company, with the
company name removed, is mostly sentiment vocabulary — which is exactly what clusters.

**Why it's recorded rather than fixed:** the honest fix is better source material
(full text, or feeds weighted toward reporting over commentary), not a better
clustering algorithm. Tuning linkage or threshold would move articles between topic
buckets without making topic buckets into events.

**Phase 7 measurement.** Hand-label one week of NVDA clusters as event vs theme and
report the ratio, alongside whether the brief's top-3 overlaps an analyst's top-3.
This is the number that says whether the news pipeline is useful or merely working.

---

<!-- Remaining required cases (CLAUDE.md Phase 7): dense-wins, stale-doc-outranks-fresh,
     chunk-boundary. Add as found during eval. -->

## FC-6 — The summarizer invents from headlines, and cites while doing it (fixed)

**Symptom.** The NVDA weekly brief asserted that analysts and investors "cite its
dominance in the AI and gaming markets", and attributed results to "a diversified
product portfolio, including its successful graphics processing units (GPUs) and
artificial intelligence (AI) offerings" — while citing [1] and [2]. Neither claim is
in either article. The model had two headlines and nothing else.

**Why it happened.** Two defects that only bite together.

1. `brief._numbered_context` passed **headlines only** and never the body text, so the
   trafilatura extraction that ingestion performs was discarded before reaching the
   prompt — and a full-text article was byte-identical to a headline-only one from the
   model's point of view.
2. `EVENT_SUMMARY_SYSTEM` demanded "potential market impact, bull implication, bear
   implication" unconditionally. With one line of evidence, that is an instruction to
   invent.

176 of 216 collected articles are Google News redirects with unrecoverable bodies, so
this was the *typical* case, not an edge one. And a headline reads like sufficient
context — which is what makes it more dangerous than an empty source. The citation is
the aggravating factor: it makes invention look sourced, so the reader's normal
defence (check the citation) confirms the wrong thing.

**Fix.** Label each article `FULL TEXT` or `HEADLINE ONLY` in the context and include a
body excerpt when one exists; add an evidence rule to the prompt that overrides the
output format; require the exact sentence "Not supported by headline-only sources." for
the four interpretive sections when the whole cluster is headline-only. See DECISIONS #25.

**Regression test.** `scripts/smoke_headline_only.py` — a fixture of three headlines that
state what happened and never why, so any cause, attribution or figure is fabricated by
construction. Deterministic regex properties, not an LLM judge: the judge shares the
generator's blind spot and cannot fail a build. Runs N trials because generation is
stochastic.

**What writing the test taught.** The first draft asserted the obvious phrases
("analysts say", "due to") and caught essentially nothing — against the pre-fix prompt
it fired on one check, the decline sentence, which is circular. Reading the real
pre-fix output showed the model rarely attributes to a named group; it writes "is seen
as a strategic expansion", "are expected to support", "is likely to have a positive
impact". Agentless hedging that reads as analysis. Widening to that class took the
negative control from 1 violation per trial to 7-10. **A gate you have not watched fail
is not a gate.**

**Evidence.** `evals/before_realtext_NVDA.md` and `evals/after_realtext_NVDA.md`, same
ticker and window. After: 4/4 fixture trials clean, 3/3 headline-only events in the live
brief decline their implications, and the two events that do carry body text still cite
real figures ("revenue surged 106% year over year to $96.2 billion") that appear verbatim
in the article bodies.

**Still open.** For an all-headline-only cluster the generated summary now degenerates to
roughly a list of the headlines — close to what the deterministic extractive fallback
already produces. That is the honest output for that input, but it means the LLM earns
its place on full-text clusters and barely any on headline-only ones. The lever worth
pulling is body-text coverage (currently 37/209), not more prompt tuning.

---

*The five cases below were found by running Phase 7's retrieval eval
(`python -m src.evals.retrieval_eval`, 25 retrieval-graded questions, 4 arms).
Numbers are from run `20260912T155301` and are reproducible from
`evals/questions.jsonl`.*

## FC-7 — BM25 wins: a rare token the embedding has never really seen (registered)

**Case.** `kw-03`, "What does NVIDIA say about the Hopper architecture?"
MRR: **dense 0.00, BM25 1.00**, hybrid 0.00, rerank 0.50.

**Why.** "Hopper" is a product codename appearing in 16 chunks corpus-wide, almost all
single-mention, and none in the current 10-K — Blackwell superseded it. bge-small has
no useful vector for a proper noun this rare, so dense retrieval falls back to general
"NVIDIA architecture" semantics and returns the Business section boilerplate. BM25 does
not need to understand the token; it only needs it to be rare, which is exactly when
IDF is largest.

**Status: registered, not fixed.** This is the case hybrid retrieval exists for. It is
recorded because it is the concrete evidence for that design choice, and because it
shows the failure is a property of the *embedding model's vocabulary*, not of a tuning
constant — no amount of reranking recovers a chunk the dense arm never retrieved.

## FC-8 — Dense wins: the question and the answer share no vocabulary (registered)

**Case.** `kw-05`, "What does Microsoft report about Azure?"
MRR: **dense 1.00, BM25 0.00**, and `sem-09` ("risk of a security breach") dense 1.00,
BM25 0.00.

**Why.** BM25 ranks by term overlap, and "Azure" appears in dozens of revenue tables
that mention it once in a list. The chunk that actually *reports on* Azure —
"Revenue in Intelligent Cloud was $30.9 billion and increased 28%" — is out-competed by
boilerplate with a higher raw term count. sem-09 is the sharper version: the 10-K says
"cybersecurity threat environment", the question says "security breach", and the lexical
overlap is nearly zero.

**Status: registered.** The mirror image of FC-7, and the reason neither arm is dropped.

## FC-9 — RRF scored *below* its own best arm (fixed)

**The most useful thing the eval found.** Hybrid was supposed to be at least as good as
its components. It was not:

| arm | MRR | Hit@5 | Precision@5 |
|---|---|---|---|
| dense | 0.613 | 0.720 | 0.360 |
| bm25 | 0.457 | 0.640 | 0.360 |
| hybrid (plain RRF) | **0.500** | **0.560** | 0.392 |

Hybrid lost to dense on both MRR and Hit@5, and had a *lower hit rate than either arm*.

**Why — the arithmetic, traced on `kw-05`.** RRF scores a chunk `1/(k+rank)` per arm,
k=60. The correct chunk was dense rank 1 and absent from BM25's list: `1/61 = 0.0164`.
An irrelevant chunk ranked 4th by dense and 6th by BM25 scored
`1/64 + 1/66 = 0.0308` — **nearly double**. Rank-only fusion therefore prefers
"mediocre in both arms" to "the single best match in one arm", by construction. When
the two arms are of comparable quality that is the desired consensus behaviour; when one
arm is clearly right and the other has no idea, it destroys the answer. kw-05 went from
MRR 1.00 (dense) to 0.00 (hybrid).

**Fix.** `rrf_fuse(..., anchor=True)` keeps each arm's rank-1 result in the fused top-2.
No weight is tuned on the eval set — the fix restores the one piece of information
rank-only fusion discards, namely that a chunk was some retriever's top choice.

| arm | MRR | Hit@5 | Precision@5 |
|---|---|---|---|
| hybrid (anchored) | **0.613** | **0.680** | **0.416** |

Hybrid now matches the best single arm on MRR and beats both on precision. The cost is
that the fused list is no longer globally score-descending, which broke an assertion in
`scripts/smoke_retrieval.py` — the scores were left truthful and the *invariant* was
changed rather than inflating scores to preserve it.

**Still open:** rerank has the best Precision@5 (0.432) but MRR 0.508, below dense's
0.613 — the cross-encoder improves the *set* while sometimes demoting the single best
chunk. Whether that trade is worth 100ms per query is a Phase 8 A/B, not a guess.

## FC-10 — Nothing prefers the current filing (open)

**Case.** Freshness — the share of retrieved relevant chunks that come from the
company's *current* filing — is **0.24-0.28 across all four arms**. `temp-04`
("Blackwell demand in its most recent quarterly filing") retrieves the right content
from FY2025Q1 and FY2024Q4 8-Ks.

**Why.** This corpus repeats disclosures near-verbatim across a 10-K and three or four
10-Qs. Retrieval ranks on similarity alone, and identical text scores identically, so
which copy surfaces is arbitrary — filing date is carried in the payload as metadata and
used only as a hard filter, never as a ranking signal.

**Why it matters more than it looks.** Every arm answers "what does the company say
about X" from whichever copy the index happened to favour. For a risk factor repeated
unchanged that is harmless; for anything restated between filings it silently answers
from stale disclosure, and the citation makes it look verified.

**Status: measured and open.** The fix is a recency prior in ranking (a small date-decay
term, or preferring the newest document among near-duplicate texts). Not applied here
because tuning a decay constant against 25 questions would fit the question set rather
than the problem. The metric is in place, so the fix is now measurable.

## FC-11 — The number and its label are in different chunks (open)

**Case.** `temp-06`, "What did JPMorgan report for net revenue in its most recent
quarterly filing?" — MRR **0.00 on every arm**.

**Why.** The chunk holding the figure is
`...054343::md-a::0040`: *"2026 2025 Change 2026 2025 Change Total net revenue
$ 20,272 $ 18,847 8 % Total noninterest expense 11,108 9,858 13 ..."* — a bare grid of
numbers. The table's header row, which says which segment and which period the columns
belong to, was split into a **different chunk** by the paragraph-splitting chunker.

So the chunk is close to unretrievable (a dense vector over mostly digits carries almost
no topical signal; BM25 sees two content words) and, worse, it is close to *uncitable*:
a generator handed it cannot tell which period `$20,272` belongs to. **149 of 3,456
chunks (4.3%) are more than 55% numeric tokens**, concentrated in JPM's MD&A (92 chunks,
7.7% of that section) — JPM is worst because bank filings are mostly tables.

Note the retriever did return `...054343::md-a::0012` — *"For the second quarter of
2026, JPMorganChase reported net income of $21.2 billion, up 41%"* — which is the better
answer for a human. It scored zero because the relevance rule demanded "net revenue".
That is a ground-truth limitation, recorded rather than quietly corrected.

**Status: open.** The real fix is table-aware chunking: keep a table's header with its
body, or serialize tables to labelled rows before chunking. That is a Phase 2 change
with re-ingestion behind it, so it is scoped as v2 — and it is the strongest argument in
this document for why **numbers live in Postgres and not in the vector store**
(CLAUDE.md principle #2). The structured path answers this question correctly today;
the RAG path cannot.

## FC-12 — A refusal that smuggles claims back in (open)

**Case.** `ie-02`, "What did the CFO say about the 2030 dividend policy?" — no such
policy is disclosed, so the correct answer is a refusal. What came back:

> The CFO of Microsoft did not explicitly mention the 2030 dividend policy in the
> provided context. **However, based on the statements from the CFO in the 10-K and 10-Q
> filings, it can be inferred that the company intends to continue returning capital to
> shareholders in the form of dividends** [...]
>
> Facts
> - The Board of Directors declared dividends totaling $27.0 billion [...]

It declines in the first sentence and then answers anyway, with a Facts section and four
citations. The judge scored it groundedness 4/5 and listed **no** unsupported claims,
because every individual sentence *is* supported — the dividend figures are real. The
defect is not a false statement; it is that a question with no answer received one.

**Why the existing guards miss it.** `check_citations` passes (all four citations
resolve). `Answer.uncited` passes (it cites plenty). The refusal test passes (the answer
does contain a refusal phrase). Every mechanical guard sees a well-formed, well-cited
answer. This is the FC-6 pattern one level up: there, invented content wore a citation;
here, *relevant-but-off-question* content wears one.

**Status: open.** The honest fix is a contract change — when the answer declines, it
must stop, rather than declining and continuing. That is a prompt change whose Phase 6
lesson was that tuning this prompt for coverage made things strictly worse, so it needs
to be made against this eval rather than by feel. The measurement now exists.

## FC-13 — The LLM routing fallback is net-negative on the eval set (open)

**Case.** Route accuracy over all 40 questions:

| routing | accuracy | fallback used | fallback wrong |
|---|---|---|---|
| rules only, abstain -> FILING_RAG | **0.88** | 18 | 1 |
| rules + LLM fallback | 0.82 | 18 | 3 |

The LLM classifier is consulted on the 18 questions no rule matched, and it does worse
than simply defaulting to FILING_RAG. It sent "What is Blackwell?" to MIXED, "demand for
its newest generation of chips" to NEWS, and "NVIDIA's market share in Brazil" to NUMERIC.

**Tension with DECISIONS #26, stated rather than smoothed over.** Phase 6 measured the
few-shot fallback at 7/7 on a hand-built set of vague queries, up from 2/7 — that result
was real, and it is not what this table contradicts. What this shows is that on the
*eval distribution*, the abstention default happens to be right more often than the
model's guess. The difference is **two questions**, which is noise at n=40. So the
fallback is not being removed on this evidence.

**What would settle it:** more questions in the ambiguous band, and a per-route breakdown
of where the fallback helps versus hurts. Recorded as an open question with a number
against it, rather than as a conclusion.

**One genuine rule bug this did surface, now fixed:** `temp-05`, "Microsoft's *latest*
annual report", routed to NEWS — "latest" matched the recency table and "annual report"
was missing from `FILING_PAT`, so nothing marked it as a filing question. Same class as
the "last quarter" bug in DECISIONS #26: a document noun has to outrank a recency
adjective. Adding the document nouns fixed it and lifted rules-only accuracy to 0.88.
