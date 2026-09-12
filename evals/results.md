# Retrieval evaluation

Run `20260912T155922` · 25 retrieval-graded questions · top-5

Ground truth is built by `scripts/build_questions.py` from explicit, re-runnable relevance rules; see that module for the method and its bias.

## Overall

| retriever | Recall@5 | Precision@5 | MRR | Hit@5 | Freshness | median ms |
|---|---|---|---|---|---|---|
| `dense` | 0.050 | 0.360 | 0.613 | 0.720 | 0.276 | 17 |
| `bm25` | 0.059 | 0.360 | 0.457 | 0.640 | 0.284 | 6 |
| `hybrid` | 0.062 | 0.416 | 0.613 | 0.680 | 0.259 | 24 |
| `rerank` | 0.060 | 0.432 | 0.508 | 0.680 | 0.239 | 133 |

Recall@5 ceiling on this question set is **0.187** — relevant sets run 7-112 chunks, so recall is bounded by `5/|relevant|`. Reported unnormalized on purpose. Precision@5 and MRR are unaffected by set size and are the fair cross-question comparison; **Freshness** is the share of retrieved relevant chunks that came from the company's current filing (FC-10).

## By category (MRR)

| category | `dense` | `bm25` | `hybrid` | `rerank` |
|---|---|---|---|---|
| cross_document | 0.458 | 0.333 | 0.389 | 0.361 |
| exact_keyword | 0.708 | 0.708 | 0.833 | 0.750 |
| semantic | 0.633 | 0.467 | 0.600 | 0.453 |
| temporal | 0.667 | 0.167 | 0.667 | 0.500 |

## By category (Precision@5)

| category | `dense` | `bm25` | `hybrid` | `rerank` |
|---|---|---|---|---|
| cross_document | 0.300 | 0.233 | 0.300 | 0.333 |
| exact_keyword | 0.467 | 0.633 | 0.600 | 0.633 |
| semantic | 0.320 | 0.360 | 0.360 | 0.360 |
| temporal | 0.400 | 0.067 | 0.467 | 0.467 |

## Per question (MRR)

| question | |rel| | `dense` | `bm25` | `hybrid` | `rerank` |
|---|---|---|---|---|---|
| kw-01 | 38 | 1.00 | 1.00 | 1.00 | 1.00 |
| kw-02 | 45 | 0.25 | 1.00 | 0.50 | 1.00 |
| kw-03 | 16 | 0.00 | 1.00 | 0.50 | 0.50 |
| kw-04 | 29 | 1.00 | 1.00 | 1.00 | 1.00 |
| kw-05 | 28 | 1.00 | 0.00 | 1.00 | 0.00 |
| kw-06 | 112 | 1.00 | 0.25 | 1.00 | 1.00 |
| sem-01 | 60 | 1.00 | 1.00 | 1.00 | 1.00 |
| sem-02 | 19 | 1.00 | 0.50 | 1.00 | 1.00 |
| sem-03 | 7 | 0.00 | 0.33 | 0.00 | 0.00 |
| sem-04 | 37 | 1.00 | 1.00 | 1.00 | 1.00 |
| sem-05 | 21 | 0.00 | 0.50 | 0.00 | 0.00 |
| sem-06 | 17 | 1.00 | 1.00 | 1.00 | 1.00 |
| sem-07 | 31 | 1.00 | 0.00 | 1.00 | 0.20 |
| sem-08 | 24 | 0.00 | 0.00 | 0.00 | 0.00 |
| sem-09 | 20 | 1.00 | 0.00 | 1.00 | 0.00 |
| sem-10 | 65 | 0.33 | 0.33 | 0.00 | 0.33 |
| temp-04 | 38 | 1.00 | 0.50 | 1.00 | 0.50 |
| temp-05 | 60 | 1.00 | 0.00 | 1.00 | 1.00 |
| temp-06 | 16 | 0.00 | 0.00 | 0.00 | 0.00 |
| x-01 | 32 | 0.00 | 0.50 | 0.33 | 0.33 |
| x-02 | 29 | 0.00 | 0.00 | 0.00 | 0.00 |
| x-03 | 81 | 0.50 | 0.00 | 0.00 | 0.33 |
| x-04 | 32 | 0.25 | 0.00 | 0.00 | 0.00 |
| x-05 | 60 | 1.00 | 1.00 | 1.00 | 1.00 |
| x-06 | 33 | 1.00 | 0.50 | 1.00 | 0.50 |

<!-- answer-eval -->
## Answer quality (LLM-as-judge)

Run `20260912T155611` · 40 questions · judge prompt is separate from the generator
and never sees the expected answer.

| overall | value |
|---|---|
| groundedness | 4.78 / 5 |
| citation correctness (judge) | 4.97 / 5 |
| relevance | 5.00 / 5 |
| route accuracy | 0.80 |
| citations valid (checked in code) | 1.00 |
| answers citing nothing | 0.20 |
| refusal accuracy (declined when it should) | 0.95 |
| sentinel accuracy (declined *in the contract's words*) | 0.95 |
| answers with >=1 unsupported claim | 0.15 |

| category | n | ground. | cite (judge) | relev. | route | cites valid |
|---|---|---|---|---|---|---|
| cross_document | 6 | 4.33 | 5.00 | 5.00 | 0.83 | 1.00 |
| exact_keyword | 6 | 4.83 | 5.00 | 5.00 | 0.83 | 1.00 |
| insufficient_evidence | 6 | 4.83 | 5.00 | 5.00 | 0.50 | 1.00 |
| numeric | 6 | 5.00 | 5.00 | 5.00 | 1.00 | 1.00 |
| semantic | 10 | 4.80 | 4.90 | 5.00 | 0.90 | 1.00 |
| temporal | 6 | 4.83 | 5.00 | 5.00 | 0.67 | 1.00 |

Citation validity is decided by `answer.check_citations`, not by the judge — whether
`[7]` exists when six items were supplied is decidable, and the judge is demonstrably
unreliable on it (it scored citation_correctness 5/5 on an answer carrying zero
citations). The judge is used only for the part that needs reading.
