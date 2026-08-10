# Engineering Decisions Log (interview prep)

Format per entry: Problem / Choice / Alternative / Tradeoff / Likely interview question.
Claude Code: append an entry after every phase. Seed entries below.

## 1. No LangChain/LlamaIndex
- Problem: demonstrate understanding, not framework wiring.
- Choice: direct qdrant-client, rank_bm25, sentence-transformers, psycopg.
- Alternative: LlamaIndex would give ingestion + retrievers for free.
- Tradeoff: more code we own; but every interview answer is "here's the 15 lines that do it" instead of "the framework handles it."
- Interview q: "Why not LangChain?" → abstractions hid the retrieval knobs we needed to tune (fusion constant, filter pushdown, rerank cutoff), and debugging retrieval quality requires seeing raw candidates.

## 2. RRF for hybrid fusion
- See docstring in src/retrieval/hybrid.py. Rank-based → no score normalization across incomparable scales. k=60 standard.
- Interview q: "When does hybrid beat dense?" → exact tickers, product names (Blackwell), acronyms (CET1), accounting terms. Show eval rows kw-01/kw-02.

## 3. Structured financials in Postgres, not vectors
- Problem: "revenue over last 4 quarters" needs exactness and arithmetic; embeddings retrieve approximately.
- Choice: XBRL companyfacts → typed rows; canned parameterized queries; router sends numeric questions to SQL.
- Alternative: text-to-SQL. Deferred: injection surface + hard to eval; v1 uses a fixed query catalog.
- Interview q: "How do you combine both?" → MIXED route runs SQL first, injects the result table into RAG context as a numbered source, so the narrative answer can cite the numbers.

## 4. ANN (HNSW) in Qdrant
- Problem: exact kNN is O(N·d) per query; fine at 10k chunks, dead at 10M.
- Choice: HNSW defaults; measure recall vs exact search on our corpus (it'll be ~1.0 at this scale — say that honestly).
- Tradeoff: memory for graph links + approximate recall vs sub-ms search. Knobs: ef (recall/latency), m (memory/recall).
- Interview q: "Why ANN if your corpus is small?" → it isn't needed at this scale; it's the production-path default so scaling is a config change, not a rearchitecture. Knowing WHY it's overkill now is the point.

## 5. Section-aware chunking
- Problem: fixed-window chunking splits Risk Factors mid-sentence into MD&A context; retrieval returns plausible-but-wrong sections.
- Choice: parse Item headers first, chunk within sections, never cross boundaries; section stored as filterable metadata.
- Interview q: "How do you pick chunk size?" → eval-driven: we compare recall@5 at 400/750/1000 tokens if time permits; 750 chosen as default because filings paragraphs are long and bge-small context is 512 wordpieces (chunks are embedded truncated — acknowledge this and note the alternative: smaller chunks or a longer-context embedder).

## 6. Hard filters before similarity for news
- Problem: "NVDA news last week" via pure vector search returns semantically-similar but stale articles.
- Choice: SQL WHERE ticker AND published_at >= now()-7d first; similarity only ranks within the filtered set.
- Interview q: "What's the general principle?" → push deterministic constraints down to the store; use embeddings only for what's genuinely fuzzy.

## 7. Dedup + event clustering for news
- Problem: 20 syndicated copies of one story = fake importance signal and redundant LLM context.
- Choice: URL exact-dedup, headline-embedding near-dedup (>0.90), agglomerative clustering (~0.75) → events; rank events by article count × source diversity.
- Interview q: "Why cluster before summarizing?" → token budget goes to distinct events; count-based importance only works after dedup.
