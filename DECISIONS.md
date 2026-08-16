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

## 8. Storage abstraction: local (SQLite + embedded Qdrant) vs docker (Postgres + Qdrant server)
- Problem: Docker Desktop on Windows drags in WSL2 + virtualization + reboots. Blocking a weekend build on container infra is a bad trade.
- Choice: one seam (`src/common/db.py::get_conn/get_qdrant`), mode flag in config.yaml. SQLite and embedded Qdrant for dev; Postgres and Qdrant server for deployment. Nothing in the v1 schema needs Postgres-specific features.
- Alternative: force Docker everywhere; or hardcode SQLite and lose the prod story.
- Tradeoff: two DDL files to keep in sync, and SQLite gives up concurrent writers + native TIMESTAMPTZ. Acceptable because ingestion is single-process and dates are stored ISO8601.
- Interview q: "How would you deploy this?" → docker-compose.yml is in the repo; flip storage.mode to "docker" and nothing downstream changes, because no module imports a driver directly. That seam is the actual design decision — the specific databases are swappable details.

## 9. EDGAR ingestion: submissions API + fixed-interval rate limiter
- Problem: pull the right filing set (1 latest 10-K, last 4 10-Qs, recent earnings 8-Ks) per ticker without hitting SEC's fair-access limits.
- Choice: `data.sec.gov/submissions/CIK##########.json` gives a compact parallel-array list of every recent filing (form, date, accession, primaryDocument, items) in one request; filter/sort/slice in Python. A `RateLimiter` (fixed 1/max_rps interval, config-driven) wraps every outbound request; `documents.doc_id = accession` makes re-running the ingest idempotent (`SELECT` before `INSERT`, skip if present) with no separate flow runner needed.
- Alternative: full-text search API (`efts.sec.gov`) — better for keyword discovery, worse for "give me the N most recent filings of type X," which is what v1 needs.
- Interview q: "How do you pick which 8-Ks matter?" → most 8-Ks are non-events for RAG purposes (exec changes, unrelated 8-K items); filtering to Item 2.02 ("Results of Operations and Financial Condition") isolates the earnings-related ones, capped at `n_8k` (config) most recent.

## 10. 8-K primary document is not the earnings release — read the exhibit
- Problem: an 8-K's `primaryDocument` (from the submissions API) is the Inline-XBRL cover form — mostly machine-readable XBRL tag scaffolding, almost no narrative text. The actual earnings press release CLAUDE.md calls for (the free substitute for a paid transcript) is filed as a separate exhibit.
- Choice: for 8-Ks, fetch the filing's own `{accession}-index.html` and read its **Type** column (EDGAR-provided document metadata, e.g. `EX-99.1`) to find the earnings-release exhibit, and download that instead of `primaryDocument`. Verified against a real JPM 8-K: primary doc was 46KB of XBRL facts; the EX-99.1 exhibit was the actual "JPMORGAN CHASE & CO. EARNINGS RELEASE" narrative.
- Alternative: guess the exhibit by filename pattern (e.g. contains "ex99"). Rejected — filer/law-firm naming conventions vary; the Type column is authoritative and no more expensive (one extra rate-limited request per 8-K).
- Interview q: "How did you find that bug?" → didn't assume the API response was semantically what its field name implied; read the actual downloaded content before trusting the pipeline, which is what caught it (first ingestion run silently stored XBRL noise as the "filing" text).

## 11. Section parsing: line-anchored Item-header regex + TOC-vs-real span heuristic + Part scoping
- Problem: 10-K/10-Q HTML has no structural section tags — regex on "Item N" headers is the only cheap signal, but (a) filings open with a table of contents that repeats every heading, and (b) 10-Qs reuse item numbers across Part I and Part II with unrelated meanings (Part I Item 2 = MD&A; Part II Item 2 = "Unregistered Sales of Equity Securities").
- Choice: match `^item\s+<id>` at line start (real headers are their own block-level line after HTML stripping; TOC entries are too, so this alone doesn't disambiguate). For each wanted item id, keep the occurrence with the largest gap to the *next* Item heading of any kind — real sections run for pages, TOC entries are immediately followed by the next TOC line. For 10-Qs additionally track Part I/II boundaries and require each wanted item to fall in its correct Part, since item-id alone is ambiguous there.
- Verified against real filings: NVDA (4/4 10-Qs) and MSFT (3/3 10-Qs) parse cleanly with sane section sizes. **JPM's 10-Qs are the counter-example**: JPM never restates the "Item 1." header before the real Financial Statements content — only the TOC contains that exact text — so the span heuristic has no real occurrence to prefer and the section is under-sized (~2K chars of the true ~600K+). MD&A and Risk Factors parse correctly for JPM (those headers *are* restated). This is the "chunk-boundary case" carried into Phase 7's failure_cases.md rather than special-cased here.
- Alternative: parse the raw HTML DOM for bold/heading tags instead of text regex. More robust to this exact failure, but filer-specific formatting variance (font tags vs. CSS classes vs. inline styles across different 10-year-old EDGAR HTML dialects) trades one narrow failure mode for a different, harder-to-predict one. Deferred — not worth it for a 3-ticker v1; revisit if eval numbers show this actually costs recall.
- Interview q: "Why not just always take the last occurrence of each header?" → tried it conceptually; breaks the opposite way when a section is legitimately referenced late in the document (e.g., "Item 7A" cross-referenced from deep in "Item 9"). The span heuristic (biggest gap to the next heading) is a better proxy for "this is real content" than positional recency.

## 12. Chunk text lives in SQL only; Qdrant payload is metadata
- Problem: three stores now hold the same chunks (SQL rows, Qdrant points, BM25 corpus). Whoever holds the text is the one that must never be stale.
- Choice: `chunks.text` in the relational store is the single source of truth. Qdrant payload carries metadata only (ticker/company/doc_type/filing_date/filing_ts/fiscal_period/section/source_url/chunk_id/doc_id/chunk_index/n_tokens); the BM25 pickle carries tokens + metadata. Both retrievers return `chunk_id`s and hydrate text with one `WHERE chunk_id IN (...)` (`src/indexing/store.load_texts`).
- Alternative: duplicate the text into the Qdrant payload — one fewer round trip per query, self-contained vector store.
- Tradeoff: we pay a SQL fetch per search (single indexed PK lookup, negligible at any scale we'd hit) and gain the guarantee that changing chunking params can't leave a Qdrant payload disagreeing with the row a citation renders from. Payload also stays small enough to keep hot.
- Interview q: "Why not just store text in the vector DB?" → because then the same string exists in three places with no transaction spanning them, and the copy the *answer* quotes could differ from the copy the *citation* links to. Text in one store, keys everywhere else.

## 13. Deterministic point ids (uuid5) make re-indexing idempotent
- Problem: Qdrant ids must be int or UUID, but our join key is a readable string (`{accession}::{section}::{index}`); a random id per upsert silently duplicates every chunk on re-run.
- Choice: `point_id = uuid5(fixed_namespace, chunk_id)` — same chunk_id always maps to the same point, so upsert overwrites. The SQL write is `ON CONFLICT (chunk_id) DO UPDATE`, same property. Re-chunking a document explicitly deletes its old chunks in both stores first (`--rebuild`), because new params emit *different* chunk_ids that would otherwise coexist with the old ones.
- Alternative: hash to an int id (collision risk), or track an id counter (state to keep in sync).
- Interview q: "How do you re-index safely?" → idempotency is a property of the id function, not of the pipeline runner. Both smoke assertions double-upsert and assert the row/point counts didn't move.

## 14. BM25 persists a tokenized corpus, not a pickled BM25Okapi
- Problem: the sparse index needs to survive process restarts without recomputing from the DB every query.
- Choice: pickle plain lists — `chunk_ids`, token lists, metadata dicts — and rebuild `BM25Okapi` at load (sub-second at this corpus size). Tokenizer: lowercase, keep in-token `.`/`-` so `10-k`, `q3`, `1.5`, `u.s.` stay single terms, small hand-written stopword list, **no stemming**.
- Alternative: pickle the fitted BM25Okapi object (faster load, breaks on library upgrade — pickles of third-party classes are version-coupled); or use Postgres full-text search (real IDF + no separate artifact, but ties sparse retrieval to the Postgres path and gives up control of the tokenizer).
- Tradeoff on stemming: "impairment"/"impaired" stay distinct terms. Accepted deliberately — BM25 is in this system precisely for exact matches (tickers, "Blackwell", "CET1"), and the dense retriever already handles morphology. NLTK's default stopword list is also rejected: it strips "no"/"not", which invert meaning in risk-factor language ("no assurance").
- Interview q: "Why keep BM25 when embeddings are better?" → they aren't, on rare tokens. bge-small has never seen "GB200"; it lands somewhere generic in embedding space, while IDF weighting makes a rare token the strongest possible signal. Hybrid exists because the two fail on disjoint query types.

## 15. Filters pushed into the store, on both retrieval paths
- Problem: CLAUDE.md principle #4 (hard filters first, similarity second) is only true if the filter runs *inside* the search, not on its results. Post-filtering k results returns fewer than k — and sometimes zero — whenever the filter is selective.
- Choice: dense side uses a Qdrant `Filter` with explicit payload indexes on ticker/doc_type/section/fiscal_period/doc_id/chunk_id/filing_ts, so filtering is pushed into the HNSW traversal. Sparse side applies the same predicates to the score array *before* the top-k cut. Dates are stored twice — `filing_date` ISO string for display, `filing_ts` epoch integer for range filters — because integer `Range` behaves identically across Qdrant versions and the embedded local-mode client, while datetime-range support varies.
- Alternative: retrieve wide (k=200) and filter in Python. Works at 10k chunks, degrades exactly when the corpus grows.
- Interview q: "What breaks if you post-filter?" → recall silently, not loudly. Asking for 30 NVDA chunks and getting 4 looks like a sparse corpus, not a bug; that's why the Phase 2 smoke test asserts a filter both *keeps* the right hit and *excludes* the wrong one — an "everything returned matches the filter" assertion passes vacuously on zero results.

## 16. bge is asymmetric: separate passage and query encode paths
- Problem: bge-* models are trained with an instruction prefix on the query side only. Embedding a query bare costs real recall, and nothing errors — scores just quietly get worse.
- Choice: two functions in `src/common/embeddings.py` (`embed_passages`, `embed_query`), the prefix living in config and applied in exactly one place, rather than one function with a flag a caller can forget. Vectors are L2-normalized at encode time so Qdrant COSINE and raw dot product agree and downstream cosine math (dedup, clustering in Phase 5) needs no renormalization.
- Interview q: "How would you catch that regression?" → you don't catch it by reading scores; you catch it with the Phase 7 retrieval eval, which is why eval is not optional. The smoke test asserts normalization, but only recall@5 would reveal a missing prefix.
