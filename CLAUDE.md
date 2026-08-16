# Financial Research Copilot — Build Instructions

You are building a production-style financial RAG system over a weekend. The owner (Amar) is a data engineer targeting FDE/AI engineering roles. Every component exists to be defensible in an interview. Do not over-engineer. Do not add LangChain/LlamaIndex — direct Python only, so every step is explainable.

## Non-negotiable principles
1. Smallest system that demonstrates the FULL RAG lifecycle: ingest → parse → chunk → embed → hybrid retrieve → rerank → generate with citations → evaluate.
2. Structured numbers live in Postgres, never in the vector store. Vector search answers "why", SQL answers "how much".
3. Every chunk carries metadata: ticker, company, doc_type, filing_date, fiscal_period, section, source_url, chunk_id.
4. News queries use hard date/ticker SQL filters FIRST, vector similarity second.
5. Answers must cite sources or say "insufficient evidence". No unsupported claims.
6. After each phase, write a short DECISIONS.md entry: problem / choice / alternative / tradeoff / likely interview question.

## Stack (fixed — do not substitute)
- Python 3.11+, uv or pip
- Storage: SQLite (local mode, default) or PostgreSQL (docker mode) — via src/common/db.py
- Vectors: embedded Qdrant (local mode, default) or Qdrant server (docker mode) — HNSW ANN either way
- sentence-transformers: `BAAI/bge-small-en-v1.5` embeddings (384d, fast on Apple Silicon)
- rank_bm25 for sparse retrieval
- cross-encoder reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Ollama for generation (default `llama3.1:8b`; allow model override via config). Optional Anthropic API fallback flag.
- Streamlit for the demo UI (LAST, keep minimal)
- No Prefect in v1 — plain scripts with idempotent CLI entry points. Add Prefect flow wrappers only if time remains (owner already has Prefect infra; wrappers are a 30-min bonus, not core).

## Companies (v1)
NVDA, MSFT, JPM. (AMZN/META only if time remains — 3 tickers proves everything 5 does.)

## Phase plan — execute in order, commit after each phase

### Phase 0 — Environment (30 min)
- STORAGE: config.yaml `storage.mode` is "local" by default = SQLite + embedded Qdrant, NO DOCKER NEEDED.
  All DB/vector access goes through src/common/db.py (get_conn / get_qdrant). Never import
  psycopg or QdrantClient directly anywhere else — that seam is what makes both modes work.
  docker-compose.yml stays in the repo as the documented production deployment path.
- Install deps from pyproject.toml
- Run `scripts/init_db.py` to create schema (already stubbed — implement DDL per src/structured/schema.sql)
- Smoke test: `scripts/smoke_test.py` connects to both stores, embeds one sentence, upserts, searches.

### Phase 1 — SEC ingestion (2-3 hrs)
- `src/ingestion/edgar.py`: pull latest 10-K, last 4 10-Qs, recent 8-Ks per ticker via EDGAR full-text/submissions API. Respect SEC fair-access: User-Agent header with contact email, ≤10 req/s.
- Store raw filings in data/raw/{ticker}/{accession}.html + a manifest row in Postgres (documents table).
- `src/parsing/sec_parser.py`: strip HTML, detect sections (Item 1 Business, Item 1A Risk Factors, Item 7 MD&A, Item 8 Financial Statements) via regex on item headers. Output: list of (section, text).
- Earnings call transcripts: do NOT scrape paid providers. Use company IR press releases / prepared remarks PDFs where freely published, or Motley Fool transcripts are NOT licensed — skip. If no free transcript, the earnings press release (8-K EX-99) is the substitute. Note this limitation in README.

### Phase 2 — Chunking + embedding + indexing (2 hrs)
- `src/chunking/chunker.py`: section-aware chunking. Within a section: split on paragraphs, target 600-900 tokens, 100 token overlap, never cross section boundaries. Attach full metadata.
- Embed with bge-small, upsert to Qdrant collection `filings` with payload = metadata. HNSW defaults (m=16, ef_construct=100).
- Build BM25 index over the same chunks (`src/retrieval/sparse.py`), persist tokenized corpus to disk (pickle) keyed by chunk_id.

### Phase 3 — Retrieval stack (2-3 hrs)
- `src/retrieval/dense.py`: query embed → Qdrant search with optional payload filters (ticker, doc_type, date range, section).
- `src/retrieval/hybrid.py`: reciprocal rank fusion, k=60 constant, over dense top-30 + BM25 top-30.
- `src/retrieval/rerank.py`: cross-encoder over fused top-30 → top 6.
- Each retriever returns a common `RetrievedChunk` dataclass (src/common/models.py) so they're swappable in eval.

### Phase 4 — Structured financials (1-2 hrs)
- `src/structured/financials.py`: load quarterly revenue, EPS, operating income, gross margin, guidance text per ticker into Postgres. v1 source: hand-extract from the earnings releases already ingested OR use the SEC companyfacts XBRL API (preferred — free, structured JSON: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json). Map us-gaap tags: Revenues/RevenueFromContractWithCustomerExcludingAssessedTax, EarningsPerShareDiluted, OperatingIncomeLoss.
- `src/structured/queries.py`: canned parameterized queries (revenue last N quarters, QoQ/YoY growth, margin trend). LLM does NOT write raw SQL in v1 — router picks a canned query + params. (Text-to-SQL is a documented v2 item; say why: injection risk + eval complexity.)

### Phase 5 — News pipeline (2 hrs)
- `src/news/ingest.py`: RSS-first (Google News RSS per ticker query, company IR RSS, Yahoo Finance RSS). Store: ticker, headline, source, published_at, url, text (fetched + trafilatura-extracted), topic. Postgres table `news_articles`.
- Dedup: exact URL + near-dup via cosine sim of headline embeddings > 0.9 → keep earliest.
- Event clustering: embed headlines, agglomerative clustering (threshold ~0.75) within a date window → `event_id`.
- `src/news/brief.py`: weekly intelligence brief generator — SQL filter (ticker + last 7 days) → cluster → rank clusters by article count + source diversity → LLM summarizes each event with bull/bear implication → assembled brief with links and dates.

### Phase 6 — Router + generation (2 hrs)
- `src/router/router.py`: rule-first (regex for tickers, date phrases, numeric keywords like "revenue/EPS/margin/growth"), LLM fallback classifier returning one of: NEWS, FILING_RAG, NUMERIC, MIXED, CROSS_SOURCE. Log every routing decision.
- `src/generation/answer.py`: prompt template that (a) numbers every context chunk, (b) requires inline [n] citations, (c) instructs "if the context does not contain the answer, say insufficient evidence", (d) separates Facts vs Interpretation sections for research queries.
- Citation post-check: verify every [n] cited exists in context; strip/flag hallucinated citations.

### Phase 7 — Evaluation (2-3 hrs) — DO NOT SKIP
- `evals/questions.jsonl`: 40 questions across categories: semantic (10), exact-keyword (6), temporal (6), numeric (6), cross-document (6), insufficient-evidence (6). Each has expected_chunk_ids or expected_answer + category.
- `src/evals/retrieval_eval.py`: Recall@5, Precision@5, MRR for dense-only vs hybrid vs hybrid+rerank. Write results to Postgres + evals/results.md table.
- `src/evals/answer_eval.py`: LLM-as-judge (separate prompt) scoring groundedness, citation correctness, relevance 1-5; hallucination = any claim judge can't map to context. Save per-question JSON.
- `evals/failure_cases.md`: document ≥5 real failures found during eval (BM25-wins case, dense-wins case, stale-doc-outranks-fresh case, chunk-boundary case, syndicated-news-inflation case) with the fix applied.

### Phase 8 — Demo + README (1-2 hrs)
- `app.py`: Streamlit, one text box, shows: routing decision, retrieved chunks with metadata, reranker scores, final cited answer. Sidebar: ticker/date filters, retrieval-mode toggle (for live A/B in interviews).
- README: architecture diagram (mermaid), quickstart, eval results table, failure cases summary, "interview talking points" link to DECISIONS.md.

## Definition of done
`docker compose up`, `python scripts/ingest_all.py`, `streamlit run app.py`, then: "Give me all important NVDA news from last week" produces a clustered, deduped, cited brief; "How has revenue changed over the last four quarters and why" hits SQL + RAG and cites both.
