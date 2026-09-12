-- Structured store. Vectors live in Qdrant; everything relational lives here.

CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,          -- accession number or news url hash
    ticker        TEXT NOT NULL,
    company       TEXT NOT NULL,
    doc_type      TEXT NOT NULL,             -- 10-K | 10-Q | 8-K | press_release | news
    filing_date   DATE,
    fiscal_period TEXT,                      -- e.g. FY2025Q3
    source_url    TEXT,
    raw_path      TEXT,
    ingested_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT REFERENCES documents(doc_id),
    section     TEXT,                        -- Business | Risk Factors | MD&A | Financials | body
    chunk_index INT,
    n_tokens    INT,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS financials (
    ticker        TEXT NOT NULL,
    fiscal_year   INT  NOT NULL,
    fiscal_qtr    INT  NOT NULL,             -- 1-4, 0 = full year
    metric        TEXT NOT NULL,             -- revenue | eps_diluted | operating_income | gross_margin | segment:<name>
    value         NUMERIC,
    unit          TEXT,                      -- usd | usd_per_share | pct
    period_end    DATE,                      -- fiscal period end; 52/53-week filers don't end on month boundaries
    source_accn   TEXT,                      -- XBRL accession the value came from (always set)
    source_doc_id TEXT REFERENCES documents(doc_id),   -- set only when that filing was also ingested
    derived       BOOLEAN DEFAULT FALSE,     -- TRUE = computed (Q4 by subtraction, gross_margin as a ratio)
    PRIMARY KEY (ticker, fiscal_year, fiscal_qtr, metric)
);

CREATE TABLE IF NOT EXISTS guidance (
    ticker        TEXT NOT NULL,
    fiscal_period TEXT NOT NULL,             -- period guidance was GIVEN for
    given_on      DATE,
    text          TEXT NOT NULL,
    source_doc_id TEXT REFERENCES documents(doc_id),
    PRIMARY KEY (ticker, fiscal_period, given_on)
);

CREATE TABLE IF NOT EXISTS news_articles (
    article_id   TEXT PRIMARY KEY,           -- sha1(url)
    ticker       TEXT NOT NULL,
    headline     TEXT NOT NULL,
    source       TEXT,
    published_at TIMESTAMPTZ NOT NULL,
    url          TEXT NOT NULL,
    text         TEXT,
    topic        TEXT,
    event_id     TEXT,                       -- cluster assignment, nullable until clustered
    is_duplicate BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_news_ticker_date ON news_articles(ticker, published_at DESC);

CREATE TABLE IF NOT EXISTS routing_log (
    id          SERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ DEFAULT now(),
    query       TEXT,
    route       TEXT,                        -- NEWS | FILING_RAG | NUMERIC | MIXED | CROSS_SOURCE
    method      TEXT,                        -- rule | llm
    latency_ms  INT
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id      TEXT,
    question_id TEXT,
    category    TEXT,
    retriever   TEXT,                        -- dense | hybrid | hybrid_rerank
    recall_at_5 NUMERIC,
    mrr         NUMERIC,
    groundedness NUMERIC,
    citation_ok BOOLEAN,
    hallucinated BOOLEAN,
    ts          TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (run_id, question_id, retriever)
);
