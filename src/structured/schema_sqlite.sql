-- SQLite variant of schema.sql for local (no-Docker) mode.
-- Differences from Postgres: SERIAL -> INTEGER PRIMARY KEY AUTOINCREMENT,
-- TIMESTAMPTZ -> TEXT (ISO8601), BOOLEAN -> INTEGER. Nothing in v1 needs
-- Postgres-specific features, so this is a drop-in.

CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    ticker        TEXT NOT NULL,
    company       TEXT NOT NULL,
    doc_type      TEXT NOT NULL,
    filing_date   TEXT,
    fiscal_period TEXT,
    source_url    TEXT,
    raw_path      TEXT,
    ingested_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT REFERENCES documents(doc_id),
    section     TEXT,
    chunk_index INTEGER,
    n_tokens    INTEGER,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS financials (
    ticker        TEXT NOT NULL,
    fiscal_year   INTEGER NOT NULL,
    fiscal_qtr    INTEGER NOT NULL,
    metric        TEXT NOT NULL,
    value         REAL,
    unit          TEXT,
    period_end    TEXT,
    source_accn   TEXT,
    source_doc_id TEXT REFERENCES documents(doc_id),
    derived       INTEGER DEFAULT 0,
    PRIMARY KEY (ticker, fiscal_year, fiscal_qtr, metric)
);

CREATE TABLE IF NOT EXISTS guidance (
    ticker        TEXT NOT NULL,
    fiscal_period TEXT NOT NULL,
    given_on      TEXT,
    text          TEXT NOT NULL,
    source_doc_id TEXT REFERENCES documents(doc_id),
    PRIMARY KEY (ticker, fiscal_period, given_on)
);

CREATE TABLE IF NOT EXISTS news_articles (
    article_id   TEXT PRIMARY KEY,
    ticker       TEXT NOT NULL,
    headline     TEXT NOT NULL,
    source       TEXT,
    published_at TEXT NOT NULL,
    url          TEXT NOT NULL,
    text         TEXT,
    topic        TEXT,
    event_id     TEXT,
    is_duplicate INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_news_ticker_date ON news_articles(ticker, published_at DESC);

CREATE TABLE IF NOT EXISTS routing_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT DEFAULT (datetime('now')),
    query       TEXT,
    route       TEXT,
    method      TEXT,
    latency_ms  INTEGER
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id       TEXT,
    question_id  TEXT,
    category     TEXT,
    retriever    TEXT,
    recall_at_5  REAL,
    precision_at_5 REAL,
    hit_at_5     INTEGER,
    mrr          REAL,
    groundedness REAL,
    citation_ok  INTEGER,
    hallucinated INTEGER,
    ts           TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (run_id, question_id, retriever)
);
