"""Phase 2 gate: exercise the indexing path on synthetic chunks.

The chunker is intentionally unimplemented (owner writes it), so this proves out
everything downstream of it — SQL upsert, embedding, Qdrant payload/filter/search,
BM25 build + filtered search, and the SQL text hydration both retrievers rely on —
against the real stores, using hand-built Chunk objects.

It writes to a throwaway collection and BM25 path, borrows a real doc_id from the
`documents` manifest (chunks.doc_id is a FK), and deletes everything it wrote.
Assertions are behavioural, not just "no exception": the semantic query must beat
the lexical distractor on the dense side, the rare-token query must win on BM25,
and filters must actually exclude.
"""
from datetime import date
from pathlib import Path

from src.common.db import get_conn, get_qdrant
from src.common.embeddings import embed_passages, embed_query
from src.common.models import Chunk, ChunkMeta
from src.indexing import store, vectors
from src.retrieval import sparse

ROOT = Path(__file__).parents[1]
COLLECTION = "smoke_filings"
BM25_PATH = ROOT / "data" / "smoke_bm25.pkl"

FIXTURES = [
    ("risk", "Risk Factors", "10-K",
     "Our business could be harmed if we are unable to obtain sufficient foundry "
     "capacity from third-party manufacturers. We depend on a limited number of "
     "suppliers and any disruption in wafer supply would materially affect our "
     "ability to meet customer demand."),
    ("mdna", "MD&A", "10-K",
     "Data Center revenue increased primarily due to higher shipments of our "
     "Blackwell architecture accelerators to hyperscale customers, partially offset "
     "by lower gaming demand."),
    ("liq", "MD&A", "10-Q",
     "Cash, cash equivalents and marketable securities totaled a higher balance at "
     "quarter end, and we returned capital to shareholders through share repurchases "
     "and cash dividends."),
    ("gov", "Business", "10-K",
     "We compete in markets characterized by rapid technological change and evolving "
     "industry standards, and our competitors include large semiconductor companies "
     "with greater resources."),
]


def build_fixtures(doc: dict) -> list[Chunk]:
    chunks = []
    for i, (key, section, doc_type, text) in enumerate(FIXTURES):
        meta = ChunkMeta(
            chunk_id=f"SMOKE::{key}",
            doc_id=doc["doc_id"],
            ticker=doc["ticker"],
            company=doc["company"],
            doc_type=doc_type,
            filing_date=date(2025, 2, 26) if doc_type == "10-K" else date(2025, 8, 27),
            fiscal_period="FY2025" if doc_type == "10-K" else "FY2025Q2",
            section=section,
            source_url=doc["source_url"],
        )
        chunks.append(Chunk(meta=meta, text=text, chunk_index=i, n_tokens=len(text.split())))
    return chunks


def cleanup(conn, client):
    cur = conn.cursor()
    cur.execute(f"DELETE FROM chunks WHERE chunk_id LIKE {store.placeholder()}", ("SMOKE::%",))
    conn.commit()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    BM25_PATH.unlink(missing_ok=True)


def main():
    conn = get_conn()
    client = get_qdrant()
    try:
        docs = store.documents(conn)
        assert docs, "no rows in `documents` — run scripts/ingest_filings.py first (Phase 1)"
        doc = docs[-1]
        cleanup(conn, client)

        chunks = build_fixtures(doc)

        # --- SQL round-trip -------------------------------------------------
        store.write_chunks(conn, chunks)
        store.write_chunks(conn, chunks)          # upsert must not duplicate
        rows = store.load_chunk_rows(conn, [c.meta.chunk_id for c in chunks])
        assert len(rows) == len(chunks), f"expected {len(chunks)} rows, got {len(rows)}"
        meta = store.row_to_meta(rows[0])
        assert meta.ticker == doc["ticker"] and isinstance(meta.filing_date, date)
        print(f"sql ok - {len(rows)} chunks upserted twice, still {len(rows)} rows; "
              f"metadata joined from documents ({meta.ticker}, {meta.doc_type})")

        # --- embed + Qdrant -------------------------------------------------
        vectors.ensure_collection(client, COLLECTION, recreate=True)
        vecs = embed_passages([c.text for c in chunks])
        assert vecs.shape == (len(chunks), 384), vecs.shape
        assert abs(float((vecs[0] ** 2).sum()) - 1.0) < 1e-4, "vectors must be L2-normalized"
        vectors.upsert_chunks(client, chunks, vecs, name=COLLECTION)
        vectors.upsert_chunks(client, chunks, vecs, name=COLLECTION)   # deterministic ids
        n_points = client.get_collection(COLLECTION).points_count
        assert n_points == len(chunks), f"re-upsert duplicated points: {n_points}"
        print(f"qdrant ok - {n_points} points after double upsert (uuid5 ids are stable)")

        # dense search: paraphrase with no shared content words must still rank first
        q = "what happens if chip manufacturing partners cannot supply enough capacity"
        hits = client.query_points(COLLECTION, query=embed_query(q), limit=4).points
        top = hits[0].payload
        assert top["chunk_id"] == "SMOKE::risk", f"dense top-1 was {top['chunk_id']}"
        assert top["section"] == "Risk Factors" and top["ticker"] == doc["ticker"]
        assert top["filing_ts"] and top["filing_date"] == "2025-02-26"
        print(f"dense ok - top1={top['chunk_id']} score={hits[0].score:.3f} "
              f"payload carries {len(top)} metadata fields")

        # payload filter: restricting to 10-Q must exclude every 10-K chunk
        filtered = client.query_points(
            COLLECTION, query=embed_query(q), limit=4,
            query_filter=vectors.build_filter(doc_types=["10-Q"]),
        ).points
        assert [h.payload["chunk_id"] for h in filtered] == ["SMOKE::liq"], filtered
        dated = client.query_points(
            COLLECTION, query=embed_query(q), limit=4,
            query_filter=vectors.build_filter(date_from=date(2025, 6, 1)),
        ).points
        assert all(h.payload["doc_type"] == "10-Q" for h in dated), dated
        print(f"filters ok - doc_type filter kept {len(filtered)}/4, "
              f"filing_ts>=2025-06-01 kept {len(dated)}/4")

        # --- BM25 -----------------------------------------------------------
        n = sparse.build_index(chunks, path=BM25_PATH)
        idx = sparse.load_index(BM25_PATH)
        assert len(idx) == n == len(chunks)
        lex = idx.search("Blackwell", top_k=3, conn=conn)
        assert lex and lex[0].meta.chunk_id == "SMOKE::mdna", lex
        assert lex[0].text.startswith("Data Center revenue"), "text not hydrated from SQL"
        assert lex[0].retriever == "bm25"
        print(f"bm25 ok - 'Blackwell' -> {lex[0].meta.chunk_id} score={lex[0].score:.3f}, "
              f"text hydrated from SQL ({len(lex[0].text)} chars)")

        # filters must both keep the right hits and exclude the wrong ones; asserting
        # only "everything returned matches the filter" passes vacuously on 0 hits
        kept = idx.search("foundry capacity supply", top_k=5, conn=conn, sections=["Risk Factors"])
        assert [r.meta.chunk_id for r in kept] == ["SMOKE::risk"], kept
        excluded = idx.search("Blackwell", top_k=5, conn=conn, sections=["Risk Factors"])
        assert excluded == [], f"section filter let a non-Risk-Factors chunk through: {excluded}"
        assert idx.search("Blackwell", top_k=5, conn=conn, tickers=["ZZZZ"]) == []
        print(f"bm25 filters ok - section filter kept {len(kept)} matching hit(s), "
              f"excluded the MD&A-only hit; unknown ticker -> 0")

        print("\nPHASE 2 INDEXING PATH PASSED - implement src/chunking/chunker.py, "
              "then run scripts/build_index.py")
    finally:
        cleanup(conn, client)
        conn.close()


if __name__ == "__main__":
    main()
