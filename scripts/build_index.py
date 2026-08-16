"""Phase 2 CLI: parse -> chunk -> persist -> embed -> Qdrant -> BM25.

Idempotent. A document already present in `chunks` is skipped, so re-running after
ingesting new filings only pays for the new ones. `--rebuild` drops and redoes a
document's chunks (both stores) — use it after changing chunking params, otherwise
old chunks linger with ids the new run never emits.

    python scripts/build_index.py                 # index everything not yet indexed
    python scripts/build_index.py --ticker NVDA   # one ticker
    python scripts/build_index.py --rebuild       # re-chunk and re-embed everything
    python scripts/build_index.py --bm25-only     # rebuild the sparse index from SQL
"""
import argparse
from pathlib import Path

from src.chunking.chunker import chunk_document
from src.common.config import CFG
from src.common.db import get_conn, get_qdrant
from src.common.embeddings import embed_passages
from src.indexing import store, vectors
from src.parsing.sec_parser import extract_sections
from src.retrieval import sparse

ROOT = Path(__file__).parents[1]


def index_document(conn, client, doc: dict, rebuild: bool) -> int:
    """Chunk one filing and write it to both stores. Returns chunks written."""
    existing = store.count_doc_chunks(conn, doc["doc_id"])
    if existing and not rebuild:
        print(f"  skip {doc['ticker']:5s} {doc['doc_type']:5s} {doc['doc_id']} "
              f"({existing} chunks already indexed)")
        return 0
    if existing:
        store.delete_doc_chunks(conn, doc["doc_id"])
        vectors.delete_doc(client, doc["doc_id"])

    raw = ROOT / doc["raw_path"]
    if not raw.exists():
        print(f"  MISS {doc['ticker']:5s} {doc['doc_type']:5s} {doc['doc_id']} - {raw} not on disk")
        return 0

    sections = extract_sections(raw.read_text(encoding="utf-8", errors="ignore"), doc["doc_type"])
    chunks = chunk_document(doc, sections)
    if not chunks:
        print(f"  EMPTY {doc['ticker']} {doc['doc_id']} - parsed {len(sections)} sections, 0 chunks")
        return 0

    store.write_chunks(conn, chunks)
    vecs = embed_passages([c.text for c in chunks], show_progress=False)
    vectors.upsert_chunks(client, chunks, vecs)

    by_section = {}
    for c in chunks:
        by_section[c.meta.section] = by_section.get(c.meta.section, 0) + 1
    print(f"  ok   {doc['ticker']:5s} {doc['doc_type']:5s} {doc['doc_id']} "
          f"{len(chunks):4d} chunks {by_section}")
    return len(chunks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", action="append", help="limit to ticker(s); repeatable")
    ap.add_argument("--rebuild", action="store_true", help="re-chunk documents already indexed")
    ap.add_argument("--bm25-only", action="store_true", help="only rebuild the BM25 index")
    args = ap.parse_args()

    conn = get_conn()
    client = get_qdrant()
    try:
        if not args.bm25_only:
            name = vectors.ensure_collection(client)
            docs = store.documents(conn, args.ticker)
            print(f"{len(docs)} documents in manifest -> collection '{name}'")
            total = sum(index_document(conn, client, d, args.rebuild) for d in docs)
            print(f"\n{total} chunks embedded ({CFG['embedding']['model']}, "
                  f"dim={CFG['embedding']['dim']})")

        n = sparse.build_index()
        print(f"BM25 index: {n} chunks -> {sparse.index_path().relative_to(ROOT)}")
        info = client.get_collection(vectors.collection_name())
        print(f"Qdrant '{vectors.collection_name()}': {info.points_count} points")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
