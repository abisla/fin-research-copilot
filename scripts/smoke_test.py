"""Phase 0 gate: DB reachable, vector store reachable, embedding model loads,
one vector round-trips. If this passes, start Phase 1."""
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from src.common.config import CFG
from src.common.db import get_conn, get_qdrant, storage_mode


def main():
    print(f"storage mode: {storage_mode()}")

    conn = get_conn()
    assert conn.execute("SELECT 1").fetchone()[0] == 1
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
        if storage_mode() == "local" else
        "SELECT tablename AS name FROM pg_tables WHERE schemaname='public'"
    ).fetchall()
    conn.close()
    print(f"db ok — {len(tables)} tables")

    qc = get_qdrant()
    model = SentenceTransformer(CFG["embedding"]["model"])
    vec = model.encode("NVIDIA data center revenue grew on AI demand").tolist()
    if qc.collection_exists("smoke"):
        qc.delete_collection("smoke")
    qc.create_collection("smoke", vectors_config=VectorParams(size=len(vec), distance=Distance.COSINE))
    qc.upsert("smoke", points=[PointStruct(id=1, vector=vec, payload={"t": "test"})])
    hit = qc.query_points("smoke", query=vec, limit=1).points[0]
    assert hit.score > 0.99, f"expected ~1.0 self-similarity, got {hit.score}"
    qc.delete_collection("smoke")
    print(f"vector store + embeddings ok (dim={len(vec)}, self-sim={hit.score:.4f})")
    print("\nPHASE 0 PASSED — proceed to Phase 1")


if __name__ == "__main__":
    main()
