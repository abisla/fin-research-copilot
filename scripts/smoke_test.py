"""Phase 0 gate: postgres reachable, qdrant reachable, embedding model loads,
one vector round-trips. If this passes, start Phase 1."""
import psycopg
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from src.common.config import CFG

def main():
    with psycopg.connect(CFG["postgres"]["dsn"]) as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1
    print("postgres ok")

    qc = QdrantClient(url=CFG["qdrant"]["url"])
    model = SentenceTransformer(CFG["embedding"]["model"])
    vec = model.encode("NVIDIA data center revenue grew on AI demand").tolist()
    qc.recreate_collection("smoke", vectors_config=VectorParams(size=len(vec), distance=Distance.COSINE))
    qc.upsert("smoke", points=[PointStruct(id=1, vector=vec, payload={"t": "test"})])
    hit = qc.search("smoke", query_vector=vec, limit=1)[0]
    assert hit.score > 0.99
    qc.delete_collection("smoke")
    print(f"qdrant + embeddings ok (dim={len(vec)})")

if __name__ == "__main__":
    main()
