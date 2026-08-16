"""Storage abstraction so local mode and docker mode share one code path.

Interview note: this is the seam that lets the same pipeline run embedded on a
laptop and containerized in prod. Everything downstream calls get_conn() /
get_qdrant() and never knows which backend it got.
"""
import sqlite3
from pathlib import Path
from src.common.config import CFG

ROOT = Path(__file__).parents[2]


def storage_mode() -> str:
    return CFG.get("storage", {}).get("mode", "local")


def get_conn():
    """Returns a DB-API connection. SQLite in local mode, Postgres in docker mode."""
    if storage_mode() == "local":
        path = ROOT / CFG["storage"]["sqlite_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn
    import psycopg
    return psycopg.connect(CFG["postgres"]["dsn"])


def get_qdrant():
    """Embedded Qdrant in local mode (same HNSW engine, no server), HTTP in docker mode."""
    from qdrant_client import QdrantClient
    if storage_mode() == "local":
        path = ROOT / CFG["storage"]["qdrant_path"]
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    return QdrantClient(url=CFG["qdrant"]["url"])


def schema_path() -> Path:
    name = "schema_sqlite.sql" if storage_mode() == "local" else "schema.sql"
    return ROOT / "src/structured" / name
