"""Create Postgres schema. Run after docker compose up."""
from pathlib import Path
import psycopg
from src.common.config import CFG

def main():
    ddl = (Path(__file__).parents[1] / "src/structured/schema.sql").read_text()
    with psycopg.connect(CFG["postgres"]["dsn"]) as conn:
        conn.execute(ddl)
        conn.commit()
    print("schema created")

if __name__ == "__main__":
    main()
