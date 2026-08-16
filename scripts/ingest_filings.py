"""Phase 1 CLI: pull SEC filings for every ticker in config.yaml.
Idempotent — safe to re-run; already-ingested accessions are skipped."""
from src.ingestion.edgar import ingest_all


def main():
    total = ingest_all()
    print(f"\ndone — {total} new filings ingested")


if __name__ == "__main__":
    main()
