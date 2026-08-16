"""Create the schema. Works in either storage mode (see config.yaml -> storage.mode)."""
from src.common.db import get_conn, schema_path, storage_mode


def main():
    ddl = schema_path().read_text()
    conn = get_conn()
    if storage_mode() == "local":
        conn.executescript(ddl)
    else:
        conn.execute(ddl)
    conn.commit()
    conn.close()
    print(f"schema created (mode={storage_mode()})")


if __name__ == "__main__":
    main()
