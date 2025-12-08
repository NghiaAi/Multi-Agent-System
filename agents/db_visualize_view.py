import sqlite3

import pandas as pd

DB_PATH = "data/djia.db"


def get_tables():
    """Return list of table names in djia.db."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]
    conn.close()
    return tables


def preview_table(table_name: str, limit: int = 20) -> pd.DataFrame:
    """Return first `limit` rows of a table as a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT * FROM {table_name} LIMIT {limit};"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


if __name__ == "__main__":
    print(f"Using database: {DB_PATH}")
    tables = get_tables()
    print("Tables:", tables)

    for t in tables:
        print(f"\n=== Preview of table: {t} ===")
        try:
            df = preview_table(t, limit=5)
            print(df)
        except Exception as e:
            print(f"Error reading table {t}: {e}")


