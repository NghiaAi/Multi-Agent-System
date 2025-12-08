import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "djia.db"

conn = sqlite3.connect(DB_PATH)
# Check AAPL data
df = pd.read_sql_query(
    """
        SELECT Dividends FROM prices WHERE Ticker = 'UNH' AND DATE(Date) = '2024-12-31'
    """,
    conn
)
print(df)


conn.close()

