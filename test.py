import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# DB_PATH = BASE_DIR / "data" / "djia.db"

# conn = sqlite3.connect(DB_PATH)
# # Check AAPL data
# df = pd.read_sql_query(
#     "SELECT Close FROM prices WHERE Ticker = 'MSFT' AND DATE(Date) BETWEEN '2024-06-01' AND '2024-09-30'",
#     conn
# )
# print(df)


# conn.close()

from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30
)
print(client.get_collections())