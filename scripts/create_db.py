import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, Table, Column, Integer, Float, Text, DateTime, MetaData, text

BASE_DIR = Path(__file__).resolve().parent.parent

# Định nghĩa schema
metadata = MetaData()

companies = Table(
    'companies', metadata,
    Column('symbol', Text, primary_key=True),
    Column('name', Text),
    Column('sector', Text),
    Column('industry', Text),
    Column('country', Text),
    Column('website', Text),
    Column('market_cap', Float),
    Column('pe_ratio', Float),
    Column('dividend_yield', Float),
    Column('52_week_high', Float),
    Column('52_week_low', Float),
    Column('description', Text)
)

prices = Table(
    'prices', metadata,
    Column('Date', DateTime),
    Column('Open', Float),
    Column('High', Float),
    Column('Low', Float),
    Column('Close', Float),
    Column('Volume', Integer),
    Column('Dividends', Float),
    Column('Stock Splits', Float),
    Column('Ticker', Text)
)
data_dir = BASE_DIR / "data"
# Đọc file CSV
companies_df = pd.read_csv(data_dir/"djia_companies_20250426.csv")
prices_df = pd.read_csv(data_dir/"djia_prices_20250426.csv")

# Chuyển đổi cột Date sang định dạng DATETIME
prices_df['Date'] = pd.to_datetime(prices_df['Date'], utc=True)

# Tạo engine SQLite
engine = create_engine(f"sqlite:///{data_dir/'djia.db'}")

# Tạo bảng với schema
metadata.create_all(engine)

# Lưu dữ liệu vào database
companies_df.to_sql("companies", engine, index=False, if_exists="append")
prices_df.to_sql("prices", engine, index=False, if_exists="append")

# Kiểm tra dữ liệu mẫu
with engine.connect() as connection:
    query = text("SELECT Date, Ticker, High FROM prices WHERE Ticker = 'AAPL' AND DATE(Date) = '2025-02-28'")
    result = connection.execute(query).fetchall()
    print("Sample data for AAPL on 2025-02-28:")
    for row in result:
        print(row)

print("Database created successfully with defined schema!")

