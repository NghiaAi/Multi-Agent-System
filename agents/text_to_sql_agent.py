import os
import re
import json
import tiktoken 
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.sql import SQLTools
from sqlalchemy import create_engine
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

db_path = BASE_DIR / "data/djia.db"
db_url = f"sqlite:///{db_path}"

engine = create_engine(db_url)

# MAX_CONTEXT_TOKENS = 14000         # tổng context mà bạn dùng để tính toán
# BASE_PROMPT_TOKENS = 2367          # system + instructions + tool spec + v.v.
# TOKENS_PER_ROW = 48                # token trung bình cho 1 dòng kết quả

# encoding = tiktoken.get_encoding("cl100k_base")

# def count_tokens(text: str) -> int:
#     if not text:
#         return 0
#     return len(encoding.encode(text))

# class UnlimitedSQLTools(SQLTools):
#     def run_sql_query(self, query: str, limit: Optional[int] = None) -> str:
#         try:
#             if limit is None:
#                 query_tokens = count_tokens(query)
#                 available_for_result = MAX_CONTEXT_TOKENS - BASE_PROMPT_TOKENS - query_tokens
#                 if available_for_result <= 0:
#                     max_rows = 1
#                 else:
#                     est_rows = available_for_result // TOKENS_PER_ROW
#                     max_rows = max(1, est_rows)
#             else:
#                 max_rows = limit

#             result = self.run_sql(sql=query, limit=max_rows)
#             return json.dumps(result, default=str)
#         except Exception as e:
#             return json.dumps({"error": f"Error running query: {e}"})
# sql_tools = UnlimitedSQLTools(db_url=db_url)

MAX_CONTEXT_TOKENS = 14000         # tổng context mà bạn dùng để tính toán
BASE_PROMPT_TOKENS = 2367          # system + instructions + tool spec + v.v.
TOKENS_PER_ROW = 48                # token trung bình cho 1 dòng kết quả

encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(encoding.encode(text))

class UnlimitedSQLTools(SQLTools):
    def run_sql_query(self, query: str, limit: Optional[int] = None) -> str:
        try:
            # Tính max_rows theo token nếu agent không truyền limit
            if limit is None:
                query_tokens = count_tokens(query)
                available_for_result = MAX_CONTEXT_TOKENS - BASE_PROMPT_TOKENS - query_tokens
                if available_for_result <= 0:
                    max_rows = 1
                else:
                    est_rows = available_for_result // TOKENS_PER_ROW
                    max_rows = max(1, est_rows)
            else:
                max_rows = limit

            # 1) Lấy FULL kết quả từ SQL (không giới hạn ở DB)
            full_result = self.run_sql(sql=query, limit=None)

            # 2) Nếu không phải list hoặc số dòng nhỏ hơn max_rows thì trả về luôn
            if not isinstance(full_result, list) or len(full_result) <= max_rows:
                result = full_result
            else:
                # 3) Nếu là list[dict] và có cột Date thì sort theo Date rồi lấy các dòng mới nhất
                if full_result and isinstance(full_result[0], dict) and "Date" in full_result[0]:
                    sorted_result = sorted(full_result, key=lambda r: r.get("Date") or "")
                    result = sorted_result[-max_rows:]
                else:
                    # Không có Date: đơn giản lấy các phần tử cuối cùng
                    result = full_result[-max_rows:]

            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": f"Error running query: {e}"})
sql_tools = UnlimitedSQLTools(db_url=db_url)



# class UnlimitedSQLTools(SQLTools):
#     def run_sql_query(self, query: str, limit: Optional[int] = None) -> str:
#         try:
#             result = self.run_sql(sql=query, limit=limit)
#             return json.dumps(result, default=str)
#         except Exception as e:
#             return f"Error running query: {e}"

# sql_tools = UnlimitedSQLTools(db_url=db_url)

company_to_symbol = {
    "Apple Inc.": "AAPL",
    "Amgen Inc.": "AMGN",
    "American Express Company": "AXP",
    "Boeing Company (The)": "BA",
    "Caterpillar, Inc.": "CAT",
    "Salesforce, Inc.": "CRM",
    "Cisco Systems, Inc.": "CSCO",
    "Chevron Corporation": "CVX",
    "Walt Disney Company (The)": "DIS",
    "Dow Inc.": "DOW",
    "Goldman Sachs Group, Inc. (The)": "GS",
    "Home Depot, Inc. (The)": "HD",
    "Honeywell International Inc.": "HON",
    "International Business Machines": "IBM",
    "Intel Corporation": "INTC",
    "Johnson & Johnson": "JNJ",
    "JP Morgan Chase & Co.": "JPM",
    "Coca-Cola Company (The)": "KO",
    "McDonald's Corporation": "MCD",
    "3M Company": "MMM",
    "Merck & Company, Inc.": "MRK",
    "Microsoft Corporation": "MSFT",
    "Nike, Inc.": "NKE",
    "Procter & Gamble Company (The)": "PG",
    "The Travelers Companies, Inc.": "TRV",
    "UnitedHealth Group Incorporated": "UNH",
    "Visa Inc.": "V",
    "Verizon Communications Inc.": "VZ",
    "Walgreens Boots Alliance, Inc.": "WBA",
    "Walmart Inc.": "WMT"
}

sql_tool_schema = {
    "type": "function",
    "function": {
        "name": "run_sql_query",
        "description": "Execute a SQL query on the DJIA database and return the result as a JSON string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The SQL query to execute."},
                "limit": {"type": "integer", "description": "Optional limit for the number of rows to return.", "default": None}
            },
            "required": ["query"]
        }
    }
}

agent = Agent(
    model=Groq(
        id="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        timeout=30,
        max_retries=5,
        temperature=0.2,
        max_tokens=4000,
        top_p=0.8,
    ),
    # tools=[sql_tools, sql_tool_schema],
    tools=[sql_tools],
    description="You are a financial data analyst agent that queries a SQL database containing DJIA companies and historical stock prices to answer questions about stock prices, returns, and trading volumes.",
    instructions=[
        "You are a helpful research assistant with access to a SQL database containing two tables: 'companies' and 'prices'. Below are their schemas, a company-to-symbol mapping, and dataset description.",
        "### Dataset Description",
        "- The database contains data on companies in the Dow Jones Industrial Average (DJIA) and their historical stock prices.",
        "- The database path is './data/djia.db'.",
        "- The 'companies' table stores company information, and the 'prices' table stores daily stock price data.",
        "- The 'symbol' column in 'companies' links to the 'Ticker' column in 'prices'.",
        "### Schema",
        "**Table: companies**",
        "- symbol: TEXT (e.g., 'AAPL', primary key, stock ticker)",
        "- name: TEXT (e.g., 'Apple Inc.', company name)",
        "- sector: TEXT (e.g., 'Technology', company sector)",
        "- industry: TEXT (e.g., 'Consumer Electronics', detailed industry)",
        "- country: TEXT (e.g., 'United States', company country)",
        "- website: TEXT (e.g., 'www.apple.com', company website)",
        "- market_cap: REAL (e.g., 3143825096704, market capitalization)",
        "- pe_ratio: REAL (e.g., 33.219048, price-to-earnings ratio)",
        "- dividend_yield: REAL (e.g., 48.0, dividend yield)",
        "- 52_week_high: REAL (e.g., 260.1, 52-week high price)",
        "- 52_week_low: REAL (e.g., 169.11, 52-week low price)",
        "- description: TEXT (company description)",
        "**Table: prices**",
        "- Date: DATETIME (e.g., '2023-04-26 00:00:00-04:00', date of the price data)",
        "- Open: REAL (e.g., 161.43199188421733, opening price)",
        "- High: REAL (e.g., 163.6298284131595, high price)",
        "- Low: REAL (e.g., 161.1745931859106, low price)",
        "- Close: REAL (e.g., 162.125, closing price)",
        "- Adj Close: REAL (e.g., 162.125, adjusted closing price)",
        "- Volume: INTEGER (e.g., 45498800, trading volume)",
        "- Dividends: REAL (e.g., 0.0, dividends paid)",
        "- Stock Splits: REAL (e.g., 0.0, stock splits)",
        "- Ticker: TEXT (e.g., 'AAPL', foreign key to companies.symbol)",
        "### Company-to-Symbol Mapping",
        f"Use this mapping to convert company names to tickers for querying the 'prices' table, or to map tickers back to company names in results: {json.dumps(company_to_symbol, ensure_ascii=False, indent=2)}",
        "### Instructions for Queries",
        "- Always use the company-to-symbol mapping to convert company names to tickers (e.g., 'UnitedHealth Group' → 'UNH') for querying the 'prices' table. Match flexibly by removing suffixes like 'Inc.', 'Company', '(The)', and use case-insensitive substring matching. If ambiguous or no match, note it and return 'I don't know'.",
        "- For queries involving all DJIA companies (e.g., total dividends per company), query the 'prices' table, group by Ticker, and calculate aggregates (e.g., SUM(Dividends)). In the Raw Result, map each Ticker to its company name using the company-to-symbol mapping, and include ALL companies (30) even if SUM=0 or no data (set Total_Dividends=0 for those).",
        "- Only query the 'companies' table if static info (e.g., market_cap, sector) is explicitly needed.",
        "- For visualization queries requiring time series (e.g., cumulative returns), select individual values (e.g., Date, Close, Ticker) from the 'prices' table, ordered by Date.",
        "- For any query asking for pie chart / distribution / breakdown of DJIA companies by sector. Use Count(*) do not use DISTINCT.",
        "- For cumulative returns queries (e.g., 'cumulative returns of UNH in 2024'), query daily closing prices for the specified ticker and year: SELECT Date, Close, Ticker FROM prices WHERE Ticker = 'UNH' AND STRFTIME('%Y', Date) = '2024' ORDER BY Date.",
        "- For dividend queries specifying a date (e.g., 'dividend per share of MSFT on May 17, 2023'), query the 'Dividends' column from the 'prices' table using the DATE() function to match the date: SELECT Dividends FROM prices WHERE Ticker = '[TICKER]' AND DATE(Date) = '[YYYY-MM-DD]'.",
        "- For historical price queries (e.g., 'last 90 trading days of AAPL'), always select Date, Open, High, Low, Close, Adj Close, Volume, Ticker. ",
        "- When the user asks for a 'boxplot of market capitalization values grouped by sector' or 'boxplot market cap by sector' ALWAYS query individual company data (Not aggregates) from the 'companies' table: SELECT sector, market_cap FROM companies ORDER BY sector, market_cap DESC;",
        "- To call the 'run_sql_query' tool, ALWAYS respond with a JSON array of tool calls in this exact format: [{'id': 'call_id', 'type': 'function', 'function': {'name': 'run_sql_query', 'arguments': '{\"query\": \"YOUR SQL QUERY HERE\"}'}}]. Replace 'call_id' with a unique ID like 'call_123'.",
        "- When calling run_sql_query, pass the SQL query as a JSON object with 'query' key.",
        "- After receiving the result from run_sql_query, process it to extract the relevant value.",
        "- If the query result is empty, return 'I don't know. No data available for [company or data type].'.",
        "- For price values, format to 2 decimal places (e.g., $123.45). For returns, format as percentage to 2 decimal places (e.g., 5.23%).",
        
        "- Example: For 'Cumulative returns of UnitedHealth Group (UNH) in 2024':",
        "  SQL: SELECT Date, Close, Ticker FROM prices WHERE Ticker = 'UNH' AND STRFTIME('%Y', Date) = '2024' ORDER BY Date",
        "  Raw Result: [{'Date': '2024-01-01 00:00:00-05:00', 'Close': 526.47, 'Ticker': 'UNH'}, ...]",
        "- If no date is specified, assume the latest available data or summarize as needed.",
        "- If the question is about a company not in the mapping or data not available, say 'I don't know.'",
        "- Present answers clearly, e.g., 'The cumulative return of UNH from 2024-01-01 to 2024-12-31 was X.XX%.'",
        "- After receiving the tool result, ALWAYS provide a final response in this format:\nSQL Query: [the SQL query you used]\nRaw Result: [the raw result with company names mapped]\nAnswer: [your formatted answer based on the result].",
        "- If the tool call fails or returns no data, return 'I don't know. No data available for [company or data type].'.",
    ],
    debug_mode=True,
    max_tool_calls=5,
    tool_call_strategy="auto",
)