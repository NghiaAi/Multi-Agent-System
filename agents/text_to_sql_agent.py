import os
import re
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.sql import SQLTools
from sqlalchemy import create_engine
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Hàm REGEXP tùy chỉnh
def regexp(expr, item):
    reg = re.compile(expr, re.IGNORECASE)
    return reg.search(item) is not None

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

db_path = BASE_DIR / "data/djia.db"
db_url = f"sqlite:///{db_path}"

engine = create_engine(db_url)
conn = engine.connect()
conn.connection.create_function("REGEXP", 2, regexp)

sql_tools = SQLTools(db_url=db_url)

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

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[sql_tools],
    description="You are a financial data analyst agent that queries a SQL database containing DJIA companies and historical stock prices to answer questions about stock prices and trading volumes.",
    instructions=[
        "You are a helpful research assistant with access to a SQL database containing two tables: 'companies' and 'prices'. Below are their schemas, a company-to-symbol mapping, and dataset description.",
        "### Dataset Description",
        "- The database contains data on companies in the Dow Jones Industrial Average (DJIA) and their historical stock prices.",
        "- The database path is './db/djia.db'.",
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
        "- market_cap: FLOAT (market capitalization in USD)",
        "- pe_ratio: FLOAT (price-to-earnings ratio)",
        "- dividend_yield: FLOAT (dividend yield percentage)",
        "- 52_week_high: FLOAT (highest stock price in the last 52 weeks)",
        "- 52_week_low: FLOAT (lowest stock price in the last 52 weeks)",
        "- description: TEXT (company description)",
        "**Table: prices**",
        "- Date: DATETIME (e.g., '2025-02-28 00:00:00-04:00', trading date with time and timezone)",
        "- Open: FLOAT (opening price of the stock)",
        "- High: FLOAT (highest price of the stock during the day)",
        "- Low: FLOAT (lowest price of the stock during the day)",
        "- Close: FLOAT (closing price of the stock)",
        "- Volume: INTEGER (number of shares traded)",
        "- Dividends: FLOAT (dividends paid, if any)",
        "- Stock Splits: FLOAT (stock split ratio, if any)",
        "- Ticker: TEXT (e.g., 'AAPL', links to 'symbol' in 'companies')",
        "### Company-to-Symbol Mapping",
        str(company_to_symbol),
        "### Instructions",
        "- Use SQLTools to query the database. Always build safe and accurate SQL queries.",
        "- If the question mentions a company name (e.g., 'Apple' or '3M'), use the provided company-to-symbol mapping to find the corresponding symbol (e.g., 'Apple Inc.' → 'AAPL').",
        "- To match company names flexibly, remove common suffixes like 'Company', 'Inc.', 'Inc', 'Corporation', or '(The)' from the company name in the mapping and perform a case-insensitive substring match with the user's input. For example, if the user inputs 'Apple', match it to 'Apple Inc.' in the mapping to get 'AAPL'.",
        "- If multiple company names match (e.g., 'Disney' matching multiple entries), return 'I don't know' and ask for clarification.",
        "- If no match is found in the mapping, return 'I don't know'.",
        "- Use the found symbol as Ticker to query the 'prices' table directly. Do NOT query the 'companies' table unless explicitly needed.",
        "- To call the 'run_sql_query' tool, ALWAYS respond with a JSON array of tool calls in this exact format: [{'id': 'call_id', 'type': 'function', 'function': {'name': 'run_sql_query', 'arguments': '{\"query\": \"YOUR SQL QUERY HERE\"}'}}]. Replace 'call_id' with a unique ID like 'call_123'.",
        "- When calling run_sql_query, pass the SQL query as a JSON object with 'query' key, e.g., {'query': 'SELECT High FROM prices WHERE Ticker = \"AAPL\" AND DATE(Date) = \"2025-02-28\"'}.",
        "- After receiving the result from run_sql_query, process it to extract the relevant value (e.g., Close, Open, High, Low, Volume).",
        "- If the query result is empty (e.g., []), return 'I don't know. No data available for [company] on [date].'",
        "- If the query result is valid, format the answer clearly, e.g., 'The closing price of Microsoft on 2024-03-15 was $X.XX.' for price queries or 'The trading volume of Apple on 2024-01-02 was X.' for volume queries.",
        "- For price values, format to 2 decimal places (e.g., $123.45). For volume, use whole numbers.",
        "- Examples of queries and responses:",
        "  - Highest price of Apple on 2025-02-28: Use mapping ('Apple Inc.' → 'AAPL'), then run_sql_query with {'query': 'SELECT High FROM prices WHERE Ticker = \"AAPL\" AND DATE(Date) = \"2025-02-28\"'}",
        "  - Closing price of 3M on 2023-12-15: Use mapping ('3M Company' → 'MMM'), then run_sql_query with {'query': 'SELECT Close FROM prices WHERE Ticker = \"MMM\" AND DATE(Date) = \"2023-12-15\"'}",
        "  - Trading volume of Walmart between 2025-01-01 and 2025-02-01: Use mapping ('Walmart Inc.' → 'WMT'), then run_sql_query with {'query': 'SELECT SUM(Volume) FROM prices WHERE Ticker = \"WMT\" AND DATE(Date) BETWEEN \"2025-01-01\" AND \"2025-02-01\"'}",
        "- If no date is specified, assume the latest available date or summarize over time.",
        "- If the question is about a company not in the mapping or data not available, say 'I don't know.' Don't make up answers.",
        "- Present answers clearly, e.g., 'The highest price of AAPL on 2025-02-28 was [value].' or 'The closing price of MSFT on 2024-03-15 was [value].'",
        "- After receiving the tool result, ALWAYS provide a final response in this format:\nSQL Query: [the SQL query you used]\nRaw Result: [the raw result from the query]\nAnswer: [your formatted answer based on the result].",
        "- If the tool call fails or returns no data, return 'I don't know. No data available for [company] on [date].'",
    ],
    debug_mode=True,
    max_tool_calls=5,  
    tool_call_strategy="auto",
)
