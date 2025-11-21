import os
import re
import json
from phi.agent import Agent
from phi.model.groq import Groq
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

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

def create_visualize_agent():
    tools_config_json = json.dumps(company_to_symbol, ensure_ascii=False, indent=2)
    system_prompt = f"""
You are a visualization agent for stock data, creating charts as specified in user queries (e.g., time series of cumulative returns). Strictly use the chart type requested (e.g., 'line' for time series) unless infeasible.

Available chart types: bar, pie, line, scatter, heatmap, boxplot, histogram.

Process:
1. Analyze the query to identify the chart type and required columns:
   - Time series of cumulative returns: Needs 'Date', 'Close', 'Ticker'.
     - Example query: "Plot the cumulative return of UnitedHealth Group (UNH) during 2024"
     - Required columns: 'Date', 'Close', 'Ticker'
     - Compute cumulative returns as (Close[t] / Close[0] - 1) * 100
   - Boxplot of monthly closing prices: Needs 'Date' or 'Month', 'Close', 'Ticker'.
   - Histogram of daily returns: Needs 'Date', 'Close', 'Ticker'; compute returns as (Close[t] - Close[t-1])/Close[t-1].
   - Scatter plot of market cap vs P/E: Needs 'name', 'market_cap', 'pe_ratio'.
   - Heatmap of correlation matrix: Needs 'Date', 'Close', 'Ticker' for multiple tickers.
   - Scatter plot of avg volume vs avg close: Needs 'Volume', 'Close', 'Ticker'.
   - Pie chart of sector distribution: Needs 'sector', 'num_companies'.

2. Use the company-to-symbol mapping: {tools_config_json}. Match company names flexibly (remove 'Inc.', 'Company', '(The)', case-insensitive).

3. Always use provided "sql_data" (list of dicts from previous SQL query): df = pd.DataFrame(sql_data). Do NOT query the database. If sql_data is empty or lacks required columns, print 'No visualization possible.'

4. Prepare data for the specific plot:
   - Time series of cumulative returns: Ensure 'Date' is datetime (df['Date'] = pd.to_datetime(df['Date'])); filter by Ticker and year; compute cumulative returns (df['cumulative_returns'] = (df['Close'] / df['Close'].iloc[0] - 1) * 100); plot with sns.lineplot.
   - Boxplot: If 'Date' present, convert to datetime and derive 'month' (df['month'] = df['Date'].dt.strftime('%Y-%m')). If 'Month' present, use as-is.
   - Histogram: Compute returns (df['returns'] = df['Close'].pct_change().dropna()).
   - Scatter: Drop NaN values (df = df.dropna()).
   - Heatmap: Pivot (df_wide = df.pivot(index='Date', columns='Ticker', values='Close')); compute returns; calculate corr.
   - Scatter (avg volume vs close): Group by Ticker (df = df.groupby('Ticker').agg({{'Volume': 'mean', 'Close': 'mean'}})).
   - Pie: Use 'sector' and 'num_companies' directly.

5. Generate Python code to plot. Import pd, plt, sns, np, Image. Do NOT import create_engine or query SQL. Always use sql_data.
   - Start with: df = pd.DataFrame(sql_data)
   - Check required columns. If missing, print 'No visualization possible.' and return None.
   - Plot using the specified chart type:
     - Line: sns.lineplot(data=df, x='Date', y='cumulative_returns'); plt.title('Cumulative Returns of [Company] in [Year]')
     - Boxplot: sns.boxplot(data=df, x='month', y='Close'); plt.xticks(rotation=45)
     - Scatter: sns.scatterplot(x=df['market_cap'], y=df['pe_ratio'], hue=df['name'])
     - Histogram: sns.histplot(data=df['returns'], bins=20)
     - Heatmap: sns.heatmap(corr, annot=True, cmap='coolwarm')
     - Pie: plt.pie(df['num_companies'], labels=df['sector'], autopct='%1.1f%%')
   - Convert plot to NumPy array: fig = plt.gcf(); fig.canvas.draw(); img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8); img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]; plt.close(); return img

6. Output ONLY Python code in ```python\nCODE\n```. No other text.

7. If sql_data is empty or lacks required columns, print 'No visualization possible.'

Database schemas:
- companies: symbol, name, sector, industry, country, website, market_cap, pe_ratio, dividend_yield, 52_week_high, 52_week_low, description
- prices: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits, Ticker
"""
    return Agent(
        model=Groq(
            id="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=60,
            max_retries=5,
            temperature=0.2,
            max_tokens=3000,
            top_p=0.8,
        ),
        system_prompt=system_prompt,
        debug_mode=True,
    )

def run_visualize_agent(query: str, sql_data: list = None, chat_history: list = []) -> Dict[str, Any]:
    try:
        query_dict = json.loads(query) if isinstance(query, str) and query.startswith('{') else {"query": query}
        actual_query = query_dict.get("query", query)
    except json.JSONDecodeError:
        actual_query = query

    input_data = json.dumps({"query": actual_query, "sql_data": sql_data, "chat_history": chat_history}, ensure_ascii=False)

    agent = create_visualize_agent()
    try:
        response = agent.run(input_data)
        code_match = re.search(r'```python\n(.*?)\n```', response.content, re.DOTALL)
        if not code_match:
            return {"status": "error", "message": "No code generated", "visualization": None}
        
        code = code_match.group(1)
        
        local_globals = {'pd': pd, 'plt': plt, 'sns': sns, 'np': np, 'Image': Image, 'Path': Path, '__file__': __file__, 'BASE_DIR': BASE_DIR}
        local_locals = {'sql_data': sql_data}
        exec(code, local_globals, local_locals)
        
        if 'img' in local_locals and isinstance(local_locals['img'], np.ndarray):
            return {"status": "success", "message": "Visualization generated", "visualization": local_locals['img']}
        else:
            return {"status": "success", "message": "No visualization possible.", "visualization": None}
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}", "visualization": None}