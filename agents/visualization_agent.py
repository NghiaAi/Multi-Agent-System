import os
import re
import json
import textwrap
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
    system_prompt = f"""
    You are a professional stock data visualization agent using Python (pandas, matplotlib, seaborn).

    Your job: Generate ONLY executable Python code to create the exact chart requested by the user.
    Never explain, never add extra text — output only ```python ... ```

    CRITICAL RULES (follow exactly):
    1. Always start with:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    2. Load data:
    # sql_data is already a Python list of dicts passed externally.
    # NEVER rewrite sql_data manually.
    df = pd.DataFrame(sql_data)

    3. To check empty data, you MUST use:
       df = pd.DataFrame(sql_data)
       if df.empty:
           print("No visualization possible.")
           raise SystemExit

    4. Detect chart type from user query and apply correct logic:

    ──────────────────────────────────────────────
    CHART TYPE DETECTION & REQUIRED COLUMNS
    ──────────────────────────────────────────────
    • "heatmap" + "correlation" + ("return" or "daily") → Correlation matrix of daily returns
        → Required: 'Date' (or date-like), 'Close', 'Ticker'
        → Must pivot → returns → corr() → sns.heatmap(annot=True, cmap='coolwarm', vmin=-1, vmax=1)

    • "cumulative return" or "cumulative performance" → Line chart of cumulative returns
        → Required: 'Date', 'Close', 'Ticker'
        → Compute: (Close / Close.first() - 1) * 100

    • - "boxplot" + ("monthly" or "month"):
        → df['Month'] = df[date_col].dt.strftime('%Y-%m')
        → month_order = sorted(df['Month'].unique())
        → sns.boxplot(x='Month', y='Close', data=df, order=month_order)
        → plt.xticks(rotation=45, ha='right')
        → Title: e.g. "Monthly Closing Price Boxplot (DIS - 2024)"

    • "histogram" or "distribution" + "return" → Histogram of daily returns
        → pct_change() → sns.histplot()

    • "scatter" + "volume" + "close" → Scatter avg volume vs avg close per stock

    • "pie" + "sector" → Pie chart of sector distribution

    • "line" + "price" or "close" → Simple price time series

    5. Always handle date properly:
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col: df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    6. For multi-stock charts (like correlation heatmap):
    - Use pivot: prices = df.pivot(index=date_col, columns='Ticker', values='Close')
    - Drop NaNs carefully: .pct_change().dropna()
    - If final returns DataFrame has < 2 columns → print("Not enough overlapping data")

    7. Always end with converting figure to numpy image:
    fig = plt.gcf()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)

    8. Use nice styling:
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    plt.title(...) with clear title
    plt.tight_layout()

    9. Company name → Ticker mapping (use only if needed):
    {json.dumps(company_to_symbol, indent=2)}

    10. If you're unsure about columns → print(df.head()) and df.columns for debugging, but final output must still be valid plot or error message.
    """
    return Agent(
        model=Groq(
            id="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=60,
            max_retries=5,
            temperature=0.2,
            max_tokens=4000,
            top_p=0.8,
        ),
        system_prompt=system_prompt,
        debug_mode=True,
    )

def run_visualize_agent(query: str, sql_data: list = None, chat_history: list = []) -> Dict[str, Any]:
    if sql_data is None:
        sql_data = []

    try:
        # Parse query
        try:
            query_dict = json.loads(query) if isinstance(query, str) and query.startswith('{') else {"query": query}
            actual_query = query_dict.get("query", query)
        except:
            actual_query = query

        input_data = json.dumps({
            "query": actual_query,
            "sql_data": sql_data,
            "chat_history": chat_history
        }, ensure_ascii=False)

        agent = create_visualize_agent()
        response = agent.run(input_data)

        code_match = re.search(r'```python\s*(.*?)\s*```', response.content, re.DOTALL)
        if not code_match:
            return {"status": "error", "message": "No python code block found", "visualization": None}

        raw_code = code_match.group(1).strip()

        # === FIX TRIỆT ĐỂ: Dùng textwrap.dedent + đảm bảo indent đúng ===
        user_code_block = textwrap.dedent("""
        try:
            # USER CODE START - DO NOT REMOVE THIS LINE
        {user_code}
            # USER CODE END
        except Exception as e:
            print("Code execution error:", e)
        """).format(user_code=textwrap.indent(raw_code, "    "))

        full_code = f"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sql_data = {repr(sql_data)}
img = None

{user_code_block}

# === ĐẢM BẢO TẠO ẢNH ===
if img is None and plt.get_fignums():
    try:
        fig = plt.gcf()
        fig.canvas.draw()
        buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        h, w = fig.canvas.get_width_height()
        img = buffer.reshape((h, w, 4))[:, :, :3]
    except:
        img = None

plt.close('all')
"""

        # === EXECUTE ===
        namespace = {
            'pd': pd, 'plt': plt, 'sns': sns, 'np': np,
            'Image': Image, 'Path': Path, '__file__': __file__, 'BASE_DIR': BASE_DIR
        }

        exec(full_code, namespace)

        img = namespace.get('img')

        if isinstance(img, np.ndarray) and img.size > 0:
            return {"status": "success", "message": "Visualization generated", "visualization": img}
        else:
            return {"status": "error", "message": "No image generated (plot failed or empty)", "visualization": None}

    except Exception as e:
        return {"status": "error", "message": f"Agent runtime error: {str(e)}", "visualization": None}