import os
import sys
import json
from dotenv import load_dotenv
import logging
from typing import Dict, Any
from pathlib import Path
import re
import pandas as pd
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from phi.agent import Agent
from phi.model.groq import Groq
from agents.text_to_sql_agent import agent as sql_agent
from agents.visualization_agent import run_visualize_agent
from agents.rag_agent import load_agent as load_rag_agent
from agents.trading_agent import run_trading_agent
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
load_dotenv()

# Environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

TOOLS_CONFIG = {
    "text2sql_agent": {
        "intents": [
            "stock", "price", "volume", "market cap", "pe ratio", "dividend yield",
            "52 week high", "52 week low", "dividends", "stock splits",
            "sector", "industry", "country", "highest price", "lowest price",
            "average price", "total volume", "average volume", "highest volume",
            "weekly volume", "daily highlow range", "djia", "company", "ticker",
            "returns", "cumulative returns"
        ],
        "sub_query_template": "{query}",
        "description": "Queries database for stock prices or company info"
    },
    "visualization_agent": {
        "intents": [
            "chart", "graph", "plot", "visualize", "bar", "pie", "line", "scatter",
            "heatmap", "boxplot", "histogram", "distribution"
        ],
        "sub_query_template": "{query}",
        "description": "Generates visualizations from stock data"
    },
    # "rag_agent": {
    #     "intents": [
    #         "pdf", "document", "paper", "research", "contribution", "summary", 
    #         "analyze", "content", "article", "report"
    #     ],
    #     "sub_query_template": "{query}",
    #     "description": "Analyzes content from PDF documents using RAG"
    # },
    "rag_agent": {
        "intents": [
            "pdf", "document", "report", "filing", "10-k", "10k", "sec",
            "annual report", "quarterly report", "financial statement",
            "income statement", "balance sheet", "cash flow",
            "risk factors", "md&a", "management discussion",
            "according to the document", "according to the report",
            "reported", "reported revenue", "reported earnings",
            "extract", "summarize section", "find in document",
            "analyze content", "article", "paper", "content"
        ],
        "sub_query_template": "{query}",
        "description": "Analyzes content from PDF documents using RAG"
    },
    "trading_agent": {
    "intents": ["buy", "sell", "hold", "invest", "trade", "recommendation", "signal", "invest", "investment"],
    "sub_query_template": "{query}",
    "description": "Provides trading decision and GenAI explanation using ML models"
    }
}

def create_orchestrator():
    tools_config_json = json.dumps(TOOLS_CONFIG, ensure_ascii=False, indent=2)
    system_prompt = f"""
You are Orchestrator, analyzing queries and delegating tasks to text2sql_agent, visualization_agent, trading_agent, or rag_agent.
Return ONLY JSON output with agents, sub-queries, tickers, and date range. 
Do NOT include text, explanations, markdown, or code outside JSON.

Input format: JSON string with "query" (current query) and "chat_history" (list of previous interactions).
- Example input: {{"query": "Plot the cumulative return of UnitedHealth Group (UNH) during 2024", "chat_history": []}}

1. Analyze Chat History for Context:
   - Use chat_history (last 2 interactions) to understand context.
   - Example: If chat_history contains "Tell me about Microsoft stock" and current query is "Plot its prices", infer "its" refers to Microsoft (MSFT).

2. Analyze Current Query:
   - Match intents:
     {tools_config_json}
   - Use text2sql_agent for stock/data queries (e.g., 'stock', 'price', 'volume', 'returns', 'cumulative returns').
   - Add visualization_agent after text2sql_agent if query contains visualization intents (e.g., 'chart', 'plot', 'line').
   - Use rag_agent for queries about PDF documents or research papers (e.g., 'pdf', 'document', 'contribution', 'summary').
   - Use trading_agent when user asks about "buy", "sell", "hold", "invest", "trade", or "recommendation".
   - Output agents in order (e.g., ["text2sql_agent", "visualization_agent"], ["text2sql_agent", "trading_agent"], or ["rag_agent"]).

3. Extract Tickers and Date Range:
   - Identify tickers from company names using mapping or directly if ticker is mentioned (e.g., 'UNH' for UnitedHealth Group).
   - Extract date range (e.g., 'during 2024' → {{"start_date": "2024-01-01", "end_date": "2024-12-31"}}).
   - If no date range, set date_range to null.
   - For rag_agent, tickers and date_range are typically null unless the query involves stock-related PDF content.

4. Special Rule — Trading Queries:
   - If the query involves "buy", "sell", "hold", "invest", or "recommendation":
     - Always include both "text2sql_agent" and "trading_agent" in order.
     - The text2sql_agent must fetch recent 60–90 trading days of historical data for the ticker.
     - The SQL MUST select at least these columns exactly: Date, Open, High, Low, Close, "Adj Close", Volume, Ticker.
     - Example SQL query to use (write it as plain text):
       SELECT Date, Open, High, Low, Close, "Adj Close", Volume, Ticker
       FROM prices
       WHERE Ticker = '[TICKER]'
       ORDER BY Date DESC
       LIMIT 90;
     - Even if user asks "for the next 5 days", still fetch the last 60–90 days of historical data for model features.
     - Then trading_agent will make the decision (BUY, HOLD, SELL) and provide explanation.
5. Create Sub-Queries:
   - For text2sql_agent, create sub-query to fetch required data.
     Tailor it to the visualization if applicable (e.g., for time series plot, "What are the closing prices of [ticker] during [period]?").
   - For trading_agent, pass the investment or recommendation question directly (e.g., "Should I buy or sell AAPL for the next 5 days?").
   - For visualization_agent, use the original query as sub-query to describe the visualization.
   - For rag_agent, use the original query as the sub-query for document analysis.

6. Output JSON Structure:
   - Always output strictly in JSON:
     {{
       "status": "success|error",
       "message": "Query analyzed successfully|Error message",
       "data": {{
         "agents": ["agent1", "agent2"],
         "sub_queries": {{"agent1": "sub_query1", "agent2": "sub_query2"}},
         "tickers": ["UNH"],
         "date_range": null|{{"start_date": "...", "end_date": "..."}},
         "sql_result": [],
         "result": "",
         "rag_result": ""
       }}
     }}
"""

    return Agent(
        model=Groq(
            id="llama-3.3-70b-versatile",
            api_key=groq_api_key,
            timeout=30,
            max_retries=5,
            temperature=0.2,
            max_tokens=1000,
            top_p=0.8,
        ),
        system_prompt=system_prompt,
        debug_mode=True,
    )
    
rag_agent, _ = load_rag_agent()
def run_orchestrator(query: str, chat_history: list = []) -> Dict[str, Any]:
    orchestrator = create_orchestrator()
    input_json = json.dumps({"query": query, "chat_history": chat_history}, ensure_ascii=False)
    try:
        response = orchestrator.run(input_json)
        # response_dict = json.loads(response.content) if hasattr(response, "content") else response
        raw_content = getattr(response, "content", response)
        if isinstance(raw_content, bytes):
            raw_content = raw_content.decode("utf-8")

        # 🧹 Loại bỏ code fences hoặc markdown (```json ... ```)
        clean_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip())

        try:
            response_dict = json.loads(clean_content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse orchestrator response:\n{clean_content}")
            response_dict = {
                "status": "error",
                "message": "Failed to parse orchestrator JSON output.",
                "data": {}
            }

        logger.debug(f"Orchestrator response: {json.dumps(response_dict, ensure_ascii=False)}")
        
        agents = response_dict.get("data", {}).get("agents", [])
        sub_queries = response_dict.get("data", {}).get("sub_queries", {})
        lower_query = query.lower()
        
        if not agents:
            return response_dict
        
        previous_result = None
        previous_result_data = []
        
        for agent_name in agents:
            sub_query = sub_queries.get(agent_name, query)
            if agent_name == "text2sql_agent":
                logger.debug(f"Executing text2sql_agent with sub-query: {sub_query}")
                try:
                    result = sql_agent.run(sub_query, stream=False, execute_tools=True)
                    previous_result = getattr(result, "content", str(result)) if result else "No data retrieved from SQL."
                    
                    # Extract tool output from agent's memory
                    tool_messages = [msg for msg in sql_agent.memory.messages if msg.role == 'tool']
                    if tool_messages:
                        tool_output = tool_messages[-1].content
                        try:
                            previous_result_data = json.loads(tool_output)
                            if not isinstance(previous_result_data, list):
                                previous_result_data = []
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing failed on tool output: {e}. Tool output: {tool_output}")
                            previous_result_data = []
                    else:
                        logger.debug("No tool messages found in memory.")
                        previous_result_data = []
                    
                    response_dict["data"]["sql_result"] = previous_result_data
                    response_dict["data"]["result"] = previous_result
                    logger.debug(f"Parsed sql_result: {previous_result_data}")
                except Exception as e:
                    logger.error(f"Text2SQL agent failed: {str(e)}")
                    # Still attempt to extract tool output even on failure
                    tool_messages = [msg for msg in sql_agent.memory.messages if msg.role == 'tool']
                    if tool_messages:
                        tool_output = tool_messages[-1].content
                        try:
                            previous_result_data = json.loads(tool_output)
                            if not isinstance(previous_result_data, list):
                                previous_result_data = []
                        except json.JSONDecodeError as je:
                            logger.error(f"JSON parsing failed on tool output: {je}. Tool output: {tool_output}")
                            previous_result_data = []
                    else:
                        previous_result_data = []
                    response_dict["data"]["sql_result"] = previous_result_data
                    response_dict["data"]["result"] = f"Text2SQL error: {str(e)} but data retrieved if available"

                # --- Post-processing for specific analytical query: DJIA equal-weight total return 2024 ---
                def _extract_sql_query(text: str) -> str:
                    if not isinstance(text, str):
                        return ""
                    m = re.search(r"SQL Query:\s*(.*)", text)
                    return m.group(1).strip() if m else ""

                def _compute_equal_weight_return(rows: list) -> float | None:
                    if not rows or not isinstance(rows, list):
                        return None
                    df = pd.DataFrame(rows)
                    if df.empty:
                        return None
                    # Find date column
                    date_col = next((c for c in df.columns if "date" in c.lower()), None)
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                        df = df.dropna(subset=[date_col])
                        df = df[df[date_col].dt.year == 2024] if not df.empty else df
                        df = df.sort_values(date_col)
                    # Determine close column
                    close_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
                    if close_col is None or "Ticker" not in df.columns:
                        return None
                    if df.empty:
                        return None
                    def ticker_return(g: pd.DataFrame) -> float | None:
                        if g.empty:
                            return None
                        first = g[close_col].iloc[0]
                        last = g[close_col].iloc[-1]
                        try:
                            return (float(last) / float(first)) - 1.0
                        except Exception:
                            return None
                    rets = df.groupby("Ticker").apply(ticker_return)
                    rets = rets.dropna()
                    if rets.empty:
                        return None
                    return float(rets.mean())

                if (
                    "djia" in lower_query
                    and "total return" in lower_query
                    and "equal" in lower_query
                    and "weight" in lower_query
                ):
                    # Luôn tính từ DB đầy đủ để tránh lỗi cắt dữ liệu do giới hạn token
                    eq_ret = None
                    try:
                        db_path = BASE_DIR / "data" / "djia.db"
                        with sqlite3.connect(db_path) as conn:
                            sql = """
                            WITH bounds AS (
                                SELECT Ticker,
                                       MIN(Date) AS start_date,
                                       MAX(Date) AS end_date
                                FROM prices
                                WHERE STRFTIME('%Y', Date) = '2024'
                                GROUP BY Ticker
                            )
                            SELECT b.Ticker,
                                   p_start.Close AS start_close,
                                   p_end.Close   AS end_close
                            FROM bounds b
                            JOIN prices p_start ON p_start.Ticker = b.Ticker AND p_start.Date = b.start_date
                            JOIN prices p_end   ON p_end.Ticker   = b.Ticker AND p_end.Date   = b.end_date
                            """
                            df_bounds = pd.read_sql_query(sql, conn)
                            if not df_bounds.empty:
                                df_bounds["ret"] = df_bounds["end_close"] / df_bounds["start_close"] - 1.0
                                eq_ret = float(df_bounds["ret"].mean())
                    except Exception as e:
                        logger.error(f"DB compute for equal-weight return failed: {e}")

                    # Nếu DB compute fail, fallback về dữ liệu agent đã trả (có thể bị cắt)
                    if eq_ret is None:
                        eq_ret = _compute_equal_weight_return(previous_result_data)

                    if eq_ret is not None:
                        sql_q = _extract_sql_query(previous_result)
                        response_dict["data"]["result"] = (
                            f"SQL Query: {sql_q or 'N/A'}\n"
                            f"Raw Result: Computed equal-weight returns across DJIA tickers in 2024 using first/last trading day per ticker\n"
                            f"Answer: Approximately {eq_ret*100:.2f}%"
                        )

                # --- Post-processing: đếm số ngày có giá đóng cửa nằm trong 1 std-dev quanh mean ---
                if (
                    "trading day" in lower_query
                    and "standard deviation" in lower_query
                    and "mean" in lower_query
                ):
                    def _count_within_std(rows: list) -> tuple[int | None, float | None, float | None]:
                        if not rows or not isinstance(rows, list):
                            return (None, None, None)
                        df = pd.DataFrame(rows)
                        if df.empty:
                            return (None, None, None)
                        # Date filter 2024 nếu có cột date
                        date_col = next((c for c in df.columns if "date" in c.lower()), None)
                        if date_col:
                            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                            df = df.dropna(subset=[date_col])
                            df = df[df[date_col].dt.year == 2024] if not df.empty else df
                        # Close column
                        close_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
                        if close_col is None or df.empty:
                            return (None, None, None)
                        mean_v = df[close_col].mean()
                        std_v = df[close_col].std(ddof=0)
                        lower = mean_v - std_v
                        upper = mean_v + std_v
                        cnt = int(df[(df[close_col] >= lower) & (df[close_col] <= upper)].shape[0])
                        return (cnt, float(mean_v), float(std_v))

                    count_days, mean_v, std_v = _count_within_std(previous_result_data)

                    # Nếu thiếu dữ liệu, thử truy vấn trực tiếp DB
                    if count_days is None:
                        try:
                            # Ước lượng ticker từ query hoặc dữ liệu
                            ticker = None
                            if previous_result_data and isinstance(previous_result_data, list):
                                tcol = "Ticker"
                                if tcol in pd.DataFrame(previous_result_data).columns:
                                    ticker = str(pd.DataFrame(previous_result_data)[tcol].iloc[-1])
                            if not ticker:
                                m = re.search(r"\b([A-Z]{1,5})\b", query)
                                if m:
                                    ticker = m.group(1)
                            if ticker:
                                db_path = BASE_DIR / "data" / "djia.db"
                                with sqlite3.connect(db_path) as conn:
                                    sql = f"""
                                    SELECT Date, Close
                                    FROM prices
                                    WHERE Ticker = '{ticker.upper()}'
                                      AND STRFTIME('%Y', Date) = '2024'
                                    ORDER BY Date
                                    """
                                    df_db = pd.read_sql_query(sql, conn)
                                    cnt, mean_v, std_v = _count_within_std(df_db.to_dict(orient="records"))
                                    count_days = cnt
                        except Exception as e:
                            logger.error(f"DB compute for std-dev count failed: {e}")

                    if count_days is not None:
                        sql_q = _extract_sql_query(previous_result)
                        response_dict["data"]["result"] = (
                            f"SQL Query: {sql_q or 'N/A'}\n"
                            f"Raw Result: mean={mean_v:.2f}, std={std_v:.2f}, days_within_1std={count_days}\n"
                            f"Answer: {count_days} days"
                        )

                # --- Post-processing: tính correlation daily returns (ví dụ AAPL vs MSFT) ---
                if (
                    "correlation" in lower_query
                    and ("daily return" in lower_query or "daily returns" in lower_query or "returns" in lower_query)
                ):
                    def _compute_corr(rows: list) -> tuple[float | None, list[str]]:
                        if not rows or not isinstance(rows, list):
                            return (None, [])
                        df = pd.DataFrame(rows)
                        if df.empty:
                            return (None, [])
                        date_col = next((c for c in df.columns if "date" in c.lower()), None)
                        if date_col:
                            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                            df = df.dropna(subset=[date_col])
                            df = df[df[date_col].dt.year == 2024] if not df.empty else df
                        close_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
                        if close_col is None or "Ticker" not in df.columns or df.empty:
                            return (None, [])
                        piv = df.pivot(index=date_col, columns="Ticker", values=close_col)
                        rets = piv.pct_change().dropna()
                        if rets.shape[1] < 2 or rets.empty:
                            return (None, list(piv.columns))
                        corr_mat = rets.corr()
                        cols = list(corr_mat.columns)
                        corr_val = None
                        if len(cols) >= 2:
                            corr_val = float(corr_mat.iloc[0, 1])
                        return (corr_val, cols)

                    corr_val, tickers = _compute_corr(previous_result_data)

                    # Fallback: truy vấn DB nếu thiếu dữ liệu
                    if corr_val is None:
                        try:
                            ticker_set = set()
                            if tickers:
                                ticker_set.update([str(t) for t in tickers if t])
                            if not ticker_set:
                                # lấy từ regex query
                                regex_tickers = re.findall(r"\b[A-Z]{1,5}\b", query)
                                ticker_set.update(regex_tickers)
                            # chỉ giữ tối đa 5 ticker để an toàn
                            ticker_list = list(ticker_set)[:5]
                            if len(ticker_list) >= 2:
                                db_path = BASE_DIR / "data" / "djia.db"
                                with sqlite3.connect(db_path) as conn:
                                    sql = f"""
                                    SELECT Date, Close, Ticker
                                    FROM prices
                                    WHERE Ticker IN ({','.join(['?' for _ in ticker_list])})
                                      AND STRFTIME('%Y', Date) = '2024'
                                    ORDER BY Date
                                    """
                                    df_db = pd.read_sql_query(sql, conn, params=ticker_list)
                                    corr_val, tickers = _compute_corr(df_db.to_dict(orient="records"))
                        except Exception as e:
                            logger.error(f"DB compute for correlation failed: {e}")

                    if corr_val is not None:
                        sql_q = _extract_sql_query(previous_result)
                        tickers_str = ", ".join(tickers) if tickers else "tickers"
                        response_dict["data"]["result"] = (
                            f"SQL Query: {sql_q or 'N/A'}\n"
                            f"Raw Result: correlation({tickers_str}) = {corr_val:.4f}\n"
                            f"Answer: {corr_val:.2f}"
                        )
                
            elif agent_name == "visualization_agent":
                logger.debug(f"Executing visualization_agent with sub-query: {sub_query}")
                logger.debug(f"Passing sql_data to visualization_agent: {len(previous_result_data)} rows")
                viz_result = run_visualize_agent(sub_query, sql_data=previous_result_data)
                response_dict["data"]["visualization"] = viz_result.get("visualization")
                # ⚠️ Quan trọng: KHÔNG ghi đè `result` văn bản từ text2sql_agent.
                # Frontend (`scripts/app.py`) dựa vào `data["result"]` để trích xuất
                # `SQL Query` và `Answer`. Nếu chúng ta thay thế bằng
                # chuỗi "Visualization generated" thì sẽ mất câu trả lời gốc.
                #
                # Vì vậy:
                # - Luôn lưu message của visualization vào một field riêng
                #   để tiện debug nếu cần.
                # - Chỉ ghi vào `result` nếu hiện tại chưa có kết quả nào.
                viz_message = viz_result.get("message", "No visualization created.")
                response_dict["data"]["viz_message"] = viz_message
                if not response_dict["data"].get("result"):
                    response_dict["data"]["result"] = viz_message

            elif agent_name == "trading_agent":
                logger.debug(f"Executing trading_agent with sql_result: {len(previous_result_data)} rows")
                trade_result = run_trading_agent(sub_query, sql_result=previous_result_data)
                response_dict["data"]["trade_result"] = trade_result

                if trade_result.get("status") == "success":
                    response_dict["data"]["result"] = trade_result["decision"]
                else:
                    response_dict["data"]["result"] = trade_result.get("message", "Trading agent failed.")   

            elif agent_name == "rag_agent":
                logger.debug(f"Executing rag_agent with sub-query: {sub_query}")
                try:
                    # rag_agent, _ = load_rag_agent()
                    result = rag_agent.run(sub_query, stream=False)
                    rag_result = getattr(result, "content", str(result)) if result else "No data retrieved from RAG."
                    response_dict["data"]["rag_result"] = rag_result
                    response_dict["data"]["result"] = rag_result
                    logger.debug(f"RAG result: {rag_result}")
                except Exception as e:
                    logger.error(f"RAG agent failed: {str(e)}")
                    response_dict["data"]["rag_result"] = f"RAG error: {str(e)}"
                    response_dict["data"]["result"] = f"RAG error: {str(e)}"
        
        return response_dict
    except Exception as e:
        logger.error(f"Error running orchestrator: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing query: {str(e)}",
            "data": {
                "agents": [], 
                "sub_queries": {}, 
                "tickers": [], 
                "date_range": None, 
                "sql_result": [], 
                "result": "No data retrieved.", 
                "rag_result": "No data retrieved from RAG."
            }
        }