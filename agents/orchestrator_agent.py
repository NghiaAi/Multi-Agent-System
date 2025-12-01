import os
import sys
import json
from dotenv import load_dotenv
import logging
from typing import Dict, Any
from pathlib import Path
import re

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
    "rag_agent": {
        "intents": [
            "pdf", "document", "paper", "research", "contribution", "summary", 
            "analyze", "content", "article", "report"
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
#     system_prompt = f"""
# You are Orchestrator, analyzing queries and delegating tasks to text2sql_agent, visualization_agent, or rag_agent. Return ONLY JSON output with agents, sub-queries, tickers, and date range. Do NOT include text, explanations, markdown, or code outside JSON.

# Input format: JSON string with "query" (current query) and "chat_history" (list of previous interactions).
# - Example input: {{"query": "Plot the cumulative return of UnitedHealth Group (UNH) during 2024", "chat_history": []}}

# 1. Analyze Chat History for Context:
#    - Use chat_history (last 5 interactions) to understand context.
#    - Example: If chat_history contains "Tell me about Microsoft stock" and current query is "Plot its prices", infer "its" refers to Microsoft (MSFT).

# 2. Analyze Current Query:
#    - Match intents:
#      {tools_config_json}
#    - Use text2sql_agent for stock/data queries (e.g., 'stock', 'price', 'volume', 'returns', 'cumulative returns').
#    - Add visualization_agent after text2sql_agent if query contains visualization intents (e.g., 'chart', 'plot', 'line').
#    - Use rag_agent for queries about PDF documents or research papers (e.g., 'pdf', 'document', 'contribution', 'summary').
#    - Use trading_agent when user asks for "buy", "sell", "hold", "invest", or "recommendation".
#    - Output agents in order (e.g., ["text2sql_agent", "visualization_agent"] or ["rag_agent"]).

# 3. Extract Tickers and Date Range:
#    - Identify tickers from company names using mapping or directly if ticker is mentioned (e.g., 'UNH' for UnitedHealth Group).
#    - Extract date range (e.g., 'during 2024' → {{"start_date": "2024-01-01", "end_date": "2024-12-31"}}).
#    - If no date range, set date_range to null.
#    - For rag_agent, tickers and date_range are typically null unless the query involves stock-related PDF content.

# 4. Create Sub-Queries:
#    - For text2sql_agent, create sub-query to fetch required data. Tailor it to the visualization if applicable (e.g., for time series plot, "What are the closing prices of [ticker] during [period]?"; for distribution pie chart, "What is the number of DJIA companies in each sector?").
#    - For visualization_agent, use the original query as sub-query to describe the visualization.
#    - For rag_agent, use the original query as the sub-query for document analysis.

# 5. Output JSON Structure:
#    - {{"status": "success|error", "message": "Query analyzed successfully|Error message", "data": {{"agents": ["agent1", "agent2"], "sub_queries": {{"agent1": "sub_query1", "agent2": "sub_query2"}}, "tickers": ["UNH"], "date_range": null|{{start_date, end_date}}, "sql_result": [], "result": "", "rag_result": ""}}}}
# """

    system_prompt = f"""
You are Orchestrator, analyzing queries and delegating tasks to text2sql_agent, visualization_agent, trading_agent, or rag_agent.
Return ONLY JSON output with agents, sub-queries, tickers, and date range. 
Do NOT include text, explanations, markdown, or code outside JSON.

Input format: JSON string with "query" (current query) and "chat_history" (list of previous interactions).
- Example input: {{"query": "Plot the cumulative return of UnitedHealth Group (UNH) during 2024", "chat_history": []}}

1. Analyze Chat History for Context:
   - Use chat_history (last 5 interactions) to understand context.
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
     - Example SQL query to use (write it as plain text):
       SELECT Date, Open, High, Low, Close, Volume, Ticker
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
                
            elif agent_name == "visualization_agent":
                logger.debug(f"Executing visualization_agent with sub-query: {sub_query}")
                # logger.debug(f"Passing sql_data to visualization_agent: {previous_result_data}")
                logger.debug(f"Passing sql_data to visualization_agent: {len(previous_result_data)} rows")
                viz_result = run_visualize_agent(sub_query, sql_data=previous_result_data)
                response_dict["data"]["visualization"] = viz_result.get("visualization")
                response_dict["data"]["result"] = viz_result.get("message", "No visualization created.")

            elif agent_name == "trading_agent":
                logger.debug(f"Executing trading_agent with sql_result: {previous_result_data}")
                trade_result = run_trading_agent(sub_query, sql_result=previous_result_data)
                response_dict["data"]["trade_result"] = trade_result

                if trade_result.get("status") == "success":
                    response_dict["data"]["result"] = trade_result["decision"]
                else:
                    response_dict["data"]["result"] = trade_result.get("message", "Trading agent failed.")   

            elif agent_name == "rag_agent":
                logger.debug(f"Executing rag_agent with sub-query: {sub_query}")
                try:
                    rag_agent, _ = load_rag_agent()
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