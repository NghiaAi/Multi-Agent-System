import os
import sys
import json
from dotenv import load_dotenv
import logging
from typing import Dict, Any
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from phi.agent import Agent
from phi.model.groq import Groq
from agents.rag_agent import load_agent
from agents.text_to_sql_agent import agent as sql_agent

# Set up basic logging
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
            "weekly volume", "daily highlow range", "djia", "company", "ticker"
        ],
        "sub_query_template": "{query}",
        "description": "Queries database for stock prices or company info"
    },
    "rag_agent": {
        "intents": [
            "gpt-4", "ai", "technical report", "alignment", "pretraining",
            "rlhf", "sft", "safety mitigation", "fine-tuning", "post-pretraining",
            "model behavior", "human feedback", "reinforcement learning"
        ],
        "sub_query_template": "{query}",
        "description": "Answers questions about GPT-4 technical report or AI alignment"
    }
}

def create_orchestrator():
    tools_config_json = json.dumps(TOOLS_CONFIG, ensure_ascii=False, indent=2)
    system_prompt = f"""
You are Orchestrator, analyzing queries and delegating tasks to text2sql_agent or rag_agent. Return ONLY JSON output with agents, sub-queries, tickers, and date range. Do NOT include text, explanations, markdown, or code outside JSON.

Input format: JSON string with "query" (current query) and "chat_history" (list of previous interactions).
- Example input: {{"query": "What methods were used to align GPT-4 after pretraining?", "chat_history": [{{"role": "user", "content": "Tell me about GPT-4", "timestamp": "2025-05-24 08:16:45"}}, {{"role": "assistant", "content": "GPT-4 is an AI model...", "timestamp": "2025-05-24 08:16:50"}}]}}

1. Analyze Chat History for Context:
   - Use chat_history to understand context and relationships between queries.
   - Example: If chat_history contains "Tell me about GPT-4" and current query is "What methods were used to align it?", infer "it" refers to GPT-4.
   - Limit history to the last 5 interactions to avoid token overflow.

2. Analyze Current Query:
   - Match intents:
     {tools_config_json}
   - Prioritize text2sql_agent for stock/data queries (e.g., 'stock', 'price', 'volume', 'djia').
   - Prioritize rag_agent for AI/GPT-4 queries (e.g., 'gpt-4', 'alignment', 'pretraining', 'rlhf').
   - Extract:
     - Tickers: e.g., 'AAPL, MSFT' from '(symbol: AAPL)' or names (e.g., 'Apple'); [] if not applicable. Use chat_history to infer tickers if query is ambiguous (e.g., "its stock price" after mentioning "Apple").
     - Date range: e.g., 'in 2024' → {{'start_date': '2024-01-01', 'end_date': '2024-12-31'}}; 'on 2025-04-26' → {{'start_date': '2025-04-26', 'end_date': '2025-04-26'}}; null for non-time-based queries. Use chat_history to infer dates if query is ambiguous (e.g., "last year" after mentioning "2024").
   - General query: Set agents=[], message='System supports GPT-4/AI queries and stock/DJIA data queries'.
   - Invalid query: Return error.

3. Delegate:
   - For text2sql_agent: Use original query as sub-query.
   - For rag_agent: Use original query as sub-query.
   - General query: Set agents=[].
   - Invalid query: Return error.

4. Output:
   - JSON: {{"status": "success"|"error", "message": "...", "data": {{"agents": ["text2sql_agent"|"rag_agent"], "sub_queries": {{}}, "tickers": [], "date_range": null|{{start_date, end_date}}}}}}
   - Example: Query: 'What is the stock price of Apple in 2024?'
     - {{"status": "success", "message": "Query analyzed successfully", "data": {{"agents": ["text2sql_agent"], "sub_queries": {{"text2sql_agent": "What is the stock price of Apple in 2024?"}}, "tickers": ["AAPL"], "date_range": {{"start_date": "2024-01-01", "end_date": "2024-12-31"}}}}}}
   - Example: Query: 'What methods were used to align GPT-4 after pretraining?'
     - {{"status": "success", "message": "Query analyzed successfully", "data": {{"agents": ["rag_agent"], "sub_queries": {{"rag_agent": "What methods were used to align GPT-4 after pretraining?"}}, "tickers": [], "date_range": null}}}}
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

# Load agents
rag_agent, _ = load_agent()
sql_agent = sql_agent

def run_orchestrator(query: str, chat_history: list = []) -> Dict[str, Any]:
    orchestrator = create_orchestrator()
    input_json = json.dumps({"query": query, "chat_history": chat_history})
    try:
        response = orchestrator.run(input_json)
        response_dict = json.loads(response.content) if hasattr(response, "content") else response
        logger.debug(f"Orchestrator response: {response_dict}")
        
        # Execute the delegated agent
        agents = response_dict.get("data", {}).get("agents", [])
        sub_queries = response_dict.get("data", {}).get("sub_queries", {})
        
        if not agents:
            return response_dict
        
        for agent_name in agents:
            sub_query = sub_queries.get(agent_name, query)
            if agent_name == "text2sql_agent":
                logger.debug(f"Executing text2sql_agent with sub-query: {sub_query}")
                result = sql_agent.run(sub_query, stream=False, execute_tools=True)
                response_dict["data"]["result"] = getattr(result, "content", str(result)) if result else "No data retrieved from SQL."
            elif agent_name == "rag_agent":
                logger.debug(f"Executing rag_agent with sub-query: {sub_query}")
                result = rag_agent.run(sub_query)
                response_dict["data"]["result"] = getattr(result, "content", str(result)) if result else "No information retrieved from RAG."
        
        return response_dict
    except Exception as e:
        logger.error(f"Error running orchestrator: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing query: {str(e)}",
            "data": {"agents": [], "sub_queries": {}, "tickers": [], "date_range": None}
        }
