import streamlit as st
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent  
sys.path.append(str(BASE_DIR)) 
from agents.orchestrator_agent import run_orchestrator  

st.title("Agent Query System")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            response = message["response"]
            if response.get("status") == "success":
                data = response.get("data", {})
                agents = data.get("agents", [])
                result = data.get("result", "No specific result.")
                sql_result = data.get("sql_result", [])
                visualization = data.get("visualization")
                rag_result = data.get("rag_result", "No result from RAG.")
                trade_result = data.get("trade_result", {})
                
                if "rag_agent" in agents:
                    st.subheader("RAG Result:")
                    # Parse KB Hits and Answer from rag_result
                    kb_match = re.search(r"\*\*KB Hits\*\*:\n(.*?)\n\n\*\*Answer\*\*:\n(.*?)$", rag_result, re.DOTALL)
                    if kb_match:
                        kb_hits = kb_match.group(1).strip()
                        answer = kb_match.group(2).strip()
                        st.write("**KB Hits:**")
                        st.markdown(kb_hits)
                        st.write("**Answer:**")
                        st.markdown(answer)
                    else:
                        st.write(rag_result)
                
                if "text2sql_agent" in agents and "trading_agent" not in agents:
                    if isinstance(result, str):
                        sql_match = re.search(r"SQL Query: (.*?)\nRaw Result:.*?\nAnswer: (.*?)(?:\n|$)", result, re.DOTALL)
                        if sql_match:
                            sql_query = sql_match.group(1).strip()
                            answer = sql_match.group(2).strip()
                            st.subheader("Answer:")
                            st.write(answer)
                            st.subheader("SQL Query:")
                            st.code(sql_query, language="sql")
                        else:
                            st.subheader("Result:")
                            st.write(result)
                    else:
                        st.subheader("Result:")
                        st.json(result)

                if "trading_agent" in agents:
                    if trade_result.get("status") == "success":
                        decision = trade_result.get("decision", {})
                        ticker = decision.get("ticker", "UNKNOWN")
                        signal = decision.get("signal", "UNKNOWN")
                        probabilities = decision.get("probabilities", {})
                        explanation = decision.get("explanation", "")

                        st.subheader(f"Trading Decision for {ticker}")
                        if signal == "BUY":
                            st.success(f"**Signal:** {signal}")
                        elif signal == "SELL":
                            st.error(f"**Signal:** {signal}")
                        elif signal == "HOLD":
                            st.info(f"**Signal:** {signal}")
                        else:
                            st.warning(f"**Signal:** {signal}")

                        st.write("**Probabilities:**")
                        st.json(probabilities)

                        # st.subheader("Explanation:")
                        st.markdown(explanation)

                    else:
                        message = trade_result.get("message", "Trading agent failed.")
                        st.error(message)

                if "visualization_agent" in agents and visualization is not None:
                    st.subheader("Visualization:")
                    try:
                        img = Image.fromarray(visualization)
                        st.image(img, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")
            else:
                st.error(f"Error: {response.get('message', 'Unknown')}")

# Chat input at the bottom
query = st.chat_input(
    placeholder="Ask about stock prices • charts • 10-K filings • buy/sell signals..."
)

if query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(query)
    
    if "full_user_history" not in st.session_state:
        st.session_state.full_user_history = []
    st.session_state.full_user_history.append(query)
    chat_history = st.session_state.full_user_history[-5:]

    with st.spinner("Processing..."):
        response = run_orchestrator(query, chat_history=chat_history,execute_agents=True)
    st.session_state.messages.append({"role": "assistant", "response": response})
    
    with st.chat_message("assistant"):
        if response.get("status") == "success":
            data = response.get("data", {})
            agents = data.get("agents", [])
            result = data.get("result", "No specific result.")
            sql_result = data.get("sql_result", [])
            visualization = data.get("visualization")
            rag_result = data.get("rag_result", "No result from RAG.")
            trade_result = data.get("trade_result", {})
            
            if "rag_agent" in agents:
                st.subheader("RAG Result:")
                kb_match = re.search(r"\*\*KB Hits\*\*:\n(.*?)\n\n\*\*Answer\*\*:\n(.*?)$", rag_result, re.DOTALL)
                if kb_match:
                    kb_hits = kb_match.group(1).strip()
                    answer = kb_match.group(2).strip()
                    st.write("**KB Hits:**")
                    st.markdown(kb_hits)
                    st.write("**Answer:**")
                    st.markdown(answer)
                else:
                    st.write(rag_result)
            
            if "text2sql_agent" in agents and "trading_agent" not in agents:
                if isinstance(result, str):
                    sql_match = re.search(r"SQL Query: (.*?)\nRaw Result:.*?\nAnswer: (.*?)(?:\n|$)", result, re.DOTALL)
                    if sql_match:
                        sql_query = sql_match.group(1).strip()
                        answer = sql_match.group(2).strip()
                        st.subheader("Answer:")
                        st.write(answer)
                        st.subheader("SQL Query:")
                        st.code(sql_query, language="sql")
                    else:
                        st.subheader("Result:")
                        st.write(result)
                else:
                    st.subheader("Result:")
                    st.json(result)

            if "trading_agent" in agents:
                if trade_result.get("status") == "success":
                    decision = trade_result.get("decision", {})
                    ticker = decision.get("ticker", "UNKNOWN")
                    signal = decision.get("signal", "UNKNOWN")
                    probabilities = decision.get("probabilities", {})
                    explanation = decision.get("explanation", "")

                    st.subheader(f"Trading Decision for {ticker}")
                    if signal == "BUY":
                        st.success(f"**Signal:** {signal}")
                    elif signal == "SELL":
                        st.error(f"**Signal:** {signal}")
                    elif signal == "HOLD":
                        st.info(f"**Signal:** {signal}")
                    else:
                        st.warning(f"**Signal:** {signal}")

                    st.write("**Probabilities:**")
                    st.json(probabilities)
                    st.subheader("Explanation:")
                    st.markdown(explanation)
                else:
                    message = trade_result.get("message", "Trading agent failed.")
                    st.error(message)

            if "visualization_agent" in agents and visualization is not None:
                st.subheader("Visualization:")
                try:
                    img = Image.fromarray(visualization)
                    st.image(img, use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying visualization: {str(e)}")
        else:
            st.error(f"Error: {response.get('message', 'Unknown')}")