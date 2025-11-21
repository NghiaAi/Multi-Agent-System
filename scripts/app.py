

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

query = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Plot the time series of Microsoft (MSFT) stock closing price from June 1, 2024 to September 30, 2024. Hoặc: What is the main contribution of the paper?")

if st.button("Gửi câu hỏi"):
    if query:
        with st.spinner("Đang xử lý câu hỏi..."):
            response = run_orchestrator(query, chat_history=[])
        
        st.subheader("Kết quả:")
        
        if response.get("status") == "success":
            data = response.get("data", {})
            agents = data.get("agents", [])
            result = data.get("result", "Không có kết quả cụ thể.")
            sql_result = data.get("sql_result", [])
            visualization = data.get("visualization")
            rag_result = data.get("rag_result", "Không có kết quả từ RAG.")
            trade_result = data.get("trade_result", {})
            
            if "rag_agent" in agents:
                st.subheader("Kết quả từ RAG Agent:")
                st.write(rag_result)
            
            # if "text2sql_agent" in agents:
            #     sql_match = re.search(r"SQL Query: (.*?)\nRaw Result:.*?\nAnswer: (.*?)(?:\n|$)", result, re.DOTALL)
            #     if sql_match:
            #         sql_query = sql_match.group(1).strip()
            #         answer = sql_match.group(2).strip()
            #         st.subheader("Answer:")
            #         st.write(answer)
            #         st.subheader("SQL Query:")
            #         st.code(sql_query, language="sql")
            #     else:
            #         st.subheader("Result:")
            #         st.write(result)
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
                    # Nếu result không phải string (tức là dict từ trading agent)
                    st.subheader("Result:")
                    st.json(result)

            # if "trading_agent" in agents:
            #     trade_result = data.get("trade_result", {})
            #     if trade_result.get("status") == "success":
            #         decision = trade_result["decision"]
            #         st.subheader(f"Trading Decision for {decision['ticker']}")
            #         st.write(f"**Signal:** {decision['signal']}")
            #         st.write(f"**Confidence:** {decision['confidence']}")
            #         st.write(f"**Explanation:** {decision.get('explanation', '')}")
            #     else:
            #         st.error(trade_result.get("message", "Trading agent failed."))
            if "trading_agent" in agents:
                trade_result = data.get("trade_result", {})
                
                if trade_result.get("status") == "success":
                    decision = trade_result.get("decision", {})
                    ticker = decision.get("ticker", "UNKNOWN")
                    signal = decision.get("signal", "UNKNOWN")
                    confidence = decision.get("confidence", 0.0)
                    explanation = decision.get("explanation", "")

                    st.subheader(f"Trading Decision for {ticker}")

                    # Hiển thị Signal theo màu
                    if signal == "BUY":
                        st.success(f"**Signal:** {signal}")
                    elif signal == "SELL":
                        st.error(f"**Signal:** {signal}")
                    elif signal == "HOLD":
                        st.info(f"**Signal:** {signal}")
                    else:
                        st.warning(f"**Signal:** {signal}")

                    # Confidence
                    st.write(f"**Confidence:** {confidence:.2f}")

                    # Probabilities nếu có
                    probs = decision.get("probabilities")
                    if probs:
                        st.write("**Probabilities:**")
                        st.json(probs)

                    # Explanation
                    st.write("**Explanation:**")
                    if isinstance(explanation, dict):
                        st.json(explanation)
                    else:
                        st.write(explanation)

                else:
                    # Nếu trading agent trả về lỗi
                    message = trade_result.get("message", "Trading agent failed.")
                    st.error(message)

            if "visualization_agent" in agents and visualization is not None:
                st.subheader("Visualization:")
                try:
                    # Convert NumPy array to PIL Image and display
                    img = Image.fromarray(visualization)
                    st.image(img, use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying visualization: {str(e)}")
            
        else:
            st.error(f"Lỗi: {response.get('message', 'Không xác định')}")
    else:
        st.warning("Vui lòng nhập câu hỏi trước khi gửi.")
