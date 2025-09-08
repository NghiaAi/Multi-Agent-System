# app.py
import streamlit as st
import json
import re

from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from agents.orchestrator_agent import run_orchestrator
import streamlit as st
import json

st.title("Agent Query System")

# Ô nhập câu hỏi
query = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: What is the closing price of Apple on 2025-02-28?")

# Nút submit
if st.button("Gửi câu hỏi"):
    if query:
        with st.spinner("Đang xử lý câu hỏi..."):
            # Chạy orchestrator mà không cần chat history
            response = run_orchestrator(query, chat_history=[])
        
        # Hiển thị kết quả
        st.subheader("Kết quả:")
        
        # Kiểm tra status
        if response.get("status") == "success":
            data = response.get("data", {})
            agents = data.get("agents", [])
            result = data.get("result", "Không có kết quả cụ thể.")
            
            if "text2sql_agent" in agents:
                # Extract SQL Query và Answer từ result với regex mạnh hơn
                sql_match = re.search(r"SQL Query: (.*?)\nRaw Result:.*?\nAnswer: (.*?)(?:\n|$)", result, re.DOTALL)
                if sql_match:
                    sql_query = sql_match.group(1).strip()
                    answer = sql_match.group(2).strip()
                    st.subheader("Answer:")
                    st.write(answer)
                    st.subheader("SQL Query:")
                    st.code(sql_query, language="sql")
                else:
                    st.error("Không thể phân tích câu SQL hoặc câu trả lời từ phản hồi.")
                    st.write(result)
            else:
                # Nếu không phải text2sql_agent, hiển thị result trực tiếp
                st.write(result)
        else:
            st.error(f"Lỗi: {response.get('message', 'Không xác định')}")
    else:
        st.warning("Vui lòng nhập câu hỏi trước khi gửi.")