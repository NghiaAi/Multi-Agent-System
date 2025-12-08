import streamlit as st
import re
import sys
import json
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from agents.text_to_sql_agent import agent

st.set_page_config(page_title="DJIA Financial Query", layout="centered")

st.title("Trợ lý truy vấn dữ liệu tài chính (DJIA)")
st.write("Nhập câu hỏi để truy vấn dữ liệu (giá mở cửa, đóng cửa, cao nhất, thấp nhất, khối lượng, cổ tức...).")

user_question = st.text_input("💬 Câu hỏi của bạn:")

if user_question:
    with st.spinner("Đang truy vấn..."):
        # 1. Gọi agent nhưng KHÔNG execute tools để lấy SQL query
        tool_response = agent.run(user_question, stream=False, execute_tools=False)

    st.markdown("### 📌 Câu trả lời")
    st.write(tool_response.content if hasattr(tool_response, "content") else str(tool_response))
    st.markdown("---")

    try:
        messages = tool_response.messages  
        if len(messages) > 2:  
            second_message = messages[2]  
            sql_query = None
            if second_message.tool_calls:  
                tool_call = second_message.tool_calls[0]  
                arguments = tool_call['function']['arguments'] 

                try:
                    args_dict = json.loads(arguments)  
                    sql_query = args_dict.get('query')  
                except json.JSONDecodeError as e:
                    st.warning(f"Lỗi phân tích JSON: {e}")
                    matches = re.findall(r'"query"\s*:\s*"(.*?)"(?=,\s*"name"|\s*})', arguments, re.DOTALL)
                    if matches:
                        sql_query = matches[0].replace('\\"', '"')  

            st.markdown("Câu lệnh SQL được tạo")
            if sql_query:
                st.code(sql_query, language="sql")

            else:
                st.warning("Không tìm thấy câu lệnh SQL trong phản hồi.")
        else:
            st.warning("Không đủ messages để truy cập message thứ hai.")
    except Exception as e:
        st.error(f"Lỗi khi truy cập messages: {e}")