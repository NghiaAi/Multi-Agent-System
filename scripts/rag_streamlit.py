from pathlib import Path
import sys
import streamlit as st
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from agents.rag_agent import load_agent


@st.cache_resource
def get_agent():
    return load_agent()

def main():
    st.set_page_config(page_title="RAG PDF Assistant", layout="centered")
    st.title("RAG PDF Assistant (Gemini + Qdrant)")

    agent, knowledge_base = get_agent()

    user_input = st.text_area("Nhập câu hỏi:", "")

    if st.button("Hỏi"):
        if user_input.strip() == "":
            st.warning("Bạn chưa nhập câu hỏi.")
        else:
            with st.spinner("Đang tìm câu trả lời..."):
                response = agent.run(user_input, stream=False)
                answer = response.content if hasattr(response, "content") else str(response)

            st.subheader(" Answer:")
            st.markdown(answer)

if __name__ == "__main__":
    main()
