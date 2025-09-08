# rag_agent.py
import os
from phi.agent import Agent
from phi.model.google import Gemini
from phi.embedder.google import GeminiEmbedder
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.qdrant import Qdrant
from phi.storage.agent.sqlite import SqlAgentStorage
from langchain_experimental.text_splitter import SemanticChunker
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
from dotenv import load_dotenv
load_dotenv()

def load_agent():
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    
    embedder = GeminiEmbedder(model="models/embedding-001")
    text_splitter = SemanticChunker(embedder)

    collection_name = "rag_collection"
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    vector_db = Qdrant(
        collection=collection_name,
        url=qdrant_url,
        api_key=qdrant_api_key,
        embedder=embedder
    )

    knowledge_base = PDFKnowledgeBase(
        path=f"{BASE_DIR}/data/2303.08774v6.pdf",
        text_splitter=text_splitter,
        vector_db=vector_db
    )

    knowledge_base.load(recreate=True)

    instructions = [
        "Bạn là một trợ lý RAG chuyên phân tích nội dung từ tài liệu PDF.",
        "Luôn luôn trích dẫn ít nhất 1-3 đoạn văn bản gốc (KB hits) trước khi đưa ra câu trả lời.",
        "Nếu câu hỏi bằng tiếng Việt, hãy dịch sang tiếng Anh để tìm trong KB.",
        "Nếu không có KB hits liên quan, chỉ trả lời: 'Không tìm thấy thông tin liên quan trong tài liệu.'",
        "Định dạng câu trả lời rõ ràng, có tách phần 'KB hits' và phần 'Answer'."
    ]

    agent = Agent(
        model=Gemini(id="gemini-1.5-flash"),
        knowledge=knowledge_base,
        use_knowledge=True,
        retriever_mode="stuff",
        storage=SqlAgentStorage(table_name="rag_agent_sessions", db_file=f"{BASE_DIR}/data/rag_agent.db"),
        show_tool_calls=True,
        markdown=True,
        add_history_to_messages=True,
        instructions=instructions,
        description="RAG Agent với Qdrant + Gemini, luôn trích dẫn KB trước khi trả lời.",
    )
    return agent, knowledge_base
