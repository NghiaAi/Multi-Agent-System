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
    embedder = GeminiEmbedder(model="models/text-embedding-004")
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
        path=f"{BASE_DIR}/data/10-K-2025.pdf",
        text_splitter=text_splitter,
        vector_db=vector_db
    )

    knowledge_base.load(recreate=False)

    instructions = [
        "You are a RAG assistant specialized in analyzing content from PDF documents.",
        "Always search the knowledge base for relevant information using the provided query.",
        "Return output in the document's language.",
        "Strictly provide a 'KB Hits' section with 2–5 verbatim quotes from the document (no paraphrasing). Enclose each quote in double quotes \"...\".",
        "When available, append page info in parentheses, e.g., (page 12).",
        "After 'KB Hits', provide an 'Answer' section with a concise, easy-to-understand summary (2–4 sentences).",
        "If no relevant KB hits are found, return only: 'No relevant information found in the document.'",
        "Response format:\n\n**KB Hits**:\n- \"[verbatim quote]\" (page X)\n- \"[verbatim quote]\" (page Y)\n\n**Answer**:\n[short, clear summary]",
    ]

    agent = Agent(
        model=Gemini(id="gemini-2.5-flash"),
        knowledge=knowledge_base,
        use_knowledge=True,
        retriever_mode="stuff",
        storage=SqlAgentStorage(table_name="rag_agent_sessions", db_file=f"{BASE_DIR}/data/rag_agent.db"),
        show_tool_calls=True,
        markdown=True,
        add_history_to_messages=True,
        instructions=instructions,
        description="RAG Agent with Qdrant + Gemini, always citing KB before responding.",
    )
    return agent, knowledge_base