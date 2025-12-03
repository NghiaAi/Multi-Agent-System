# import os
# from phi.agent import Agent
# from phi.model.groq import Groq
# from sentence_transformers import SentenceTransformer
# from phi.knowledge.pdf import PDFKnowledgeBase
# from phi.vectordb.qdrant import Qdrant
# from phi.storage.agent.sqlite import SqlAgentStorage
# from langchain_experimental.text_splitter import SemanticChunker
# from pathlib import Path
# BASE_DIR = Path(__file__).resolve().parent.parent
# from dotenv import load_dotenv
# load_dotenv()
# import logging
# from typing import List, Any
# from phi.embedder.base import Embedder

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class SentenceTransformerEmbedder(Embedder):
#     model: Any = None

#     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
#         super().__init__()
#         logger.info(f"Initializing SentenceTransformer with model: {model_name}")
#         object.__setattr__(self, "model", SentenceTransformer(model_name))
#         dims = self.model.get_sentence_embedding_dimension()
#         object.__setattr__(self, "dimensions", dims)
#         logger.info(f"Embedding dimensions: {dims}")

#     def get_embedding(self, text: str):
#         """Generate embedding for the given text."""
#         embedding = self.model.encode(text, convert_to_numpy=True).tolist()
#         logger.debug(f"Generated embedding for text of length {len(text)} tokens")
#         return embedding

#     def get_embedding_and_usage(self, text: str):
#         """Generate embedding and usage info for the given text."""
#         embedding = self.model.encode(text, convert_to_numpy=True).tolist()
#         usage = {"input_tokens": len(text.split()), "output_tokens": 0}
#         logger.debug(f"Generated embedding for text of length {len(text)} tokens")
#         return embedding, usage

#     def get_dimension(self):
#         return self.dimensions

#     # LangChain Embeddings API compatibility for SemanticChunker
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return self.model.encode(texts, convert_to_numpy=True).tolist()

#     def embed_query(self, text: str) -> List[float]:
#         return self.model.encode(text, convert_to_numpy=True).tolist()

# def load_agent():
#     os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#     model_name = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
#     embedder = SentenceTransformerEmbedder(model_name=model_name)
#     text_splitter = SemanticChunker(embedder)

#     # Separate collection per embedding model & dimension to avoid vector-size mismatch
#     safe_model_name = model_name.lower().replace("/", "_").replace("-", "_")
#     collection_name = f"rag_collection_{safe_model_name}_{embedder.get_dimension()}"
#     qdrant_url = os.getenv("QDRANT_URL")
#     qdrant_api_key = os.getenv("QDRANT_API_KEY")

#     vector_db = Qdrant(
#         collection=collection_name,
#         url=qdrant_url,
#         api_key=qdrant_api_key,
#         embedder=embedder,
#         timeout=120 
#     )

#     # Verify that the PDF file exists
#     pdf_path = BASE_DIR / "data" / "2303.08774v6.pdf"
#     if not pdf_path.exists():
#         logger.error(f"PDF file not found at: {pdf_path}")
#         raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

#     knowledge_base = PDFKnowledgeBase(
#         path=str(pdf_path),
#         text_splitter=text_splitter,
#         vector_db=vector_db
#     )

#     # Load knowledge base, allow force re-create via env
#     recreate = os.getenv("RAG_RECREATE", "false").lower() == "true"
#     knowledge_base.load(recreate=recreate)

#     instructions = [
        # "You are a RAG assistant specialized in analyzing content from PDF documents.",
        # "Always search the knowledge base for relevant information using the provided query.",
        # "Return output in the document's language.",
        # "Strictly provide a 'KB Hits' section with 2–5 verbatim quotes from the document (no paraphrasing). Enclose each quote in double quotes \"...\".",
        # "When available, append page info in parentheses, e.g., (page 12).",
        # "After 'KB Hits', provide an 'Answer' section with a concise, easy-to-understand summary (2–4 sentences).",
        # "Do NOT include any tool calls or function markup in the output.",
        # "If no relevant KB hits are found, return only: 'No relevant information found in the document.'",
        # "Response format:\n\n**KB Hits**:\n- \"[verbatim quote]\" (page X)\n- \"[verbatim quote]\" (page Y)\n\n**Answer**:\n[short, clear summary]",
#     ]

#     agent = Agent(
#         model=Groq(id="llama-3.3-70b-versatile",
#             timeout=30,
#             max_retries=5,
#             temperature=0.2,
#             max_tokens=1000,
#             top_p=0.8,),
#         knowledge=knowledge_base,
#         use_knowledge=True,
#         retriever_mode="stuff",
#         storage=SqlAgentStorage(table_name="rag_agent_sessions", db_file=f"{BASE_DIR}/data/rag_agent.db"),
#         show_tool_calls=False,
#         markdown=True,
#         add_history_to_messages=True,
#         instructions=instructions,
#         description="RAG Agent with Qdrant + Groq, always citing KB before responding.",
#     )
#     return agent, knowledge_base

# def search_kb(query: str, top_k: int = 5) -> List[dict]:
#     agent, knowledge_base = load_agent()
#     docs = knowledge_base.search(query=query, num_documents=top_k)
#     hits = []
#     for d in docs:
#         content = getattr(d, 'content', None) or getattr(d, 'page_content', None) or getattr(d, 'text', '')
#         meta = getattr(d, 'metadata', None) or getattr(d, 'meta', {})
#         if isinstance(meta, dict):
#             page = meta.get('page') or meta.get('page_number') or meta.get('loc', {}).get('page') if isinstance(meta.get('loc'), dict) else None
#         else:
#             page = None
#         hits.append({
#             "text": content,
#             "page": page,
#             "metadata": meta if isinstance(meta, dict) else {}
#         })
#     return hits


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
        path=f"{BASE_DIR}/data/2303.08774v6.pdf",
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