# import importlib
# import warnings
# import pytest
# import time
# import re
# from types import SimpleNamespace
# from pathlib import Path
# import sys

# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.append(str(BASE_DIR))

# # Silence noisy httpx deprecation warnings during tests
# warnings.filterwarnings("ignore", category=DeprecationWarning, module="httpx._models")
# warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")
# pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:httpx._models")


# # ==================== CONFIGURATION ====================

# # Delay giữa các test để tránh rate limit
# TEST_DELAY = 2

# # Số lần retry khi gặp rate limit
# MAX_RETRIES = 3

# # Backoff multiplier (seconds)
# BACKOFF_BASE = 5


# # ==================== HELPER FUNCTIONS ====================

# def run_rag_with_retry(agent, query, retries=MAX_RETRIES):
#     """Chạy RAG agent với retry logic"""
#     for attempt in range(retries):
#         try:
#             result = agent.run(query, stream=False)
#             content = getattr(result, "content", str(result))
#             return {"status": "success", "content": content}
#         except Exception as e:
#             error_msg = str(e).lower()
#             if "rate_limit" in error_msg or "429" in error_msg or "quota" in error_msg:
#                 wait_time = BACKOFF_BASE * (2 ** attempt)
#                 print(f"\n[Retry {attempt + 1}/{retries}] Rate limit hit, waiting {wait_time}s...")
#                 time.sleep(wait_time)
#                 continue
#             return {"status": "error", "message": str(e)}
    
#     return {"status": "error", "message": "Max retries exceeded"}


# def assert_rag_response_format(content):
#     """Kiểm tra response có đúng format KB Hits + Answer"""
#     # Có thể có KB Hits hoặc "No relevant information"
#     has_kb_hits = "KB Hits" in content or "kb hits" in content.lower()
#     has_answer = "Answer" in content or "answer" in content.lower()
#     has_no_info = "no relevant information" in content.lower()
    
#     assert has_kb_hits or has_no_info, f"Response should have 'KB Hits' or 'No relevant information'. Got: {content[:200]}"
#     if has_kb_hits:
#         assert has_answer, f"Response with KB Hits should have 'Answer' section. Got: {content[:200]}"


# @pytest.fixture(autouse=True)
# def add_delay():
#     """Thêm delay giữa các test"""
#     yield
#     time.sleep(TEST_DELAY)


# # ==================== UNIT TESTS (Mocked) ====================

# class TestRAGAgentUnit:
#     """Unit tests với mocked dependencies"""
    
#     def test_load_agent_stubbed(self, monkeypatch):
#         """Test load_agent hoạt động với stubbed components"""
#         import agents.rag_agent as rag

#         class DummyModel:
#             def __init__(self, id=None):
#                 self.id = id or "dummy-model"

#         class DummyAgent:
#             def __init__(self, *_, **__):
#                 self.model = DummyModel()

#             def run(self, query, stream=False):
#                 return SimpleNamespace(
#                     content="**KB Hits**:\n- \"stub quote\" (page 1)\n\n**Answer**:\nstub answer"
#                 )

#         class DummyEmbedder:
#             def __init__(self, *_, **__):
#                 pass

#         class DummyKB:
#             def __init__(self, *_, **__):
#                 pass

#             def load(self, recreate=False):
#                 return None

#         class DummyQdrant:
#             def __init__(self, *_, **__):
#                 pass

#         monkeypatch.setattr(rag, "Gemini", DummyModel)
#         monkeypatch.setattr(rag, "GeminiEmbedder", DummyEmbedder)
#         monkeypatch.setattr(rag, "PDFKnowledgeBase", DummyKB)
#         monkeypatch.setattr(rag, "Qdrant", DummyQdrant)
#         monkeypatch.setattr(rag, "Agent", DummyAgent)

#         importlib.reload(rag)
#         agent, kb = rag.load_agent()

#         assert agent is not None
#         assert hasattr(agent, "run")
#         assert kb is not None
        
#         result = agent.run("test query")
#         assert "KB Hits" in result.content
#         assert "Answer" in result.content

#     def test_response_format_validation(self):
#         """Test helper function validate response format"""
#         # Valid responses
#         valid_kb_response = "**KB Hits**:\n- \"quote\" (page 1)\n\n**Answer**:\nSummary"
#         assert_rag_response_format(valid_kb_response)  # Should not raise
        
#         valid_no_info = "No relevant information found in the document."
#         assert_rag_response_format(valid_no_info)  # Should not raise
        
#         # Invalid response
#         with pytest.raises(AssertionError):
#             assert_rag_response_format("Random text without proper format")


# # ==================== INTEGRATION TESTS (Real API) ====================

# class TestRAGAgentIntegration:
#     """Integration tests với real RAG agent (requires API keys)"""
    
#     @pytest.fixture(scope="class")
#     def rag_agent(self):
#         """Load RAG agent một lần cho tất cả tests trong class"""
#         try:
#             from agents.rag_agent import load_agent
#             agent, kb = load_agent()
#             return agent
#         except Exception as e:
#             pytest.skip(f"Could not load RAG agent: {e}")
    
#     def test_agent_loads_successfully(self, rag_agent):
#         """Test RAG agent load thành công"""
#         assert rag_agent is not None
#         assert hasattr(rag_agent, "run")
#         assert hasattr(rag_agent, "model")
    
#     def test_basic_query(self, rag_agent):
#         """Test query cơ bản về 10-K document"""
#         query = "What is the company name mentioned in this 10-K filing?"
#         result = run_rag_with_retry(rag_agent, query)
        
#         if result["status"] == "error":
#             if "rate_limit" in result.get("message", "").lower():
#                 pytest.skip("Rate limit exceeded")
#             pytest.fail(f"RAG agent error: {result['message']}")
        
#         content = result["content"]
#         print(f"\nQuery: {query}")
#         print(f"Response: {content[:500]}...")
        
#         assert content, "Response should not be empty"
    
#     def test_response_has_kb_hits_format(self, rag_agent):
#         """Test response có đúng format KB Hits + Answer"""
#         query = "What are the main risk factors mentioned in the 10-K?"
#         result = run_rag_with_retry(rag_agent, query)
        
#         if result["status"] == "error":
#             if "rate_limit" in result.get("message", "").lower():
#                 pytest.skip("Rate limit exceeded")
#             pytest.fail(f"RAG agent error: {result['message']}")
        
#         content = result["content"]
#         print(f"\nQuery: {query}")
#         print(f"Response: {content[:500]}...")
        
#         assert_rag_response_format(content)
    
#     def test_specific_page_query(self, rag_agent):
#         """Test query về trang cụ thể"""
#         query = "What information is on page 1 of the document?"
#         result = run_rag_with_retry(rag_agent, query)
        
#         if result["status"] == "error":
#             if "rate_limit" in result.get("message", "").lower():
#                 pytest.skip("Rate limit exceeded")
#             pytest.fail(f"RAG agent error: {result['message']}")
        
#         content = result["content"]
#         print(f"\nQuery: {query}")
#         print(f"Response: {content[:500]}...")
        
#         assert content, "Response should not be empty"
    
#     def test_revenue_query(self, rag_agent):
#         """Test query về revenue/doanh thu"""
#         query = "What does the document say about revenue or sales?"
#         result = run_rag_with_retry(rag_agent, query)
        
#         if result["status"] == "error":
#             if "rate_limit" in result.get("message", "").lower():
#                 pytest.skip("Rate limit exceeded")
#             pytest.fail(f"RAG agent error: {result['message']}")
        
#         content = result["content"]
#         print(f"\nQuery: {query}")
#         print(f"Response: {content[:500]}...")
        
#         assert_rag_response_format(content)
    
#     def test_mda_section_query(self, rag_agent):
#         """Test query về MD&A section"""
#         query = "Summarize the Management Discussion and Analysis (MD&A) section"
#         result = run_rag_with_retry(rag_agent, query)
        
#         if result["status"] == "error":
#             if "rate_limit" in result.get("message", "").lower():
#                 pytest.skip("Rate limit exceeded")
#             pytest.fail(f"RAG agent error: {result['message']}")
        
#         content = result["content"]
#         print(f"\nQuery: {query}")
#         print(f"Response: {content[:500]}...")
        
#         assert content, "Response should not be empty"


# # ==================== RESPONSE STRUCTURE TESTS ====================

# class TestRAGResponseStructure:
#     """Test cấu trúc response của RAG agent"""
    
#     @pytest.fixture(scope="class")
#     def rag_agent(self):
#         """Load RAG agent"""
#         try:
#             from agents.rag_agent import load_agent
#             agent, _ = load_agent()
#             return agent
#         except Exception as e:
#             pytest.skip(f"Could not load RAG agent: {e}")
    
#     def test_kb_hits_has_quotes(self, rag_agent):
#         """Test KB Hits section có quotes trong ngoặc kép"""
#         query = "What are the company's main products or services?"
#         result = run_rag_with_retry(rag_agent, query)
        
#         if result["status"] == "error":
#             pytest.skip(f"RAG error: {result.get('message')}")
        
#         content = result["content"]
        
#         # Tìm phần KB Hits
#         if "KB Hits" in content:
#             # Kiểm tra có quotes (text trong "...")
#             has_quotes = re.search(r'"[^"]+"', content) is not None
#             print(f"\nContent has quoted text: {has_quotes}")
#             print(f"Response preview: {content[:300]}...")
    
#     def test_answer_is_concise(self, rag_agent):
#         """Test Answer section ngắn gọn (2-4 câu)"""
#         query = "What is the fiscal year end date?"
#         result = run_rag_with_retry(rag_agent, query)
        
#         if result["status"] == "error":
#             pytest.skip(f"RAG error: {result.get('message')}")
        
#         content = result["content"]
        
#         # Tách phần Answer
#         if "**Answer**:" in content:
#             answer_part = content.split("**Answer**:")[-1].strip()
#             # Đếm số câu (rough estimate)
#             sentences = len(re.findall(r'[.!?]+', answer_part))
#             print(f"\nAnswer has ~{sentences} sentences")
#             print(f"Answer: {answer_part[:200]}...")


# # ==================== ERROR HANDLING TESTS ====================

# class TestRAGErrorHandling:
#     """Test xử lý lỗi của RAG agent"""
    
#     @pytest.fixture(scope="class")
#     def rag_agent(self):
#         """Load RAG agent"""
#         try:
#             from agents.rag_agent import load_agent
#             agent, _ = load_agent()
#             return agent
#         except Exception as e:
#             pytest.skip(f"Could not load RAG agent: {e}")
    
#     def test_empty_query(self, rag_agent):
#         """Test với query rỗng"""
#         result = run_rag_with_retry(rag_agent, "")
        
#         # Không crash, có response
#         assert result["status"] in ["success", "error"]
    
#     def test_irrelevant_query(self, rag_agent):
#         """Test với query không liên quan đến document"""
#         query = "What is the weather like today?"
#         result = run_rag_with_retry(rag_agent, query)
        
#         if result["status"] == "error":
#             pytest.skip(f"RAG error: {result.get('message')}")
        
#         content = result["content"]
#         print(f"\nIrrelevant query response: {content[:300]}...")
        
#         # Có thể trả về "No relevant information" hoặc cố gắng trả lời
#         assert content, "Should return some response"
    
#     def test_very_long_query(self, rag_agent):
#         """Test với query rất dài"""
#         long_query = "What is " + "the company's revenue " * 50 + "?"
#         result = run_rag_with_retry(rag_agent, long_query)
        
#         # Không crash
#         assert result["status"] in ["success", "error"]


# # ==================== SMOKE TESTS ====================

# class TestRAGSmoke:
#     """Quick smoke tests"""
    
#     def test_rag_agent_importable(self):
#         """Test có thể import RAG agent"""
#         try:
#             from agents.rag_agent import load_agent
#             assert load_agent is not None
#             assert callable(load_agent)
#         except ImportError as e:
#             pytest.fail(f"Cannot import rag_agent: {e}")
    
#     def test_rag_agent_loads(self):
#         """Test RAG agent load được"""
#         try:
#             from agents.rag_agent import load_agent
#             agent, kb = load_agent()
            
#             assert agent is not None
#             assert kb is not None
#             print("\nRAG agent loaded successfully")
#         except Exception as e:
#             pytest.skip(f"Could not load RAG agent (may need API keys): {e}")


# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "-s", "--tb=short"])



# test_agent/test_rag_agent_minimal.py
# Bộ test RAG agent rút gọn – chỉ những phần QUAN TRỌNG NHẤT

import json
import pytest
import re
import time
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from agents.rag_agent import load_agent

# Delay nhẹ giữa các test để tránh rate limit Gemini
TEST_DELAY = 3

@pytest.fixture(scope="module")
def rag_agent():
    """Load agent một lần cho toàn bộ test"""
    try:
        agent, _ = load_agent()
        print("\nRAG agent loaded successfully")
        return agent
    except Exception as e:
        pytest.skip(f"Không load được RAG agent: {e}")

@pytest.fixture(autouse=True)
def delay():
    yield
    time.sleep(TEST_DELAY)

def run_query(agent, query):
    """Chạy query với retry đơn giản khi rate limit"""
    for _ in range(2):  # retry 1 lần
        try:
            result = agent.run(query, stream=False)
            return getattr(result, "content", str(result))
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(10)
                continue
            raise
    raise Exception("Rate limit hoặc lỗi không khắc phục được")

# ==================== CÁC TEST QUAN TRỌNG NHẤT ====================

def test_load_and_basic_response(rag_agent):
    """Test load + trả lời cơ bản"""
    content = run_query(rag_agent, "What is the company name in this 10-K?")
    print(f"\nResponse: {content[:300]}...")
    assert "Apple Inc." in content or "Apple" in content

def test_format_kb_hits_and_answer(rag_agent):
    """Test format nghiêm ngặt: KB Hits + Answer + quotes"""
    content = run_query(rag_agent, "What are the main risk factors?")
    print(f"\nResponse: {content[:500]}...")
    
    assert "**KB Hits**" in content
    assert "**Answer**" in content
    
    # Có ít nhất 2 trích dẫn với dấu ngoặc kép
    quotes = re.findall(r'"([^"]+)"', content)
    assert len(quotes) >= 2, f"Chỉ có {len(quotes)} quotes"
    
    # Có trích dẫn page nếu có
    assert "(page" in content

def test_revenue_query(rag_agent):
    """Test query về số liệu tài chính"""
    content = run_query(rag_agent, "What is the total net sales or revenue?")
    print(f"\nResponse: {content[:400]}...")
    assert any(word in content.lower() for word in ["net sales", "revenue", "$", "million", "billion"])

def test_mda_summary(rag_agent):
    """Test tóm tắt phần quan trọng"""
    content = run_query(rag_agent, "Summarize the MD&A section")
    print(f"\nResponse: {content[:400]}...")
    assert "Management" in content or "MD&A" in content

def test_irrelevant_query(rag_agent):
    """Test query không liên quan → không hallucinate"""
    content = run_query(rag_agent, "What is the weather today?")
    print(f"\nResponse: {content}")
    assert "no relevant" in content.lower() or "cannot" in content.lower() or "sorry" in content.lower()

def test_empty_or_weird_query(rag_agent):
    """Test edge case → không crash"""
    content = run_query(rag_agent, "")
    assert content  # Có trả gì đó

    content = run_query(rag_agent, "abc xyz 123 !!!")
    assert content

print("\n=== RAG Minimal Test Suite Ready ===")