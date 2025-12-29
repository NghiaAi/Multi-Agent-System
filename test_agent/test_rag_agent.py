import json
import pytest
import re
import time
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from agents.rag_agent import load_agent

TEST_DELAY = 3

@pytest.fixture(scope="module")
def rag_agent():
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


def test_irrelevant_query(rag_agent):
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