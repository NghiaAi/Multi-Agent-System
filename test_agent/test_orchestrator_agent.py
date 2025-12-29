import json
import pytest
import time
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from agents.orchestrator_agent import run_orchestrator

# ==================== CONFIGURATION ====================

# Delay giữa các test để tránh rate limit (seconds)
TEST_DELAY = 3

# Số lần retry khi gặp rate limit
MAX_RETRIES = 3

# Backoff multiplier (seconds)
BACKOFF_BASE = 10


# ==================== HELPER FUNCTIONS ====================

def run_with_retry(question, chat_history=None, retries=MAX_RETRIES, execute_agents=False):
    """
    Chạy orchestrator với retry logic khi gặp rate limit.
    
    Args:
        question: Câu hỏi cần xử lý
        chat_history: Lịch sử chat
        retries: Số lần retry khi gặp rate limit
        execute_agents: Nếu False (default), chỉ trả về JSON routing mà không chạy agents.
                       Nếu True, chạy đầy đủ các agents.
    """
    if chat_history is None:
        chat_history = []
    
    for attempt in range(retries):
        result = run_orchestrator(question, chat_history=chat_history, execute_agents=execute_agents)
        
        # Kiểm tra rate limit error
        error_msg = result.get("message", "")
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            wait_time = BACKOFF_BASE * (2 ** attempt)
            print(f"\n[Retry {attempt + 1}/{retries}] Rate limit hit, waiting {wait_time}s...")
            time.sleep(wait_time)
            continue
        
        return result
    
    return result  # Return last result even if failed


def assert_agents(expected_agents, result):
    """Kiểm tra danh sách agents trả về đúng"""
    agents = result.get("agents", []) if isinstance(result, dict) else []
    missing = [a for a in expected_agents if a not in agents]
    extra = [a for a in agents if a not in expected_agents]
    assert not missing and not extra, f"Expected {expected_agents}, got {agents}. Missing: {missing}, Extra: {extra}"


def assert_status_ok(result):
    """Kiểm tra status không phải error (cho phép rate limit skip)"""
    if result["status"] == "error":
        error_msg = result.get("message", "")
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            pytest.skip("Rate limit exceeded - skipping test")
        else:
            pytest.fail(f"Orchestrator returned error: {error_msg}")


@pytest.fixture(autouse=True)
def add_delay():
    """Thêm delay giữa các test để tránh rate limit"""
    yield
    time.sleep(TEST_DELAY)


# ==================== TEST ROUTING AGENTS ====================

class TestAgentRouting:
    """Test phân loại đúng agent dựa trên query"""
    
    @pytest.mark.parametrize("question, expected_agents", [
        # 1. Câu hỏi thuần dữ liệu - chỉ text2sql
        ("What was the closing price of Apple on March 15, 2024?", ["text2sql_agent"]),
        
        # 2. Yêu cầu vẽ chart - text2sql + visualization
        ("Plot the cumulative returns of Microsoft in 2024", ["text2sql_agent", "visualization_agent"]),
        
        # 3. Yêu cầu trading decision - text2sql + trading
        ("Should I buy MSFT now?", ["text2sql_agent", "trading_agent"]),
        
        # 4. Câu hỏi về PDF/10-K - chỉ rag_agent
        ("According to Apple's 10-K, what are the main risk factors?", ["rag_agent"]),
        
        # 5. Câu hỏi kết hợp
        ("Plot Apple's price in 2024 and tell me if I should buy it", ["text2sql_agent", "visualization_agent", "trading_agent"]),
    ])


    def test_orchestrator_routing(self, question, expected_agents):
        """Test routing cơ bản - phân loại đúng agent"""
        print(f"\n=== Question: {question} ===")
        result = run_with_retry(question)
        
        print(f"Status: {result.get('status')}")
        print(f"Returned agents: {result.get('data', {}).get('agents', [])}")
        print(f"Tickers: {result.get('data', {}).get('tickers', [])}")
        
        assert_status_ok(result)
        assert_agents(expected_agents, result["data"])


# ==================== TEST TICKER EXTRACTION ====================

class TestTickerExtraction:
    """Test trích xuất ticker từ query"""
    
    @pytest.mark.parametrize("question, expected_tickers", [
        # Ticker trực tiếp
        ("What is the price of AAPL?", ["AAPL"]),
        
        # Tên công ty đầy đủ
        ("What is Apple's stock price?", ["AAPL"]),
    ])
    def test_ticker_extraction(self, question, expected_tickers):
        """Test trích xuất ticker từ tên công ty hoặc mã trực tiếp"""
        result = run_with_retry(question)
        
        assert_status_ok(result)
        tickers = result.get("data", {}).get("tickers", [])
        print(f"Query: {question}")
        print(f"Expected: {expected_tickers}, Got: {tickers}")
        
        for ticker in expected_tickers:
            assert ticker in tickers, f"Missing ticker {ticker} in result {tickers}"


# ==================== TEST DATE RANGE EXTRACTION ====================

class TestDateRangeExtraction:
    """Test trích xuất date range từ query"""
    
    @pytest.mark.parametrize("question, expected_start, expected_end", [
        # Năm cụ thể
        ("Apple stock price in 2024", "2024-01-01", "2024-12-31"),
        ("Plot MSFT prices during 2023", "2023-01-01", "2023-12-31"),
    ])
    def test_date_range_extraction(self, question, expected_start, expected_end):
        """Test trích xuất date range từ query"""
        result = run_with_retry(question)
        
        assert_status_ok(result)
        date_range = result.get("data", {}).get("date_range")
        
        print(f"Query: {question}")
        print(f"Expected: {expected_start} - {expected_end}")
        print(f"Got: {date_range}")
        
        assert date_range is not None, "Date range should be extracted"
        assert date_range.get("start_date") == expected_start, f"Start date mismatch"
        assert date_range.get("end_date") == expected_end, f"End date mismatch"
    
    def test_no_date_range(self):
        """Test query không có date range"""
        question = "What is the current price of Apple?"
        result = run_with_retry(question)
        
        assert_status_ok(result)
        date_range = result.get("data", {}).get("date_range")
        assert date_range is None, f"Expected no date_range but got {date_range}"


# ==================== TEST CHAT HISTORY CONTEXT ====================

class TestChatHistoryContext:
    """Test sử dụng ngữ cảnh từ chat history"""
    
    def test_pronoun_resolution(self):
        """Test giải quyết đại từ 'its', 'it' từ lịch sử chat"""
        chat_history = [
            {"role": "user", "content": "Tell me about Microsoft stock"},
            {"role": "assistant", "content": "Microsoft (MSFT) is a technology company..."}
        ]
        
        result = run_with_retry("Plot its prices in 2024", chat_history=chat_history)
        
        assert_status_ok(result)
        tickers = result.get("data", {}).get("tickers", [])
        print(f"Resolved tickers: {tickers}")
        
        # Phải nhận ra 'its' là Microsoft
        assert "MSFT" in tickers, f"Should resolve 'its' to MSFT, got {tickers}"
    
    def test_empty_chat_history(self):
        """Test với chat history rỗng"""
        result = run_with_retry("What is Apple stock price?", chat_history=[])
        
        assert_status_ok(result)
        assert "text2sql_agent" in result.get("data", {}).get("agents", [])


# ==================== TEST SUB-QUERY GENERATION ====================

class TestSubQueryGeneration:
    """Test tạo sub-query đúng cho từng agent"""
    
    def test_text2sql_factual_query(self):
        """Test text2sql giữ nguyên câu hỏi factual"""
        question = "What was the highest closing price of Apple in 2024?"
        result = run_with_retry(question)
        
        assert_status_ok(result)
        sub_queries = result.get("data", {}).get("sub_queries", {})
        
        sql_query = sub_queries.get("text2sql_agent", "")
        print(f"Original: {question}")
        print(f"Sub-query: {sql_query}")
        
        # Query phải chứa các từ khóa chính
        assert "highest" in sql_query.lower() or "closing" in sql_query.lower() or "price" in sql_query.lower()
    
    def test_visualization_data_fetch_query(self):
        """Test sub-query cho visualization phải fetch data đúng"""
        question = "Plot the cumulative returns of MSFT in 2024"
        result = run_with_retry(question)
        
        assert_status_ok(result)
        sub_queries = result.get("data", {}).get("sub_queries", {})
        
        # text2sql sub-query phải là fetch prices
        sql_query = sub_queries.get("text2sql_agent", "")
        print(f"Text2SQL sub-query: {sql_query}")
        assert sql_query, "Text2SQL sub-query should not be empty"
        
        # visualization sub-query
        viz_query = sub_queries.get("visualization_agent", "")
        print(f"Visualization sub-query: {viz_query}")
        assert viz_query, "Visualization sub-query should not be empty"


# ==================== TEST ERROR HANDLING ====================

class TestErrorHandling:
    """Test xử lý lỗi và edge cases (routing only)"""
    
    def test_empty_query(self):
        """Test với query rỗng"""
        result = run_orchestrator("", chat_history=[], execute_agents=False)
        
        # Quan trọng là không crash
        assert "status" in result
    
    def test_gibberish_query(self):
        """Test với query vô nghĩa"""
        result = run_orchestrator("asdfghjkl zxcvbnm", chat_history=[], execute_agents=False)
        
        assert "status" in result
    
    def test_special_characters(self):
        """Test với ký tự đặc biệt"""
        query = "What is Apple 52-week high?"
        result = run_with_retry(query, execute_agents=False)
        
        # Không crash, có thể thành công hoặc skip rate limit
        assert "status" in result


# ==================== TEST RESPONSE STRUCTURE ====================

class TestResponseStructure:
    """Test cấu trúc response đúng format"""
    
    def test_response_has_required_fields(self):
        """Test response có đủ các field bắt buộc"""
        result = run_with_retry("What is Apple stock price?")
        
        assert_status_ok(result)
        
        # Top level
        assert "status" in result
        assert "message" in result
        assert "data" in result
        
        # Data level
        data = result.get("data", {})
        assert "agents" in data
        assert "sub_queries" in data
        assert "tickers" in data
        assert "date_range" in data
    
    def test_agents_is_list(self):
        """Test agents là list"""
        result = run_with_retry("Plot AAPL prices")
        
        assert_status_ok(result)
        
        agents = result.get("data", {}).get("agents", [])
        assert isinstance(agents, list), f"agents should be list, got {type(agents)}"
    
    def test_sub_queries_is_dict(self):
        """Test sub_queries là dict"""
        result = run_with_retry("Plot AAPL prices")
        
        assert_status_ok(result)
        
        sub_queries = result.get("data", {}).get("sub_queries", {})
        assert isinstance(sub_queries, dict), f"sub_queries should be dict, got {type(sub_queries)}"


# ==================== TEST EDGE CASES ====================

class TestEdgeCases:
    """Test các trường hợp đặc biệt"""
    
    def test_case_insensitivity(self):
        """Test không phân biệt hoa thường"""
        question = "PLOT aapl PRICES"
        result = run_with_retry(question)
        
        assert_status_ok(result)
        agents = result.get("data", {}).get("agents", [])
        assert "visualization_agent" in agents
    
    def test_rag_with_filing_keyword(self):
        """Test RAG với từ khóa filing"""
        question = "According to the filing, what is Apple's revenue breakdown?"
        result = run_with_retry(question)
        
        assert_status_ok(result)
        agents = result.get("data", {}).get("agents", [])
        assert "rag_agent" in agents


# ==================== SMOKE TEST ====================

class TestSmoke:
    """Quick smoke tests để verify basic functionality (routing only, no agent execution)"""
    
    def test_basic_query(self):
        """Test một query đơn giản hoạt động"""
        result = run_with_retry("What is AAPL price?", execute_agents=False)
        
        assert "status" in result
        assert "data" in result
        
        if result["status"] != "error":
            assert result.get("data", {}).get("agents")
    
    def test_orchestrator_does_not_crash(self):
        """Test orchestrator không crash với input hợp lệ"""
        queries = [
            "Price of Apple",
            "Plot MSFT",
            "Buy AAPL?",
            "10-K report"
        ]
        
        for q in queries:
            result = run_orchestrator(q, chat_history=[], execute_agents=False)
            assert "status" in result, f"Query '{q}' should return valid response"
            time.sleep(1)  


# ==================== INTEGRATION TEST (Full Execution) ====================

class TestIntegration:
    @pytest.mark.slow
    def test_full_text2sql_execution(self):
        result = run_with_retry("What is the closing price of AAPL?", execute_agents=True)
        
        assert_status_ok(result)

        sql_result = result.get("data", {}).get("sql_result")
        assert sql_result is not None, "sql_result should be present when execute_agents=True"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
