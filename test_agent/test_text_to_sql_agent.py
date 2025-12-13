from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
import json
import pytest
from agents.text_to_sql_agent import agent as sql_agent
from agents.text_to_sql_agent import sql_tools  


@pytest.fixture(autouse=True)
def clear_memory():
    """Xóa memory trước và sau mỗi test"""
    sql_agent.memory.clear()
    yield
    sql_agent.memory.clear()


def extract_generated_sql():
    """Trích xuất SQL từ tool call (tương thích phi 2.x+)"""
    for msg in reversed(sql_agent.memory.messages):
        if msg.role == "assistant" and msg.tool_calls:
            tool_call = msg.tool_calls[0]
            try:
                args_str = tool_call["function"]["arguments"]
                args = json.loads(args_str)
                return args["query"].strip()
            except Exception:
                continue
    return None


def test_count_trading_days_boeing_2024():
    question = "How many trading days did Boeing have in 2024?"
    print(f"\n=== Testing: {question} ===")
    
    sql_agent.run(question, stream=False)
    
    sql = extract_generated_sql()
    assert sql is not None, "Không tìm thấy SQL"
    print(f"Generated SQL:\n{sql}")
    
    sql_upper = sql.upper()
    assert "COUNT(*)" in sql_upper
    assert "FROM PRICES" in sql_upper
    assert "TICKER" in sql_upper and "'BA'" in sql
    assert "STRFTIME" in sql_upper and "%Y" in sql
    
    rows = json.loads(sql_tools.run_sql_query(sql))
    assert len(rows) == 1
    count = list(rows[0].values())[0]
    assert isinstance(count, int)
    assert 200 <= count <= 260


def test_sector_of_ibm():
    question = "What is the sector of IBM?"
    print(f"\n=== Testing: {question} ===")
    
    sql_agent.run(question, stream=False)
    
    sql = extract_generated_sql()
    assert sql is not None
    print(f"Generated SQL:\n{sql}")
    
    sql_upper = sql.upper()
    assert "FROM COMPANIES" in sql_upper
    assert "SECTOR" in sql_upper
    # Linh hoạt: dùng ticker hoặc tên đầy đủ
    assert ("IBM" in sql) or ("International Business Machines" in sql)
    
    rows = json.loads(sql_tools.run_sql_query(sql))
    assert len(rows) >= 1, "Không có kết quả trả về cho sector của IBM"
    first_row = rows[0]
    assert "sector" in first_row, "Kết quả phải có cột 'sector'"
    sector_value = first_row["sector"]
    assert isinstance(sector_value, str) and len(sector_value.strip()) > 0
    print(f"IBM sector: {sector_value}")


def test_plot_cumulative_returns_microsoft_2024():
    question = "Plot cumulative returns of Microsoft in 2024"
    print(f"\n=== Testing: {question} ===")
    
    sql_agent.run(question, stream=False)
    
    sql = extract_generated_sql()
    assert sql is not None
    print(f"Generated SQL:\n{sql}")
    
    sql_upper = sql.upper()
    assert "DATE" in sql_upper
    assert "CLOSE" in sql_upper
    assert "TICKER" in sql_upper
    assert "FROM PRICES" in sql_upper
    assert ("MSFT" in sql) or ("Microsoft" in sql)
    
    rows = json.loads(sql_tools.run_sql_query(sql))
    assert len(rows) > 50
    assert {"Date", "Close", "Ticker"}.issubset(rows[0].keys())


def test_boxplot_market_cap_by_sector():
    question = "Show me a boxplot of market capitalization by sector"
    print(f"\n=== Testing: {question} ===")
    
    sql_agent.run(question, stream=False)
    
    sql = extract_generated_sql()
    assert sql is not None
    print(f"Generated SQL:\n{sql}")
    
    sql_lower = sql.lower()
    assert "sector" in sql_lower
    assert "market_cap" in sql_lower
    assert "from companies" in sql_lower
    
    rows = json.loads(sql_tools.run_sql_query(sql))
    assert len(rows) == 30
    assert "sector" in rows[0] and "market_cap" in rows[0]


def test_standard_deviation_aapl_2024():
    question = "What is the standard deviation of daily closing prices for AAPL in 2024?"
    print(f"\n=== Testing: {question} ===")
    
    sql_agent.run(question, stream=False)
    
    sql = extract_generated_sql()
    assert sql is not None
    print(f"Generated SQL:\n{sql}")
    
    sql_upper = sql.upper()
    assert "SQRT(" in sql_upper
    assert "AVG(" in sql_upper
    assert "STDEV" not in sql_upper
    assert "AAPL" in sql
    
    rows = json.loads(sql_tools.run_sql_query(sql))
    assert len(rows) == 1
    value = list(rows[0].values())[0]
    assert isinstance(value, (int, float))


def test_no_forbidden_functions():
    """Kiểm tra không dùng hàm cấm trong tất cả các SQL đã sinh"""
    forbidden = ["STDEV(", "STDDEV(", "VARIANCE(", "COVAR_POP(", "CORR("]
    all_sql = []
    
    for msg in sql_agent.memory.messages:
        if msg.role == "assistant" and msg.tool_calls:
            try:
                tool_call = msg.tool_calls[0]
                args_str = tool_call["function"]["arguments"]
                query = json.loads(args_str)["query"].upper()
                all_sql.append(query)
            except:
                continue
    
    for sql in all_sql:
        for f in forbidden:
            assert f not in sql, f"Dùng hàm cấm {f} trong: {sql}"