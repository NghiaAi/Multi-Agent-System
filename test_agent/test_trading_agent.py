import os
import json
import pytest
import warnings
from types import SimpleNamespace
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

os.environ.setdefault("GROQ_API_KEY", "dummy")
warnings.filterwarnings("ignore", category=FutureWarning)

import agents.trading_agent as ta
from agents.text_to_sql_agent import sql_tools


# ==================== SAMPLE DATA ====================

SAMPLE_SQL_DATA = [
    {"Date": "2024-03-15", "Open": 100, "High": 105, "Low": 99, "Close": 103, "Adj Close": 103, "Volume": 1000000, "Ticker": "MSFT"},
    {"Date": "2024-03-14", "Open": 98, "High": 102, "Low": 97, "Close": 100, "Adj Close": 100, "Volume": 900000, "Ticker": "MSFT"},
    {"Date": "2024-03-13", "Open": 95, "High": 99, "Low": 94, "Close": 98, "Adj Close": 98, "Volume": 850000, "Ticker": "MSFT"},
]


# ==================== HELPER ====================

def create_mock_predict(signal="BUY"):
    """Tạo mock predict_signal"""
    def fake_predict_signal(ticker, df):
        return {
            "ticker": ticker,
            "signal": signal,
            "probabilities": {"BUY": 0.9, "HOLD": 0.05, "SELL": 0.05},
            "feature_values": {"Close": 103.0, "Volume": 1000000},
        }
    return fake_predict_signal


def create_mock_explainer():
    """Tạo mock explainer"""
    class DummyExplainer:
        def run(self, payload):
            return SimpleNamespace(content=json.dumps({"explanation": "Mock explanation"}))
    return DummyExplainer()


# ==================== TESTS ====================

class TestTradingAgent:
    """Test Trading Agent"""
    
    def test_buy_signal(self, monkeypatch):
        """Test trading agent trả về BUY signal"""
        monkeypatch.setattr(ta, "predict_signal", create_mock_predict("BUY"))
        monkeypatch.setattr(ta, "create_genai_explainer", lambda features: create_mock_explainer())
        
        result = ta.run_trading_agent("Should I buy MSFT?", sql_result=SAMPLE_SQL_DATA, ticker="MSFT")
        
        assert result["status"] == "success"
        assert result["decision"]["signal"] == "BUY"
        assert result["decision"]["ticker"] == "MSFT"
        assert "explanation" in result["decision"]
        assert "probabilities" in result["decision"]
    
    def test_sell_signal(self, monkeypatch):
        """Test trading agent trả về SELL signal"""
        monkeypatch.setattr(ta, "predict_signal", create_mock_predict("SELL"))
        monkeypatch.setattr(ta, "create_genai_explainer", lambda features: create_mock_explainer())
        
        result = ta.run_trading_agent("Sell MSFT?", sql_result=SAMPLE_SQL_DATA, ticker="MSFT")
        
        assert result["status"] == "success"
        assert result["decision"]["signal"] == "SELL"
    
    def test_no_data_error(self):
        """Test lỗi khi không có data"""
        result = ta.run_trading_agent("Buy?", sql_result=[], ticker="MSFT")
        assert result["status"] == "error"
        assert "No data" in result["message"]
    
    def test_no_ticker_error(self):
        """Test lỗi khi không có ticker"""
        result = ta.run_trading_agent("Buy?", sql_result=SAMPLE_SQL_DATA, ticker="")
        assert result["status"] == "error"
        assert "Ticker" in result["message"]
    
    def test_response_structure(self, monkeypatch):
        """Test cấu trúc response đúng"""
        monkeypatch.setattr(ta, "predict_signal", create_mock_predict())
        monkeypatch.setattr(ta, "create_genai_explainer", lambda features: create_mock_explainer())
        
        result = ta.run_trading_agent("Buy?", sql_result=SAMPLE_SQL_DATA, ticker="MSFT")
        
        # Check structure
        assert "status" in result
        assert "decision" in result
        
        decision = result["decision"]
        assert "ticker" in decision
        assert "signal" in decision
        assert decision["signal"] in ["BUY", "HOLD", "SELL"]
        assert "probabilities" in decision
        assert all(k in decision["probabilities"] for k in ["BUY", "HOLD", "SELL"])
    
    def test_with_real_sql_data(self, monkeypatch):
        """Test với data từ SQL thật"""
        monkeypatch.setattr(ta, "predict_signal", create_mock_predict())
        monkeypatch.setattr(ta, "create_genai_explainer", lambda features: create_mock_explainer())
        
        sql = 'SELECT Date, Open, High, Low, Close, "Adj Close", Volume, Ticker FROM prices WHERE Ticker = \'MSFT\' ORDER BY Date DESC LIMIT 50'
        
        try:
            rows = json.loads(sql_tools.run_sql_query(sql, limit=50))
        except Exception as e:
            pytest.skip(f"Could not fetch SQL data: {e}")
        
        result = ta.run_trading_agent("Buy MSFT?", sql_result=rows, ticker="MSFT")
        assert result["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
