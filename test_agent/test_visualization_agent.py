import json
import pytest
import warnings
import numpy as np
from pathlib import Path
import sys

# Set matplotlib backend to Agg (non-interactive)
import matplotlib
matplotlib.use('Agg')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import agents.visualization_agent as viz
from agents.text_to_sql_agent import sql_tools


# ==================== SAMPLE DATA ====================

SAMPLE_PRICE_DATA = [
    {"Date": "2024-03-15", "Close": 103.0, "Ticker": "MSFT", "Volume": 1000000},
    {"Date": "2024-03-14", "Close": 100.0, "Ticker": "MSFT", "Volume": 900000},
    {"Date": "2024-03-13", "Close": 98.0, "Ticker": "MSFT", "Volume": 850000},
]

SAMPLE_SECTOR_DATA = [
    {"sector": "Technology", "count": 8},
    {"sector": "Healthcare", "count": 4},
    {"sector": "Financial", "count": 6},
]


# ==================== HELPER ====================

def create_mock_viz_agent(chart_type="simple"):
    """Tạo mock visualization agent"""
    code_templates = {
        "simple": """
```python
import numpy as np
df = pd.DataFrame(sql_data)
img = np.zeros((100, 100, 3), dtype=np.uint8)
```
""",
        "line": """
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame(sql_data)
plt.figure(figsize=(10, 6))
plt.plot(range(len(df)), df['Close'])
plt.title('Price')

fig = plt.gcf()
fig.canvas.draw()
img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
plt.close(fig)
```
""",
    }
    
    class DummyAgent:
        def run(self, _input):
            return type("Resp", (), {"content": code_templates.get(chart_type, code_templates["simple"])})
    
    return DummyAgent()


# ==================== TESTS ====================

class TestVisualizationAgent:
    """Test Visualization Agent"""
    
    def test_returns_image(self, monkeypatch):
        """Test visualization agent trả về image"""
        monkeypatch.setattr(viz, "create_visualize_agent", lambda: create_mock_viz_agent("simple"))
        
        result = viz.run_visualize_agent("Plot MSFT", sql_data=SAMPLE_PRICE_DATA)
        
        assert result["status"] == "success"
        assert result["visualization"] is not None
        assert isinstance(result["visualization"], np.ndarray)
    
    def test_image_is_rgb(self, monkeypatch):
        """Test image có 3 channels (RGB)"""
        monkeypatch.setattr(viz, "create_visualize_agent", lambda: create_mock_viz_agent("simple"))
        
        result = viz.run_visualize_agent("Plot", sql_data=SAMPLE_PRICE_DATA)
        
        assert result["status"] == "success"
        assert result["visualization"].shape[2] == 3  # RGB
    
    def test_line_chart(self, monkeypatch):
        """Test tạo line chart"""
        monkeypatch.setattr(viz, "create_visualize_agent", lambda: create_mock_viz_agent("line"))
        
        result = viz.run_visualize_agent("Line chart MSFT", sql_data=SAMPLE_PRICE_DATA)
        
        assert result["status"] == "success"
        assert result["visualization"] is not None
    
    def test_no_code_block_error(self, monkeypatch):
        """Test lỗi khi không có code block"""
        class NoCodeAgent:
            def run(self, _input):
                return type("Resp", (), {"content": "Just text, no code"})
        
        monkeypatch.setattr(viz, "create_visualize_agent", lambda: NoCodeAgent())
        
        result = viz.run_visualize_agent("Plot", sql_data=SAMPLE_PRICE_DATA)
        
        assert result["status"] == "error"
        assert "No python code block" in result["message"]
    
    def test_response_structure(self, monkeypatch):
        """Test cấu trúc response"""
        monkeypatch.setattr(viz, "create_visualize_agent", lambda: create_mock_viz_agent())
        
        result = viz.run_visualize_agent("Plot", sql_data=SAMPLE_PRICE_DATA)
        
        assert "status" in result
        assert "message" in result
        assert "visualization" in result
    
    def test_with_real_sql_data(self, monkeypatch):
        """Test với data từ SQL thật"""
        monkeypatch.setattr(viz, "create_visualize_agent", lambda: create_mock_viz_agent("line"))
        
        sql = "SELECT Date, Close, Ticker FROM prices WHERE Ticker = 'MSFT' ORDER BY Date DESC LIMIT 30"
        
        try:
            rows = json.loads(sql_tools.run_sql_query(sql, limit=30))
        except Exception as e:
            pytest.skip(f"Could not fetch SQL data: {e}")
        
        result = viz.run_visualize_agent("Plot MSFT", sql_data=rows)
        assert result["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
