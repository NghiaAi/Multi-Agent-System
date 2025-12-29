# 🧠 Multi-Agent System

Hệ thống **Multi-Agent System** được xây dựng bằng [**phidata**](https://github.com/phidatahq/phidata), tích hợp nhiều agent chuyên biệt để:
- Truy vấn dữ liệu tài chính (Dow Jones Industrial Average – DJIA) bằng SQL
- Tạo biểu đồ và visualization từ dữ liệu tài chính
- Đưa ra khuyến nghị giao dịch (Buy/Sell/Hold) dựa trên mô hình ML
- Phân tích và trả lời câu hỏi từ tài liệu PDF với RAG (Gemini + Qdrant)
- Điều phối các agent thông qua **Orchestrator Agent**

---

## 🚀 Tính năng chính

### **Phidata-based Agents**
Các agent trong hệ thống được định nghĩa và điều phối bởi framework **phidata**, giúp dễ dàng:
- Quản lý nhiều agent độc lập
- Xây dựng pipeline trả lời phức tạp
- Lưu lịch sử hội thoại và state của agent
- Tích hợp với các LLM (Groq, Gemini)

### **Text-to-SQL Agent**
- Hiểu câu hỏi tài chính (giá, khối lượng, cổ tức, returns, v.v.)
- Chuyển đổi câu hỏi tự nhiên sang SQL để truy vấn database `djia.db`
- Tự động ánh xạ tên công ty sang ticker (VD: *Apple Inc.* → `AAPL`)
- Hỗ trợ các truy vấn phức tạp: cumulative returns, statistical analysis, sector distribution
- Tự động giới hạn kết quả dựa trên token limit để tối ưu hiệu suất

### **Visualization Agent**
- Tạo các biểu đồ chuyên nghiệp từ dữ liệu SQL
- Hỗ trợ nhiều loại biểu đồ:
  - **Line charts**: Giá cổ phiếu theo thời gian, cumulative returns
  - **Pie charts**: Phân bố theo sector, industry
  - **Bar charts**: So sánh giá trị giữa các công ty
  - **Boxplots**: Phân bố giá theo tháng/năm
  - **Heatmaps**: Correlation matrix của daily returns
  - **Histograms**: Phân bố returns
  - **Scatter plots**: Mối quan hệ giữa các biến
- Sử dụng **matplotlib** và **seaborn** để tạo visualization chất lượng cao
- Tự động xử lý dữ liệu và format biểu đồ

### **Trading Agent**
- Đưa ra khuyến nghị giao dịch (BUY/HOLD/SELL) dựa trên mô hình ML
- Sử dụng **XGBoost** với 50+ features kỹ thuật:
  - Technical indicators: RSI, MACD, Stochastic, ADX
  - Moving averages: EMA, SMA crossovers
  - Volatility metrics: ATR, Bollinger Bands
  - Volume analysis: VWAP, MFI, volume momentum
  - Price patterns: gaps, higher highs, lower lows
- Mô hình được huấn luyện riêng cho từng ticker DJIA (30 mô hình)
- Giải thích quyết định bằng GenAI dựa trên feature values
- Hiển thị xác suất cho từng signal (BUY/HOLD/SELL)

### **RAG Agent**
- Truy xuất kiến thức từ PDF documents (10-K filings, research papers)
- Sử dụng **Gemini** để sinh embedding + **Qdrant** làm vector database
- Luôn trích dẫn các đoạn văn bản liên quan (KB hits) trước khi trả lời
- Hỗ trợ phân tích tài liệu tài chính phức tạp

### **Orchestrator Agent**
- Phân tích query người dùng và quyết định routing
- Hỗ trợ gọi nhiều agent tuần tự (VD: text2sql → visualization)
- Xử lý ngữ cảnh từ chat history (5 tin nhắn gần nhất)
- Tự động extract tickers và date ranges từ query
- Tạo sub-queries tối ưu cho từng agent

### **Streamlit Apps**
- `app.py`: Giao diện web tích hợp tất cả agents với chat interface
- `djia_streamlit.py`: Giao diện riêng cho Text-to-SQL Agent
- `rag_streamlit.py`: Giao diện riêng cho RAG Agent

---

## 🗂 Cấu trúc thư mục

```
Multi_Agent/
├── agents/
│   ├── orchestrator_agent.py    # Orchestrator Agent (phidata)
│   ├── rag_agent.py             # RAG Agent (Gemini + Qdrant, phidata)
│   ├── text_to_sql_agent.py     # Text-to-SQL Agent (DJIA DB, phidata)
│   ├── visualization_agent.py   # Visualization Agent (matplotlib/seaborn)
│   └── trading_agent.py          # Trading Agent (XGBoost + GenAI)
├── scripts/
│   ├── create_db.py             # Tạo database từ CSV
│   ├── app.py                   # Giao diện Streamlit tích hợp
│   ├── djia_streamlit.py        # Giao diện Streamlit cho DJIA
│   ├── rag_streamlit.py         # Giao diện Streamlit cho RAG
│   ├── Train.ipynb              # Notebook huấn luyện trading models
│   └── download_data.ipynb      # Notebook tải dữ liệu
├── data/
│   ├── djia_companies_*.csv     # Dữ liệu công ty DJIA
│   ├── djia_prices_*.csv        # Dữ liệu giá cổ phiếu
│   ├── djia.db                  # SQLite database
│   ├── rag_agent.db             # RAG vector database
│   ├── 10-K-2025.pdf            # SEC filing document
│   └── 2303.08774v6.pdf         # Research paper
├── models/
│   ├── AAPL_model.joblib        # Trading models cho từng ticker
│   ├── MSFT_model.joblib
│   └── ...                      # (30 models total)
├── test_agent/                  # Test scripts cho các agents
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/NghiaAi/Multi-Agent-System.git
cd Multi-Agent-System
```

### 2. Tạo virtual environment & cài dependencies
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Tạo file `.env`
Tạo file `.env` trong thư mục gốc với nội dung:
```env
# API keys
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key

# Qdrant (Vector Database)
QDRANT_URL=https://your-qdrant-url
QDRANT_API_KEY=your_qdrant_api_key
```

**Lấy API keys:**
- **GROQ_API_KEY**: Đăng ký tại [console.groq.com](https://console.groq.com)
- **GOOGLE_API_KEY**: Đăng ký tại [Google AI Studio](https://makersuite.google.com/app/apikey)
- **QDRANT**: Sử dụng [Qdrant Cloud](https://cloud.qdrant.io) hoặc self-hosted

### 4. Khởi tạo database
```bash
python scripts/create_db.py
```

Lệnh này sẽ:
- Tạo SQLite database `data/djia.db` từ CSV files
- Tạo các bảng `companies` và `prices`
- Index các cột quan trọng để tối ưu truy vấn

### 5. (Tùy chọn) Huấn luyện Trading Models
Nếu bạn muốn retrain các trading models:
```bash
# Mở Jupyter notebook
jupyter notebook scripts/Train.ipynb
```

---

## ▶️ Chạy ứng dụng

### 1. Giao diện tích hợp (Khuyến nghị)
Chạy ứng dụng Streamlit với tất cả agents:
```bash
streamlit run scripts/app.py
```
👉 Mở trình duyệt tại [http://localhost:8501](http://localhost:8501)

### 2. Giao diện riêng lẻ
```bash
# Chỉ Text-to-SQL Agent
streamlit run scripts/djia_streamlit.py

# Chỉ RAG Agent
streamlit run scripts/rag_streamlit.py
```

---

## 🏗 Kiến trúc hệ thống

```
                  ┌─────────────────┐
                  │  User Query      │
                  └───────┬─────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  Orchestrator Agent │  (phidata + Groq)
                │  - Intent Detection │
                │  - Agent Routing    │
                │  - Context Handling │
                └───────┬────────────┘
                        │
        ┌───────────────┼───────────────┬──────────────┐
        │               │               │              │
        ▼               ▼               ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Text-to-SQL  │ │Visualization │ │  Trading     │ │   RAG        │
│    Agent     │ │    Agent     │ │   Agent      │ │   Agent      │
│              │ │              │ │              │ │              │
│ - SQL Query  │ │ - Charts     │ │ - ML Models  │ │ - PDF Search │
│ - Data Fetch │ │ - Graphs     │ │ - Signals    │ │ - Embeddings │
│ - Mapping    │ │ - Plots      │ │ - Explain    │ │ - Qdrant     │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Final Response │
              └─────────────────┘
```

---

## 📊 Ví dụ sử dụng

### 1. Query dữ liệu tài chính (Text-to-SQL Agent)

**Input**:  
```
What was the closing price of Apple on 2024-03-15?
```

**Output**:
```
SQL Query: SELECT Close FROM prices WHERE Ticker = "AAPL" AND DATE(Date) = "2024-03-15"
Raw Result: [{"Close": 155.23}]
Answer: The closing price of Apple (AAPL) on 2024-03-15 was $155.23.
```

**Input**:  
```
How many trading days in 2024 did Boeing trade?
```

**Output**:
```
SQL Query: SELECT COUNT(*) AS trading_days FROM prices WHERE Ticker = 'BA' AND STRFTIME('%Y', Date) = '2024'
Answer: Boeing (BA) had 252 trading days in 2024.
```

### 2. Tạo biểu đồ (Visualization Agent)

**Input**:  
```
Create a pie chart showing the distribution of DJIA companies by sector.
```

**Output**:  
- Biểu đồ pie chart hiển thị phân bố các công ty theo sector
- Tự động tính toán và format percentages

**Input**:  
```
Plot the cumulative return of UnitedHealth Group (UNH) in 2024.
```

**Output**:  
- Line chart hiển thị cumulative returns theo thời gian
- Tính toán: (Close / Close.first() - 1) * 100

**Input**:  
```
Create a boxplot of monthly closing prices of Walt Disney (DIS) for each month of 2024.
```

**Output**:  
- Boxplot hiển thị phân bố giá đóng cửa theo từng tháng
- Tự động group theo tháng và format labels

### 3. Khuyến nghị giao dịch (Trading Agent)

**Input**:  
```
Should I buy or sell AAPL for the next 5 days?
```

**Output**:
```
Trading Decision for AAPL
Signal: BUY
Probabilities:
  - BUY: 0.65
  - HOLD: 0.28
  - SELL: 0.07

Explanation:
Based on the current technical indicators, AAPL shows strong bullish signals:
- RSI (14) at 58.3 indicates healthy momentum without overbought conditions
- MACD crossover above signal line suggests upward trend continuation
- Volume ratio of 1.23 indicates increased buying interest
- Price is 2.1% above 20-day EMA, showing positive momentum
- ATR ratio of 0.018 suggests moderate volatility, favorable for trend continuation
Recommendation: BUY with 65% confidence. Consider setting stop-loss at recent support levels.
```

### 4. Phân tích tài liệu (RAG Agent)

**Input**:  
```
What methods were used to align GPT-4 after pretraining?
```

**Output**:
```
KB Hits:
- "...alignment techniques including RLHF and safety mitigations..."
- "...supervised fine-tuning (SFT) followed by reinforcement learning..."

Answer:
GPT-4 was aligned after pretraining using supervised fine-tuning (SFT), 
reinforcement learning with human feedback (RLHF), and post-training 
safety mitigations.
```

**Input**:  
```
According to the 10-K filing, what were the main risk factors?
```

**Output**:
```
KB Hits:
- [Extracted sections from 10-K-2025.pdf about risk factors]

Answer:
[Detailed analysis of risk factors from the document]
```

### 5. Query phức tạp với nhiều agents

**Input**:  
```
Show me the correlation heatmap of daily returns for Microsoft, Apple, and Google in 2024.
```

**Output**:
- Text-to-SQL Agent: Lấy dữ liệu giá cho 3 tickers
- Visualization Agent: Tạo correlation heatmap
- Hiển thị correlation matrix với color coding

---

## 🔧 Cấu hình nâng cao

### Token Management (Text-to-SQL Agent)
Agent tự động giới hạn kết quả SQL dựa trên token limit:
- `MAX_CONTEXT_TOKENS = 14000`
- `BASE_PROMPT_TOKENS = 2400`
- `TOKENS_PER_ROW = 48`

### Trading Model Features
Trading Agent sử dụng 50+ features bao gồm:
- **Returns**: 1d, 2d, 5d, 10d, 20d returns
- **Momentum**: RSI (7, 14), MACD, Stochastic
- **Trend**: EMA crossovers, ADX
- **Volatility**: ATR, Bollinger Bands
- **Volume**: VWAP, MFI, volume ratios
- **Patterns**: Gaps, higher highs, lower lows

### Visualization Types
Agent hỗ trợ tự động detect và tạo:
- Line charts (time series)
- Pie charts (distributions)
- Bar charts (comparisons)
- Boxplots (distributions by group)
- Heatmaps (correlations)
- Histograms (return distributions)
- Scatter plots (relationships)

---

## 🧪 Testing

Chạy tests cho từng agent:
```bash
# Test Text-to-SQL Agent
python test_agent/test_text_to_sql_agent.py

# Test RAG Agent
python test_agent/test_rag_agent.py

# Test Visualization Agent
python test_agent/test_visualization_agent.py

# Test Trading Agent
python test_agent/test_trading_agent.py

# Test Orchestrator Agent
python test_agent/test_orchestrator_agent.py
```

---

## 📚 Dependencies chính

- **phidata** (2.7.10): Framework cho agent orchestration
- **groq** (0.37.0): LLM API (Llama 3.3 70B)
- **google-generativeai** (0.8.5): Gemini embeddings
- **qdrant_client** (1.6.0): Vector database
- **pandas** (2.3.3): Data manipulation
- **sqlalchemy** (2.0.44): Database ORM
- **matplotlib** (3.10.7) & **seaborn** (0.13.2): Visualization
- **xgboost** (3.1.0): Trading ML models
- **scikit-learn** (1.7.2): Feature scaling
- **ta** (0.11.0): Technical analysis indicators
- **streamlit** (1.51.0): Web interface

Xem đầy đủ trong `requirements.txt`.

---

## 🤝 Đóng góp

Pull requests và issues được hoan nghênh!  
Hãy đảm bảo:
- Code tuân thủ chuẩn **PEP8**
- Có **docstring rõ ràng** cho functions/classes
- Thêm tests cho tính năng mới
- Cập nhật README nếu cần

---

## 📜 License

MIT License © 2025 NghiaAi

---

## 🔗 Links

- [phidata Documentation](https://docs.phidata.com)
- [Groq API](https://console.groq.com)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Streamlit Documentation](https://docs.streamlit.io)
