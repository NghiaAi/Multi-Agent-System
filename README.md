# 🧠 Multi-Agent System

Hệ thống **Multi-Agent System** được xây dựng bằng [**phidata**](https://github.com/phidatahq/phidata), kết hợp **Text-to-SQL Agent** và **RAG Agent** để:
- Truy vấn dữ liệu tài chính (Dow Jones Industrial Average – DJIA) bằng SQL.
- Phân tích và trả lời câu hỏi từ tài liệu PDF (GPT-4 Technical Report) với RAG (Gemini + Qdrant).
- Điều phối các agent thông qua **Orchestrator Agent**.

## 🚀 Tính năng chính
- **Phidata-based Agents**  
  Các agent trong hệ thống được định nghĩa và điều phối bởi framework **phidata**, giúp dễ dàng:
  - Quản lý nhiều agent độc lập.
  - Xây dựng pipeline trả lời phức tạp.
  - Lưu lịch sử hội thoại và state của agent.

- **Text-to-SQL Agent**  
  - Hiểu câu hỏi tài chính (giá, khối lượng, cổ tức, v.v.).
  - Chuyển đổi câu hỏi sang SQL để truy vấn database `djia.db`.
  - Tự động ánh xạ tên công ty sang ticker (VD: *Apple* → `AAPL`).

- **RAG Agent**  
  - Truy xuất kiến thức từ PDF (`2303.08774v6.pdf`).
  - Sử dụng **Gemini** để sinh embedding + **Qdrant** làm vector DB.
  - Luôn trích dẫn các đoạn văn bản liên quan trước khi trả lời.

- **Orchestrator Agent**  
  - Xây dựng bằng **phidata Agent** để phân tích query người dùng.
  - Quyết định gọi **Text-to-SQL Agent** hoặc **RAG Agent**.
  - Hỗ trợ ngữ cảnh dựa trên chat history.

- **Streamlit Apps**  
  - `app.py`: giao diện web truy vấn dữ liệu DJIA và RAG.

---

## 🗂 Cấu trúc thư mục

```bash
Multi-Agent-System/
├── agents/
│   ├── orchestrator_agent.py   # Orchestrator Agent (phidata)
│   ├── rag_agent.py            # RAG Agent (Gemini + Qdrant, phidata)
│   └── text_to_sql_agent.py    # Text-to-SQL Agent (DJIA DB, phidata)
├── scripts/
│   ├── create_db.py            # Tạo database từ CSV
│   ├── djia_streamlit.py       # Giao diện Streamlit cho DJIA
│   └── rag_streamlit.py        # Giao diện Streamlit cho RAG
│   └── app.py                  # Giao diện Streamlit cho DJIA và RAG
├── data/
│   ├── djia_companies_20250426.csv
│   ├── djia_prices_20250426.csv
│   └── 2303.08774v6.pdf
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt

### 1. Clone repo
```bash
git clone https://github.com/NghiaAi/Multi-Agent-System.git
cd Multi-Agent-System
```

### 2. Tạo virtual env & cài dependencies
```bash
python -m venv venv
source venv/bin/activate   # hoặc: venv\Scripts\activate trên Windows
pip install -r requirements.txt
```

### 3. Tạo file `.env`
```env
# API keys
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key

# Qdrant
QDRANT_URL=https://your-qdrant-url
QDRANT_API_KEY=your_qdrant_api_key
```

### 4. Khởi tạo database
```bash
python scripts/create_db.py
```

---

## ▶️ Chạy ứng dụng

### 1. Trợ lý truy vấn DJIA VÀ RAG
```bash
streamlit run scripts/app.py
```
👉 Mở trình duyệt tại [http://localhost:8501](http://localhost:8501)


## 🏗 Kiến trúc hệ thống

```
                  ┌─────────────────┐
                  │  User Query      │
                  └───────┬─────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  Orchestrator Agent │  (phidata)
                └───────┬────────────┘
             ┌──────────┴────────────┐
             ▼                       ▼
 ┌────────────────────┐     ┌──────────────────────┐
 │  Text-to-SQL Agent │     │     RAG Agent        │
 │  (DJIA Database)   │     │ (Gemini + Qdrant +   │
 │                    │     │   PDF Knowledge)     │
 └─────────┬──────────┘     └──────────┬───────────┘
           │                           │
           ▼                           ▼
   SQL Queries & Results         PDF Context + Answer
```

---

## 📊 Ví dụ sử dụng

### Query DJIA:
**Input**:  
`What was the closing price of Apple on 2024-03-15?`

**Output**:
```
SQL Query: SELECT Close FROM prices WHERE Ticker = "AAPL" AND DATE(Date) = "2024-03-15"
Raw Result: [(155.23,)]
Answer: The closing price of Apple (AAPL) on 2024-03-15 was $155.23.
```

### Query RAG:
**Input**:  
`What methods were used to align GPT-4 after pretraining?`

**Output**:
```
KB hits:
- "...alignment techniques including RLHF and safety mitigations..."

Answer:
GPT-4 was aligned after pretraining using supervised fine-tuning (SFT), reinforcement learning with human feedback (RLHF), and post-training safety mitigations.
```

---

## 🤝 Đóng góp
Pull requests và issues được hoan nghênh!  
Hãy đảm bảo code tuân thủ chuẩn **PEP8** và có **docstring rõ ràng**.

---

## 📜 License
MIT License © 2025 NghiaAi
