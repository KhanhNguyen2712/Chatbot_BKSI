# 🎓 Chatbot BKSI

RAG-based Q&A Chatbot cho sinh viên Trường Đại Học Bách Khoa - ĐHQG-HCM.

## ✨ Features

- 📄 **Document Processing**: Parse PDF, DOCX, PPTX với Docling
- 🇻🇳 **Vietnamese Embeddings**: Tối ưu cho tiếng Việt với `dangvantuan/vietnamese-document-embedding`
- 🗄️ **LanceDB Vector Store**: Serverless, cross-platform vector database
- ⚡ **Response Caching**: Tăng tốc các truy vấn lặp lại
- 💬 **Conversation Memory**: Chat theo ngữ cảnh
- 🎯 **Reranking**: Cải thiện độ chính xác với CrossEncoder (ms-marco-MiniLM-L-6-v2)
- 🖥️ **Streamlit UI**: Giao diện chat hiện đại, responsive

## 🛠️ Tech Stack

| Component       | Technology                               |
| --------------- | ---------------------------------------- |
| Package Manager | **uv**                                   |
| Vector Database | **LanceDB** (serverless, cross-platform) |
| LLM Framework   | **LangChain**                            |
| LLM Provider    | **OpenRouter** (via ChatOpenAI)          |
| Document Parser | **Docling** (PDF, DOCX, PPTX)            |
| Embeddings      | **sentence-transformers**                |
| Reranking       | **CrossEncoder** (ms-marco-MiniLM-L-6-v2)    |
| Caching         | **LangChain SQLiteCache**                |
| UI              | **Streamlit**                            |

## 📁 Project Structure

```
Chatbot_BKSI/
├── configs/               # Configuration files
│   ├── settings.yaml      # App settings
│   └── prompts.yaml       # Prompt templates
├── data/                  # Data directory
│   ├── raw/              # Original documents (PDF, DOCX)
│   └── processed/        # Processed markdown files
├── scripts/              # CLI scripts
│   └── cli.py           # Typer CLI
├── src/                  # Source code
│   ├── cache/           # Response caching
│   ├── config.py        # Settings management
│   ├── document_processing/  # Docling parser, chunker
│   ├── embeddings/      # sentence-transformers
│   ├── models/          # Pydantic models
│   ├── rag/             # RAG chain, retriever, reranker, memory
│   ├── utils/           # Logging, helpers
│   └── vectorstore/     # LanceDB vector store
├── ui/                  # User interface
│   ├── streamlit_app.py # Streamlit UI
│   ├── style.css        # Custom styling
│   └── hero.js          # Hero mode JavaScript
├── .env.example         # Environment template
├── pyproject.toml       # uv project config
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone repository
git clone <repository-url>
cd Chatbot_BKSI

# Install dependencies with uv
uv sync

# Copy environment template
cp .env.example .env

# Edit .env with your API key
# OPENROUTER_API_KEY=your_key_here
```

## 📖 Usage

### 1. Ingest Documents

Đặt tài liệu (PDF, DOCX, MD) vào thư mục `data/raw/`, sau đó:

```bash
# Ingest documents
uv run bksi ingest

# Rebuild index from scratch
uv run bksi ingest --rebuild
```

### 2. Run Streamlit UI

```bash
uv run bksi streamlit
```

Truy cập `http://localhost:8501` để sử dụng chatbot.

### 3. Interactive Chat (Terminal)

```bash
uv run bksi chat
```

### 4. Clear Cache

```bash
uv run bksi clear-cache
```

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# Required
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional
LLM_MODEL=openai/gpt-oss-120b:free
EMBEDDING_MODEL=dangvantuan/vietnamese-document-embedding
EMBEDDING_DEVICE=cpu  # or cuda
```

### Settings (`configs/settings.yaml`)

```yaml
llm:
  model: openai/gpt-oss-120b:free
  temperature: 0.7
  max_tokens: 2048

embeddings:
  model: dangvantuan/vietnamese-document-embedding
  device: cuda # or cpu

vectorstore:
  persist_dir: ./lancedb_data
  table_name: chunks

rag:
  top_k: 5
  chunk_size: 512
  chunk_overlap: 50
  rerank_enabled: true
  rerank_model: ms-marco-MiniLM-L-6-v2
  rerank_top_n: 3

cache:
  enabled: true
  directory: ./.cache

memory:
  max_messages: 20
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src
```

