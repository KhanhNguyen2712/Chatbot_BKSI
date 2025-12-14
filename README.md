# 🎓 Chatbot BKSI

RAG-based Q&A Chatbot cho sinh viên Trường Đại Học Bách Khoa - ĐHQG-HCM.

## ✨ Features

- 📄 **Document Processing**: Parse PDF, DOCX, PPTX với Docling
- 🇻🇳 **Vietnamese Embeddings**: Tối ưu cho tiếng Việt với `dangvantuan/vietnamese-document-embedding`
- 🗄️ **LanceDB Vector Store**: Serverless, cross-platform vector database
- ⚡ **Response Caching**: Tăng tốc các truy vấn lặp lại
- 💬 **Conversation Memory**: Chat theo ngữ cảnh
- 🎯 **Reranking**: Cải thiện độ chính xác với CrossEncoder (BGE-Reranker multilingual)
- 📡 **Document Management API**: Upload/delete/reindex documents
- 🖥️ **Multiple UI Options**: Gradio + Streamlit

## 🛠️ Tech Stack

| Component       | Technology                                |
| --------------- | ----------------------------------------- |
| Package Manager | **uv**                                    |
| Vector Database | **LanceDB** (serverless, cross-platform)  |
| LLM Framework   | **LangChain**                             |
| LLM Provider    | **OpenRouter** (via ChatOpenAI)           |
| Document Parser | **Docling** (PDF, DOCX, PPTX)             |
| Embeddings      | **sentence-transformers**                 |
| Reranking       | **CrossEncoder** (ms-marco-MiniLM-L-6-v2) |
| Caching         | **LangChain SQLiteCache**                 |
| Backend API     | **FastAPI**                               |
| UI              | **Gradio** + **Streamlit**                |

## 📁 Project Structure

```
Chatbot_BKSI/
├── api/                    # FastAPI backend
│   ├── main.py            # FastAPI app
│   ├── dependencies.py    # Dependency injection
│   └── routes/            # API routes
│       ├── chat.py        # Chat endpoints
│       └── documents.py   # Document management
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
├── tests/               # Pytest tests
├── ui/                  # User interfaces
│   ├── gradio_app.py   # Gradio UI
│   └── streamlit_app.py # Streamlit UI
├── .env.example         # Environment template
├── pyproject.toml       # uv project config
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker & Docker Compose (optional, for containerized deployment)

### Setup (Local)

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

### Setup (Docker)

```bash
# Clone repository
git clone <repository-url>
cd Chatbot_BKSI

# Copy environment template
cp .env.example .env
# Edit .env with your API key

# Build and run all services
docker-compose up -d

# Or run specific service
docker-compose up api -d      # FastAPI only
docker-compose up gradio -d   # Gradio only
docker-compose up streamlit -d # Streamlit only

# Ingest documents (run once)
docker-compose --profile tools run ingest

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

#### GPU Support (NVIDIA)

```bash
# Requires NVIDIA Container Toolkit
docker-compose -f docker-compose.gpu.yml up -d
```

## 📖 Usage

### 1. Ingest Documents

Đặt tài liệu (PDF, DOCX, MD) vào thư mục `data/raw/`, sau đó:

```bash
# Local
uv run bksi ingest

# Docker
docker-compose --profile tools run ingest

# Rebuild index from scratch
uv run bksi ingest --rebuild
```

### 2. Run Gradio UI

```bash
uv run bksi gradio

# Or with custom options
uv run bksi gradio --host 0.0.0.0 --port 7860 --share
```

### 3. Run Streamlit UI

```bash
uv run bksi streamlit
```

### 4. Run FastAPI Server

```bash
uv run bksi api

# Or with uvicorn directly
uv run uvicorn api.main:app --reload
```

### 5. Interactive Chat (Terminal)

```bash
uv run bksi chat
```

### 6. Clear Cache

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
  rerank_model: BAAI/bge-reranker-v2-m3
  rerank_top_n: 3

cache:
  enabled: true
  directory: ./.cache

memory:
  max_messages: 20
```

## 📡 API Endpoints

### Chat

- `POST /chat/` - Chat với RAG
- `POST /chat/search` - Semantic search
- `POST /chat/clear-memory` - Xóa memory session
- `POST /chat/clear-cache` - Xóa response cache

### Documents

- `POST /documents/upload` - Upload document
- `GET /documents/` - List documents
- `DELETE /documents/{document_id}` - Delete document
- `GET /documents/stats` - Get statistics
- `POST /documents/reindex` - Rebuild vector index

### Health

- `GET /health` - Health check
- `GET /` - API info

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/test_components.py -v
```

## 📚 Documentation

API documentation available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details
