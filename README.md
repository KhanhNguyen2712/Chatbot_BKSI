# Chatbot BKSI - RAG Chatbot cho Trường Đại Học

Chatbot RAG (Retrieval-Augmented Generation) phục vụ truy xuất thông tin về quy chế, học phí, đăng ký môn học, chương trình đào tạo của Trường Đại Học Bách Khoa - ĐHQGHCM.

## Tính năng

- **Truy xuất thông tin thông minh**: Sử dụng RAG để tìm kiếm và trả lời dựa trên tài liệu trường học
- **Giao diện thân thiện**: Hỗ trợ cả Gradio (web đơn giản) và Streamlit (dashboard)
- **Embedding tiếng Việt**: Sử dụng model embedding chuyên cho tiếng Việt
- **Miễn phí**: Sử dụng các công nghệ và API miễn phí

## Công nghệ sử dụng

- **Langchain**: Framework cho LLM applications
- **ChromaDB**: Vector database cho lưu trữ embeddings
- **OpenRouter API**: API cho model GPT-OSS 120B (miễn phí)
- **Sentence Transformers**: Model embedding tiếng Việt (keepitreal/vietnamese-sbert)
- **Gradio & Streamlit**: Giao diện người dùng
- **Python-dotenv**: Quản lý biến môi trường

## Cài đặt

1. **Clone repository** (nếu có) hoặc tạo thư mục dự án

2. **Cài đặt dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Cấu hình biến môi trường**:

   - Sao chép `.env` và điền API key từ [OpenRouter](https://openrouter.ai/)
   - API key miễn phí, đăng ký tài khoản để lấy

4. **Chuẩn bị dữ liệu**:

   - Đặt các file PDF/DOCX/Markdown về quy chế, học phí, v.v. vào thư mục `data/raw/`

5. **Khởi tạo database**:
   ```bash
   python scripts/setup_db.py
   ```

## Sử dụng

### Chạy Gradio App (Đơn giản)

```bash
python ui/gradio_app.py
```

### Chạy Streamlit App (Nâng cao)

```bash
streamlit run ui/streamlit_app.py
```

## Cấu trúc dự án

```
Chatbot_BKSI/
├── .env                    # Biến môi trường
├── README.md               # Tài liệu này
├── requirements.txt        # Dependencies
├── data/                   # Dữ liệu
│   ├── raw/                # Tài liệu gốc
│   └── processed/          # Dữ liệu đã xử lý
├── src/                    # Code chính
│   ├── config.py           # Cấu hình
│   ├── data_ingestion.py   # Xử lý dữ liệu
│   ├── embedding.py        # Embedding model
│   ├── vector_store.py     # ChromaDB
│   ├── llm.py              # OpenRouter LLM
│   ├── rag_chain.py        # RAG pipeline
│   └── utils.py            # Utilities
├── ui/                     # Giao diện
│   ├── gradio_app.py       # Gradio interface
│   └── streamlit_app.py    # Streamlit interface
├── scripts/                # Scripts hỗ trợ
│   ├── setup_db.py         # Khởi tạo DB
│   └── run_ingestion.py    # Chạy ingestion
└── tests/                  # Unit tests
    └── test_rag.py
```

## API Keys

- **OpenRouter**: Đăng ký tại https://openrouter.ai/ để lấy API key miễn phí
- Model sử dụng: `gpt-oss-120b` (miễn phí)

## Phát triển thêm

- Thêm nhiều tài liệu: Đặt file vào `data/raw/` và chạy lại `setup_db.py`
- Tối ưu hóa: Điều chỉnh chunk_size, top_k trong `config.py`
- Logging: Xem logs trong console hoặc file log

## License

Miễn phí sử dụng cho mục đích giáo dục.

## Liên hệ

Nếu có vấn đề, kiểm tra logs hoặc tạo issue trên repository.
