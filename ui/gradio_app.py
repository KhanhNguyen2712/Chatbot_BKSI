"""Gradio UI for Chatbot BKSI."""

import gradio as gr

from src import __version__
from src.config import ConfigManager, get_settings
from src.embeddings import EmbeddingModel
from src.rag import RAGChain
from src.utils import setup_logging
from src.vectorstore import LanceDBVectorStore

# Setup
setup_logging()
settings = get_settings()
config = ConfigManager()

# Initialize components
embedding_model = EmbeddingModel(
    model_name=settings.embedding_model,
    device=settings.embedding_device,
)

vector_store = LanceDBVectorStore(
    persist_dir=settings.lancedb_persist_dir,
    embedding_model=embedding_model,
)

rag_chain = RAGChain(
    vector_store=vector_store,
    use_rerank=settings.rag_rerank_enabled,
    use_memory=settings.memory_enabled,
)


def chat(message: str, history: list, session_id: str) -> tuple[str, list]:
    """Process chat message and return response with history."""
    if not message.strip():
        return "", history

    # Generate session ID if empty
    if not session_id:
        import uuid

        session_id = str(uuid.uuid4())

    # Get response from RAG chain
    result = rag_chain.chat(
        message=message,
        session_id=session_id,
    )

    # Format sources
    sources_text = ""
    if result.sources:
        sources_text = "\n\n---\n**📚 Nguồn tham khảo:**\n"
        for i, source in enumerate(result.sources[:3], 1):
            doc_id = source.get("document_id", "N/A")
            content = source.get("content", "")[:150] + "..."
            sources_text += f"\n{i}. **{doc_id}**: {content}\n"

    # Add to history
    response = result.answer or "Xin lỗi, tôi không thể trả lời câu hỏi này."
    full_response = response + sources_text

    history.append((message, full_response))

    return "", history


def clear_chat(session_id: str) -> tuple[list, str]:
    """Clear chat history."""
    if session_id:
        rag_chain.clear_memory(session_id)
    import uuid

    return [], str(uuid.uuid4())


def get_stats() -> str:
    """Get system statistics."""
    stats = []
    stats.append(f"📊 **Thống kê hệ thống**")
    stats.append(f"- Model: {settings.llm_model}")
    stats.append(f"- Embedding: {settings.embedding_model}")
    stats.append(f"- Vector DB: LanceDB")
    stats.append(f"- Reranking: {'Bật' if settings.rag_rerank_enabled else 'Tắt'}")
    stats.append(f"- Caching: {'Bật' if settings.cache_enabled else 'Tắt'}")
    return "\n".join(stats)


# Create Gradio interface
with gr.Blocks(
    title="Chatbot BKSI",
    theme=gr.themes.Soft(),
    css="""
    .chatbot {height: 500px;}
    .source-box {max-height: 200px; overflow-y: auto;}
    """,
) as demo:
    gr.Markdown(
        f"""
        # 🎓 Chatbot BKSI - Trợ lý sinh viên
        ### Hệ thống hỏi đáp tự động về quy định, quy trình của BKSI
        **Version:** {__version__}
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Hội thoại",
                elem_classes=["chatbot"],
                height=500,
                show_copy_button=True,
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Nhập câu hỏi",
                    placeholder="Ví dụ: Làm thế nào để đăng ký môn học?",
                    scale=4,
                    lines=2,
                )
                submit_btn = gr.Button("Gửi 📤", scale=1, variant="primary")

            with gr.Row():
                clear_btn = gr.Button("🗑️ Xóa hội thoại", variant="secondary")
                session_id = gr.Textbox(
                    label="Session ID",
                    value="",
                    visible=False,
                )

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Cài đặt")
            stats_display = gr.Markdown(get_stats())
            refresh_btn = gr.Button("🔄 Làm mới thống kê")

            gr.Markdown("### 📝 Hướng dẫn")
            gr.Markdown(
                """
                1. Nhập câu hỏi vào ô chat
                2. Nhấn **Gửi** hoặc Enter
                3. Xem câu trả lời và nguồn tham khảo
                4. Tiếp tục hội thoại theo ngữ cảnh

                **Một số câu hỏi mẫu:**
                - Quy trình đăng ký môn học như thế nào?
                - Thời gian đăng ký môn học là khi nào?
                - Làm sao để xem điểm thi?
                """
            )

    # Event handlers
    msg.submit(
        chat,
        inputs=[msg, chatbot, session_id],
        outputs=[msg, chatbot],
    )

    submit_btn.click(
        chat,
        inputs=[msg, chatbot, session_id],
        outputs=[msg, chatbot],
    )

    clear_btn.click(
        clear_chat,
        inputs=[session_id],
        outputs=[chatbot, session_id],
    )

    refresh_btn.click(
        get_stats,
        outputs=[stats_display],
    )


def main():
    """Run Gradio app."""
    demo.launch(
        server_name=settings.gradio_host,
        server_port=settings.gradio_port,
        share=settings.gradio_share,
    )


if __name__ == "__main__":
    main()
