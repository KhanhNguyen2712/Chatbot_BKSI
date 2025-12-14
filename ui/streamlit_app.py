"""Streamlit UI for Chatbot BKSI."""

import time
import uuid
from typing import Generator

import streamlit as st

from src import __version__
from src.config import ConfigManager, get_settings
from src.embeddings import EmbeddingModel
from src.rag import RAGChain
from src.utils import setup_logging
from src.vectorstore import LanceDBVectorStore

# Page config
st.set_page_config(
    page_title="Chatbot BKSI",
    page_icon="🎓",
    layout="centered",  # Centered layout looks more like a chat app
    initial_sidebar_state="auto",
)

# Custom CSS for ChatGPT-like styling
ST_CSS = """
<style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main App Background */
    .stApp {
        background-color: #343541;
    }
    
    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 6rem; /* Space for chat input */
    }
    
    /* Text Color */
    p, h1, h2, h3 {
        color: #ECECF1;
    }

    /* Chat message styling */
    .stChatMessage {
        background-color: transparent;
        border: none;
    }
    
    /* User message avatar area */
    [data-testid="stChatMessage"] {
        padding: 1.5rem;
    }

    /* User message background (Darker/Different) */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #343541; /* Match background or slight vary */
    }
    
    /* Assistant message background (Lighter) */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #444654;
    }
    
    /* Code block styling */
    code {
        color: #E0E0E0 !important;
        background-color: #2D2F34 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #202123;
    }
    
    /* Input box styling */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    
    /* Expander styling in dark mode */
    .streamlit-expanderHeader {
        background-color: #444654;
        color: #ECECF1;
    }
</style>
"""
st.markdown(ST_CSS, unsafe_allow_html=True)

# Setup logging
setup_logging()


@st.cache_resource
def init_components():
    """Initialize RAG components (cached)."""
    settings = get_settings()
    # config = ConfigManager() # Not used directly in UI currently

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

    return rag_chain, settings


def init_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_sidebar(settings):
    """Display sidebar with settings and info."""
    with st.sidebar:
        st.title("🎓 Chatbot BKSI")
        st.caption(f"v{__version__} | Powered by RAG")

        # New Chat Button
        if st.button("✨ Đoạn chat mới", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        st.divider()

        # Settings
        with st.expander("⚙️ Cấu hình nâng cao"):
            top_k = st.slider(
                "Số lượng tài liệu",
                min_value=1,
                max_value=10,
                value=settings.rag_top_k,
                help="Số lượng văn bản được trích xuất.",
            )

            use_rerank = st.checkbox(
                "Kích hoạt Rerank",
                value=settings.rag_rerank_enabled,
                help="Sắp xếp lại kết quả để chính xác hơn.",
            )
        
        st.divider()
        st.markdown("### ℹ️ Thông tin")
        st.markdown(f"- **LLM**: `{settings.llm_model}`")
        st.markdown(f"- **Embedding**: `Unknown`") #  Simplified for UI cleanliness

        st.caption("© 2024 BKSI Project")
        
        return top_k, use_rerank


def stream_text(text: str) -> Generator[str, None, None]:
    """Simulate streaming text effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


def display_welcome_screen():
    """Display welcome screen when chat is empty."""
    st.markdown(
        """
        <div style="text-align: center; margin-top: 5rem;">
            <h1>👋 Chào bạn!</h1>
            <p>Tôi là trợ lý ảo BKSI, tôi có thể giúp gì cho bạn hôm nay?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Suggestion chips
    cols = st.columns(3)
    suggestions = [
        "Quy trình đăng ký môn học?",
        "Thời gian đóng học phí?",
        "Điều kiện tốt nghiệp là gì?",
    ]
    
    # Allow clicking suggestions to populate chat input (requires Streamlit hack or just copy-paste)
    # Since st.button inside chat input isn't native, we just display them.
    for i, col in enumerate(cols):
        with col:
            st.info(suggestions[i], icon="💡")


def main():
    """Main Streamlit app."""
    init_session_state()
    rag_chain, settings = init_components()

    top_k, use_rerank = display_sidebar(settings)

    # Display chat messages
    if not st.session_state.messages:
        display_welcome_screen()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display sources for assistant messages if available within the message object
            # Note: We need to store sources in the message history structure
            if message.get("sources"):
                with st.expander("📚 Nguồn tham khảo"):
                    for idx, src in enumerate(message["sources"], 1):
                        st.markdown(f"**[{idx}] {src.get('document_id', 'Doc')}** ({src.get('score', 0):.2f})")
                        st.caption(src.get('content', '')[:300] + "...")


    # Chat input
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                t_start = time.time()
                response_obj = rag_chain.chat(
                    message=prompt,
                    session_id=st.session_state.session_id,
                    top_k=top_k,
                    use_rerank=use_rerank,
                )
                t_end = time.time()
                
            # Simulate streaming
            response_text = response_obj.answer or "Xin lỗi, tôi không có câu trả lời."
            placeholder = st.empty()
            full_response = ""
            for chunk in stream_text(response_text):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

            # Display sources immediately below
            if response_obj.sources:
                with st.expander("📚 Nguồn tham khảo"):
                    for idx, src in enumerate(response_obj.sources, 1):
                         st.markdown(f"**[{idx}] {src.get('document_id', 'Doc')}** ({src.get('score', 0):.2f})")
                         st.caption(src.get('content', '')[:300] + "...")

            # Save to history including sources for persistence
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": response_obj.sources
            })


if __name__ == "__main__":
    main()
