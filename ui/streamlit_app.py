"""Streamlit UI for Chatbot BKSI."""

import re
import subprocess
import sys
import time
import uuid

from pathlib import Path
from typing import Generator

import streamlit as st

from src import __version__
from src.config import ConfigManager, get_settings
from src.embeddings import EmbeddingModel
from src.rag import RAGChain
from src.utils import setup_logging
from src.vectorstore import LanceDBVectorStore
from src.document_processing import DocumentParser, TextChunker

# Page config
st.set_page_config(
    page_title="Chatbot BKSI",
    page_icon="🎓",
    layout="wide",  # Wide layout needed for side-by-side view
    initial_sidebar_state="auto",
)

# Setup logging
setup_logging()


def load_css():
    """Load external CSS."""
    with open("ui/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def clean_text(text: str) -> str:
    """Clean text by removing artifacts like <--br-->."""
    if not text:
        return ""
    # Remove <--br--> tags
    text = re.sub(r"<--br-->", "\n", text)
    # Remove multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@st.cache_resource
def init_components():
    """Initialize RAG components (cached)."""
    settings = get_settings()

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
    
    # Initialize document processing
    doc_parser = DocumentParser()
    text_chunker = TextChunker()

    return rag_chain, vector_store, doc_parser, text_chunker, settings


def init_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []


def reindex_data(chunk_size: int, chunk_overlap: int):
    """Trigger data re-indexing."""
    try:
        # Update settings.yaml temporarily or pass args to CLI?
        # CLI supports args, but ingestion logic reads from settings.
        # We will use the CLI command logic directly or via subprocess with flags if supported.
        # The current CLI (cli.py) reads from settings. 
        # For this demo, we'll try to run the ingest command. 
        # Ideally, we should update the Settings object or passed args.
        # Since cl.py is simple, let's run it.
        # Note: Changing chunk size requires rebuilding the index.
        
        with st.spinner("Đang xử lý dữ liệu (có thể mất vài phút)..."):
            # We can't easily pass chunk params to CLI without modifying it to accept them.
            # Assuming for now we just trigger a rebuild, but really the user wants to TEST params.
            # We'll stick to a placeholder "ingest" call for now.
            cmd = [sys.executable, "scripts/cli.py", "ingest", "--rebuild"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("Xử lý dữ liệu thành công!")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Lỗi: {result.stderr}")
                
    except Exception as e:
        st.error(f"Đã có lỗi xảy ra: {e}")


def display_sidebar(settings):
    """Display sidebar with settings and info."""
    with st.sidebar:
        st.title("🎓 Chatbot BKSI")
        st.caption(f"v{__version__} | Powered by Groq")

        if st.button("✨ Đoạn chat mới", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        st.divider()

        with st.expander("🛠️ Cấu hình tham số", expanded=True):
            # Runtime Params
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, help="Độ sáng tạo của câu trả lời.")
            top_k = st.slider("Top-K", 1, 10, settings.rag_top_k, 1, help="Số lượng tài liệu tham khảo.")
        
        st.divider()
        
        st.markdown("### 📝 Hướng dẫn")
        st.markdown("""
        **Cách sử dụng:**
        1. Nhập câu hỏi và nhấn Enter
        2. Xem câu trả lời kèm nguồn
        
        **🎓 Tự động học:**
        Chatbot sẽ **tự động nhận biết** 
        và lưu thông tin mới bạn cung cấp!
        
        **Ví dụ:**
        - "Địa chỉ trường là 268 Lý Thường Kiệt"
        - "Email: support@hcmut.edu.vn"
        
        ➡️ Bot sẽ tự động lưu và nhớ!
        """)

        return temperature, top_k


def stream_text(text: str) -> Generator[str, None, None]:
    """Simulate streaming text effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)


def display_hero_css():
    """Inject CSS to center the chat input when history is empty."""
    st.markdown(
        """
        <style>
        /* Force the bottom container to move up */
        div[data-testid="stBottomBlockContainer"] {
            position: absolute;
            bottom: 40% !important; /* Force position */
            left: 50%;
            transform: translateX(-50%);
            width: 100%;
            max-width: 800px; /* Match input width */
            padding-bottom: 0;
            background: transparent;
            z-index: 999;
        }
        
        /* Ensure the input container itself is centered inside */
        .stChatInputContainer {
             margin: 0 auto;
        }

        /* Hide the footer spacer */
        div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stChatInputContainer"]) {
             display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def display_welcome_screen():
    """Display welcome screen centrally."""
    # Layout: [Spacer] [Content] [Spacer] - Symmetric for Hero Mode
    _, col_center, _ = st.columns([0.2, 0.6, 0.2])
    
    with col_center:
        st.markdown(
            """
            <div style="text-align: center; margin-top: 5vh; margin-bottom: 3rem;">
                <div class="welcome-text">Trợ lý ảo BKSI</div>
                <div class="welcome-subtext">Hôm nay tôi có thể giúp gì cho bạn?</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Suggestions
        cols = st.columns(3)
        suggestions = [
            "Đăng ký môn học?",
            "Quy chế mới nhất?",
            "Học phí đại học?",
        ]
        for i, col in enumerate(cols):
            with col:
                st.info(suggestions[i], icon="✨")


def render_message(role: str, content: str, sources: list = None):
    """Render a single chat message with 3-column layout."""
    # Layout: [Spacer (10%)] | [Chat (60%)] | [Sources (30%)]
    
    # We use a container first to manage the row
    with st.container():
        col_spacer, col_chat, col_src = st.columns([0.1, 0.6, 0.3])
        
        with col_chat:
            with st.chat_message(role):
                st.markdown(content)
        
        # Sources only for assistant and if available
        if role == "assistant" and sources:
            with col_src:
                st.caption("📚 Nguồn tham khảo")
                for idx, src in enumerate(sources, 1):
                    with st.expander(f"[{idx}] {src.get('document_id', 'Doc')}", expanded=False):
                        st.caption(f"Score: {src.get('score', 0):.2f}")
                        st.markdown(f"<small>{src.get('content', '')[:150]}...</small>", unsafe_allow_html=True)


def main():
    """Main Streamlit app."""
    load_css()
    init_session_state()
    rag_chain, vector_store, doc_parser, text_chunker, settings = init_components()

    # Tabs for different sections
    tab1 = st.tabs(["💬 Hội thoại"])
    
    with tab1:
        render_chat_tab(rag_chain, settings)
def render_chat_tab(rag_chain, settings):
    """Render the chat interface tab."""
    temp, top_k = display_sidebar(settings)
    
    # Update runtime settings
    rag_chain.settings.llm_temperature = temp

    # DYNAMIC LAYOUT LOGIC
    # Create a placeholder for the Hero UI (CSS + Welcome)
    hero_placeholder = st.empty()
    
    # Check if user has just submitted an input (Hero Mode should vanish immediately)
    user_input = st.session_state.get("main_chat_input")
    
    # Render Hero if history is empty AND no new input
    if not st.session_state.messages and not user_input:
        with hero_placeholder.container():
            display_hero_css()
            display_welcome_screen()
    else:
        # Chat Mode: Standard history display
        for message in st.session_state.messages:
            render_message(message["role"], message["content"], message.get("sources"))

    # Chat Input
    # The input box is centrally styled by CSS max-width, so we just render it.
    if prompt := st.chat_input("Nhập câu hỏi của bạn...", key="main_chat_input"):
        # CRITICAL: Instantly clear the Hero UI to prevent ghosting
        hero_placeholder.empty()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Force a rerun to ensure layout updates immediately
        st.rerun() 
        
    # PROCESS NEW MESSAGE AFTER RERUN
    # NOTE: Logic flow needs adjustment because st.chat_input triggers a rerun automatically.
    # The actual processing needs to happen if the LAST message is from USER.
    
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        # Get the latest user prompt
        prompt = st.session_state.messages[-1]["content"]
        
        # We need to re-render history first because of rerun
        # (This is handled by the 'else' block above in the NEXT run, but we need to show the NEW user message NOW if we want streaming)
        # Actually proper streamlit flow:
        # 1. Rerun happens.
        # 2. History loop renders existing conversation (including new user msg).
        # 3. We detect user msg needs response.
        
        # Let's fix the flow:
        # If last msg is User, generate Assistant response.
        pass # Logic handled below layout

    # Handle Generation ONLY if last message is USER
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        
        # Render the just-added user message (if not already rendered in history loop)
        # In current flow, if we just appended and reran, it WILL be in history loop above.
        # Wait, if we use st.rerun(), the script stops and restarts. 
        # So the `if prompt` block executes, appends, reruns.
        # Next run: `messages` is NOT empty. `display_hero_css` is SKIPPED. `history loop` runs.
        # Last message is USER. We need to generate response.      
        with st.container():
             col_spacer, col_chat, col_src = st.columns([0.1, 0.6, 0.3])
             
             with col_chat:
                with st.chat_message("assistant"):
                    with st.spinner("Đang suy nghĩ..."):
                        response_obj = rag_chain.chat(
                            message=st.session_state.messages[-1]["content"],
                            session_id=st.session_state.session_id,
                            top_k=top_k,
                        )
                    
                    full_response = clean_text(response_obj.answer or "Xin lỗi, tôi không có câu trả lời.")                    
                    placeholder = st.empty()
                    streamed_text = ""
                    for chunk in stream_text(full_response):
                        streamed_text += chunk
                        placeholder.markdown(streamed_text + "▌")
                    placeholder.markdown(streamed_text)
            
             if response_obj.sources:
                with col_src:
                    st.caption("📚 Nguồn tham khảo")
                    for idx, src in enumerate(response_obj.sources, 1):
                        with st.expander(f"[{idx}] {src.get('document_id', 'Doc')}", expanded=False):
                            st.caption(f"Score: {src.get('score', 0):.2f}")
                            st.markdown(f"<small>{src.get('content', '')[:150]}...</small>", unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": response_obj.sources
        })
        st.rerun() # Rerun again to clean up streaming placeholders


if __name__ == "__main__":
    main()
