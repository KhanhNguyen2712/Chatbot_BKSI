import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_chain import ask_question
from src.utils import setup_logging

logger = setup_logging()

st.set_page_config(page_title="Chatbot BKSI", page_icon="🎓")

st.title("🎓 Chatbot Tư Vấn Trường Đại Học BKSI")
st.markdown("Hỏi tôi về quy chế, học phí, đăng ký môn học, chương trình đào tạo, v.v.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        try:
            response = ask_question(prompt)
            st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            logger.error(f"Error: {e}")
            error_msg = "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại."
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
