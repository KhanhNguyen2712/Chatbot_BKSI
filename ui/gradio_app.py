import os
import sys

import gradio as gr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_chain import ask_question
from src.utils import setup_logging

logger = setup_logging()


def chatbot_response(message, history):
    """Handle chatbot response."""
    try:
        response = ask_question(message)
        # Append user message and bot response to history
        history = history + [[message, response]]
        return "", history
    except Exception as e:
        logger.error(f"Error in chatbot response: {e}")
        error_msg = "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại."
        history = history + [[message, error_msg]]
        return "", history


def create_gradio_interface():
    """Create Gradio interface."""
    with gr.Blocks(title="Chatbot Trường Đại Học Bách Khoa-ĐHQGHCM") as interface:
        gr.Markdown("# Chatbot Tư Vấn Trường Đại Học Bách Khoa-ĐHQGHCM")
        gr.Markdown(
            "Hỏi tôi về quy chế, học phí, đăng ký môn học, chương trình đào tạo, v.v."
        )

        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Nhập câu hỏi của bạn...")
        clear = gr.Button("Xóa cuộc trò chuyện")

        msg.submit(chatbot_response, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: ("", []), None, [msg, chatbot], queue=False)

    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()
