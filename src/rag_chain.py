from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from src.llm import create_llm
from src.vector_store import create_vector_store


def create_rag_chain():
    """Create RAG chain for Q&A."""
    llm = create_llm()
    vector_store = create_vector_store()

    # Custom prompt for university chatbot
    prompt_template = """
    Bạn là trợ lý AI chuyên về thông tin trường đại học cho Trường Đại Học Bách Khoa - ĐHQGHCM. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.
    Nếu không tìm thấy thông tin liên quan, hãy nói rằng bạn không biết và gợi ý liên hệ với phòng ban liên quan.

    Thông tin tham khảo:
    {context}

    Câu hỏi: {question}

    Trả lời:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
    )

    return chain


def ask_question(question: str):
    """Ask a question using RAG chain."""
    chain = create_rag_chain()
    result = chain.invoke({"query": question})
    print(f"DEBUG: RAG result type: {type(result)}, content: {result}")
    if isinstance(result, dict) and "result" in result:
        return result["result"]
    elif isinstance(result, str):
        return result
    else:
        return str(result)
