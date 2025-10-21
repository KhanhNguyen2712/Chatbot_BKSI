import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config

def load_documents():
    """Load documents from raw data directory."""
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".md": TextLoader,
    }

    documents = []
    for file in os.listdir(Config.DATA_RAW_PATH):
        if file.endswith(('.pdf', '.docx', '.md')):
            file_path = os.path.join(Config.DATA_RAW_PATH, file)
            ext = os.path.splitext(file)[1]
            loader_class = loaders.get(ext)
            if loader_class:
                try:
                    if ext == ".md":
                        # Specify UTF-8 encoding for markdown files
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded {len(docs)} documents from {file}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    # Skip this file and continue
                    continue
    return documents

def split_documents(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def preprocess_data():
    """Main function to preprocess data."""
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")

    print("Splitting documents...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # Save processed chunks if needed
    os.makedirs(Config.DATA_PROCESSED_PATH, exist_ok=True)
    # For now, just return chunks; vector store will handle embedding

    return chunks
