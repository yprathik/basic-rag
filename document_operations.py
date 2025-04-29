from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

def preprocess(file_path, chunk_size = 1000, chunk_overlap = 100):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap= chunk_overlap)
    docs = text_splitter.split_documents(documents)
    
    # Add metadata including source filename and chunk ID
    source_filename = os.path.basename(file_path)
    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = i 
        doc.metadata["source_filename"] = source_filename

    return docs

def store_documents(docs, embedding_model, persist_directory="./chromadb"):

    # Ensure each document has metadata
    for doc in docs:
        if "chunk_id" not in doc.metadata:
            doc.metadata["chunk_id"] = "missing_id"

    # Store in vector database with explicit metadata assignment
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embedding_model, 
        persist_directory=persist_directory
    )

    return vectorstore

def get_full_text(file_path):
    """
    Loads a document (PDF or DOCX) and returns its full text content.
    """
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        # Ensure you have installed python-docx: pip install python-docx
        # Also requires Docx2txtLoader dependency: pip install docx2txt
        loader = Docx2txtLoader(file_path)
    else:
        print(f"Unsupported file type: {file_extension}")
        return "" # Return empty string for unsupported types

    try:
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])
        return full_text
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        return "" # Return empty string on error

