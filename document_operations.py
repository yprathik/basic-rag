from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

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

