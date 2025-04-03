from document_operations import preprocess, store_documents
from generate_embeddings import OllamaEmbedModel
from langchain_community.llms import Ollama
from ui import chat_interface
import streamlit as st

# Load documents
file_path = "/home/prathik/coding-prathik/rci-work/basic-rag/documents/16.-Streamlit.pdf"
docs = preprocess(file_path)

# Create embedding model and store vectors
ollama_embed_model = OllamaEmbedModel()
vectorstore = store_documents(docs, ollama_embed_model)

# Create retriever and model
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
ollama_model = Ollama(base_url="http://localhost:11434", model="mistral")

# Run the chat interface
#chat_interface(retriever, ollama_model)

















"""
from document_operations import preprocess, store_documents
from generate_embeddings import OllamaEmbedModel
from langchain_community.llms import Ollama 
from langchain.chains import RetrievalQA
from ui import chat_interface


file_path = "/home/prathik/coding-prathik/rci-work/basic-rag/documents/16.-Streamlit.pdf"

docs = preprocess(file_path)

ollama_embed_model = OllamaEmbedModel()

vectorstore = store_documents(docs, ollama_embed_model)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Fetch top 5 results

ollama_model = Ollama(base_url="http://localhost:11434", model="mistral")

messages = []

while True:
    query = input("Human: ")
    messages.append(query)

    if query.lower() == "exit":
        break

    retrieved_docs = retriever.get_relevant_documents(query)

    # Deduplicate chunk IDs and store content
    unique_chunks = {}
    for doc in retrieved_docs:
        chunk_id = doc.metadata.get("chunk_id", "Unknown")
        if chunk_id not in unique_chunks:
            unique_chunks[chunk_id] = doc.page_content  

    print("\nRetrieved Chunks:")
    for chunk_id, content in unique_chunks.items():
        print(f"\nChunk ID: {chunk_id}\nContent: {content[:200]}...")  # Show first 500 characters for readability

    response = ollama_model.invoke(messages)
    messages.append(response)

    # response = conversation.predict(input=query)

    print("\nAI:")
    print(response)
"
"""