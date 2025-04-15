import streamlit as st
import os
from document_operations import preprocess, store_documents
from generate_embeddings import OllamaEmbedModel
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="PDF RAG Application",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa' not in st.session_state:
    st.session_state.qa = None
if 'ollama_embed_model' not in st.session_state:
    st.session_state.ollama_embed_model = OllamaEmbedModel()

# Title and description
st.title("PDF RAG Application")
st.markdown("Upload a PDF document and ask questions about its content.")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Document Settings")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    # Settings for chunking
    st.subheader("Chunking Settings")
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=100, step=50)
    
    # Settings for retrieval
    st.subheader("Retrieval Settings")
    k_value = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
    similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # Process document button
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            # Save the uploaded file temporarily
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the document
            docs = preprocess(temp_file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Use a progress bar for document processing
            progress_bar = st.progress(0)
            
            # Process in batches if there are many documents
            batch_size = 10
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i+batch_size]
                if i == 0:  # First batch
                    st.session_state.vectorstore = store_documents(batch, st.session_state.ollama_embed_model)
                else:  # Subsequent batches
                    st.session_state.vectorstore.add_documents(batch)
                
                # Update progress
                progress = min(1.0, (i + batch_size) / len(docs))
                progress_bar.progress(progress)
            
            # Initialize the QA chain with optimized retriever
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k_value}
            )
            ollama_model = Ollama(base_url="http://localhost:11434", model="mistral")
            st.session_state.qa = RetrievalQA.from_chain_type(llm=ollama_model, retriever=retriever)
            
            # Clean up the temporary file
            os.remove(temp_file_path)
            
            st.success("Document processed successfully!")

# Cache expensive operations
@st.cache_data(ttl=3600)
def get_embeddings(text):
    return st.session_state.ollama_embed_model.embed_query(text)

# Function to calculate semantic similarity
def calculate_semantic_similarity(query_embedding, doc_embedding):
    return np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))

# Display chat messages
st.subheader("Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask a question about the document"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Check if document has been processed
    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            st.markdown("Please upload and process a document first.")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload and process a document first."})
    else:
        # Process the query
        with st.spinner("Thinking..."):
            # Configure retriever with similarity threshold
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": k_value,
                    "score_threshold": similarity_threshold
                }
            )
            
            # Get relevant documents in a single operation
            retrieved_docs = retriever.get_relevant_documents(query)
            
            # Generate response
            if retrieved_docs:
                response = st.session_state.qa({"query": query, "chat_history": []})
                answer = response.get('result', "No result available.")
            else:
                # Fallback to direct LLM if no relevant chunks
                ollama_model = Ollama(base_url="http://localhost:11434", model="mistral")
                answer = ollama_model.invoke(query)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Display retrieved chunks in an expander
                if retrieved_docs:
                    with st.expander("View Retrieved Chunks"):
                        for i, chunk in enumerate(retrieved_docs):
                            st.markdown(f"**Chunk {chunk.metadata['chunk_id']}**")
                            st.text(chunk.page_content[:200] + "...")
                            if i < len(retrieved_docs) - 1:
                                st.divider()
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Clear cache for next query if needed
            if st.button("Clear Cache"):
                st.cache_data.clear()
