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
    if uploaded_files and st.button("Process Documents"):
        with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
            all_docs = []
            temp_files = []
            
            # Process each uploaded file
            file_progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                st.write(f"Processing {uploaded_file.name}...")
                # Save the uploaded file temporarily
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_files.append(temp_file_path)
                
                # Process the document
                docs = preprocess(temp_file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                all_docs.extend(docs)
                
                # Update file processing progress
                file_progress_bar.progress((i + 1) / len(uploaded_files))

            st.write("Storing documents in vector database...")
            # Use a progress bar for storing documents
            store_progress_bar = st.progress(0)
            
            # Process in batches if there are many documents
            batch_size = 50 # Adjust batch size as needed for embedding/storage
            st.session_state.vectorstore = None # Reset vectorstore
            
            for i in range(0, len(all_docs), batch_size):
                batch = all_docs[i:i+batch_size]
                if st.session_state.vectorstore is None:  # First batch creates the store
                    st.session_state.vectorstore = store_documents(batch, st.session_state.ollama_embed_model)
                else:  # Subsequent batches add to the existing store
                    st.session_state.vectorstore.add_documents(batch)
                
                # Update storage progress
                progress = min(1.0, (i + batch_size) / len(all_docs))
                store_progress_bar.progress(progress)
            
            # Initialize the QA chain with standard retriever
            if st.session_state.vectorstore:
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": k_value}
                )
                ollama_model = Ollama(base_url="http://localhost:11434", model="mistral")
                st.session_state.qa = RetrievalQA.from_chain_type(llm=ollama_model, retriever=retriever)
                st.success(f"{len(uploaded_files)} document(s) processed successfully!")
            else:
                 st.error("No documents were processed or stored.")

            # Clean up the temporary files
            for temp_file_path in temp_files:
                try:
                    os.remove(temp_file_path)
                except OSError as e:
                    st.warning(f"Could not remove temporary file {temp_file_path}: {e}")

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
            st.markdown("Please upload and process one or more documents first.")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload and process one or more documents first."})
    else:
        # Process the query
        with st.spinner("Thinking..."):
            # Configure retriever without score_threshold (which isn't supported)
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k_value * 2}  # Retrieve more docs than needed to filter later
            )
            
            # Get relevant documents
            retrieved_docs = retriever.get_relevant_documents(query)
            
            # Get query embedding for filtering
            query_embedding = get_embeddings(query)
            
            # Filter documents by similarity threshold and deduplicate
            filtered_docs = []
            seen_chunk_ids = set()  # Track seen chunk IDs
            
            for doc in retrieved_docs:
                chunk_id = doc.metadata.get("chunk_id", "Unknown")
                
                # Skip if we've already seen this chunk
                if chunk_id in seen_chunk_ids:
                    continue
                
                # Mark this chunk as seen
                seen_chunk_ids.add(chunk_id)
                
                # Get document embedding
                doc_embedding = st.session_state.ollama_embed_model.embed_documents([doc.page_content])[0]
                
                # Calculate similarity
                similarity_score = calculate_semantic_similarity(query_embedding, doc_embedding)
                
                # Filter by threshold
                if similarity_score >= similarity_threshold:
                    filtered_docs.append(doc)
                    
                # Stop if we have enough documents
                if len(filtered_docs) >= k_value:
                    break
            
            # Update retrieved_docs with filtered results
            retrieved_docs = filtered_docs
            
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
                            # Display source filename along with chunk ID
                            source_file = chunk.metadata.get('source_filename', 'Unknown Source')
                            chunk_id = chunk.metadata.get('chunk_id', 'Unknown ID')
                            st.markdown(f"**Source:** {source_file} | **Chunk:** {chunk_id}")
                            st.text(chunk.page_content[:200] + "...")
                            if i < len(retrieved_docs) - 1:
                                st.divider()
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Clear cache for next query if needed
            if st.button("Clear Cache"):
                st.cache_data.clear()
