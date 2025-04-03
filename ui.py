import streamlit as st
from langchain_community.llms import Ollama  

def chat_interface(retriever, ollama_model):
    st.title("💬 RAG Chatbot")
    st.write("Ask me anything! Type below to chat.")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Type your message here...")

    if query:
        # Append user message and display it immediately
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Retrieve relevant document chunks
        retrieved_docs = retriever.get_relevant_documents(query)

        # Deduplicate chunk IDs and store content
        unique_chunks = {}
        for doc in retrieved_docs:
            chunk_id = doc.metadata.get("chunk_id", "Unknown")
            if chunk_id not in unique_chunks:
                unique_chunks[chunk_id] = doc.page_content  

        # Display retrieved document snippets
        with st.expander("📄 Retrieved Chunks"):
            for chunk_id, content in unique_chunks.items():
                st.markdown(f"**Chunk ID {chunk_id}:** {content[:300]}...")  # Show first 300 characters

        # Pass the full conversation history to the model
        response = ollama_model.invoke(st.session_state.messages)

        # Append AI response and display it
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# Ensure the script runs only when executed directly
if __name__ == "__main__":
    # Placeholder retriever and model (replace with actual instances)
    retriever = None  # Replace with an actual retriever instance
    ollama_model = Ollama()  # Ensure this is correctly initialized

    chat_interface(retriever, ollama_model)

    





"""
import streamlit as st
from langchain_community.llms import Ollama  

def chat_interface():
    st.title("💬 Chatbot")
    st.write("Ask me anything! Type below to chat.")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Type your message here...")

    if query:
        # Initialize model only when needed
        ollama_model = Ollama(base_url="http://localhost:11434", model="mistral")

        # Append user message and display it immediately
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate AI response
        response = ollama_model.invoke(st.session_state.messages)

        # Append AI response and display it immediately
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
"""