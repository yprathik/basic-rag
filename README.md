This is my first implementation of RAG application.# PDF RAG Application

This is a Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents and ask questions about their content.

## Features

- Upload and process PDF documents
- Chunk documents with customizable settings
- Ask questions about the document content
- View retrieved chunks used for generating answers
- Adjust similarity thresholds and retrieval parameters

## Requirements

- Python 3.8+
- Ollama running locally with the "mistral" and "nomic-embed-text" models

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Make sure Ollama is running with the required models:
   ```
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run ui.py
   ```
2. Upload a PDF document using the file uploader in the sidebar
3. Adjust chunking and retrieval settings if needed
4. Click "Process Document" to analyze the PDF
5. Ask questions in the chat input at the bottom of the page

## How It Works

1. The application uses PyPDF to load and extract text from PDF documents
2. The text is split into chunks using RecursiveCharacterTextSplitter
3. Chunks are embedded using the Ollama embedding model and stored in a Chroma vector database
4. When you ask a question, the application:
   - Embeds your question
   - Retrieves the most relevant chunks from the vector database
   - Filters chunks based on semantic similarity
   - Uses the Mistral LLM to generate an answer based on the retrieved chunks
