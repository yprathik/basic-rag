#Import necessary libraries

from document_operations import preprocess, store_documents
from generate_embeddings import OllamaEmbedModel
from langchain_community.llms import Ollama 
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import numpy as np

file_path = "/home/skykarthik/ai-code/basic-rag/16.-Streamlit.pdf"

docs = preprocess(file_path)

ollama_embed_model = OllamaEmbedModel()

vectorstore = store_documents(docs, ollama_embed_model)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Fetch top 5 results

ollama_model = Ollama(base_url="http://localhost:11434", model="mistral")

qa = RetrievalQA.from_chain_type(llm=ollama_model, retriever=retriever)


def calculate_semantic_similarity(query_embedding , doc_embedding):
    return np.dot(query_embedding , doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))

while True:
    query = input("\nHuman: ")

    if query.lower() == "exit":
        print("GOODBYE!")
        break
    
    query_embeddings = ollama_embed_model.embed_query(query)
    retrieved_docs = retriever.get_relevant_documents(query)

    unique_chunks = {}
    relevant_chunks = []

    for doc in retrieved_docs:
        chunk_id = doc.metadata.get("chunk_id", "Unknown")
        if chunk_id not in unique_chunks:
            unique_chunks[chunk_id] = doc.page_content  

            doc_embeddings = ollama_embed_model.embed_documents([doc.page_content])

            similarity_score = calculate_semantic_similarity(query_embeddings , doc_embeddings[0])

            if similarity_score >= 0.5:
                relevant_chunks.append(doc)
    
    if relevant_chunks:
        print("\nRetrieved Chunks: ")
        for chunk in relevant_chunks:
            content = chunk.page_content
            print(f"\nChunk ID: {chunk.metadata['chunk_id']}\nContent:{content[:200]}....")
        
        response = qa({"query" : query , "chat_history": []})

        print("\nAI")

        if 'result' in response:
            print(response['result'])

        else:
            print("No result available.")
    
    else:
        response = ollama_model.invoke(query)
        print("LLM response:")

        print(response)


