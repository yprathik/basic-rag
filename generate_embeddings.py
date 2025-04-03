from langchain_community.embeddings import OllamaEmbeddings

class OllamaEmbedModel:
    def __init__(self, base_url="http://localhost:11434"):
        self.model = OllamaEmbeddings(base_url=base_url, model="nomic-embed-text")

    def embed_documents(self, documents):
        return self.model.embed_documents(documents)
    
    def embed_query(self , query):
        #print("inside embed_query function")
        embedding = self.model.embed_query(query)
        return embedding
