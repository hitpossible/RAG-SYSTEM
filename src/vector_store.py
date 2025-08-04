import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from langchain.schema import Document
import uuid

class VectorStore:
    def __init__(self, db_path: str, embedding_model: str):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.embedding_model = SentenceTransformer(embedding_model)
        self.collection_name = "documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        
        # Format results
        documents = []
        for i in range(len(results['documents'][0])):
            doc = {
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            }
            documents.append(doc)
        
        return documents
    
    def get_collection_info(self):
        """Get information about the collection"""
        return {
            "count": self.collection.count(),
            "name": self.collection_name
        }