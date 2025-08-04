from config.settings import settings
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.llm_client import LlamaClient
import os

class RAGSystem:
    def __init__(self):
        self.document_loader = DocumentLoader(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        self.vector_store = VectorStore(
            db_path=settings.VECTOR_DB_PATH,
            embedding_model=settings.EMBEDDING_MODEL
        )
        
        self.llm_client = LlamaClient(
            model_name=settings.LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
    
    def ingest_documents(self, directory_path: str = None):
        """Ingest documents into the vector store"""
        if directory_path is None:
            directory_path = settings.DOCUMENTS_PATH
        documents = self.document_loader.load_directory(directory_path)
        if documents:
            print("Adding documents to vector store...")
            self.vector_store.add_documents(documents)
            print("Document ingestion completed!")
        else:
            print("No documents found to ingest")
    
    def query(self, question: str) -> dict:
        """Query the RAG system and handle both specific and general knowledge questions"""
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search(
            question, 
            k=settings.TOP_K_RESULTS
        )

        if not retrieved_docs:
            # หากไม่พบเอกสารที่เกี่ยวข้อง, ให้ LLM ตอบคำถามทั่วไป
            answer = self.llm_client.generate_response(question)
            return {
                "answer": answer,
                "sources": [],
                "retrieved_docs_count": 0
            }
        
        # Filter out documents with low similarity (e.g., below 0.5)
        filtered_docs = []
        for doc in retrieved_docs:
            similarity = 1 - doc['distance']  # Convert distance to similarity
            filtered_docs.append(doc)
            
        if not filtered_docs:
            # หากไม่พบเอกสารที่มีความคล้ายคลึงเพียงพอ, ให้ LLM ตอบคำถามทั่วไป
            answer = self.llm_client.generate_response(question)
            return {
                "answer": answer,
                "sources": [],
                "retrieved_docs_count": 0
            }
        
        # Generate response using relevant documents
        answer = self.llm_client.generate_response(question, filtered_docs)
        
        # Format sources
        sources = []
        for doc in filtered_docs:
            sources.append({
                "source": doc['metadata'].get('source', 'Unknown'),
                "similarity": 1 - doc['distance'],  # Convert distance to similarity
                "content_preview": doc['content'][:200] + "..."
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs_count": len(filtered_docs)
        }

    
    def get_system_info(self):
        """Get system information"""
        return {
            "vector_store": self.vector_store.get_collection_info(),
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL
        }