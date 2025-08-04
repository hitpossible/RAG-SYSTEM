import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Paths
    DOCUMENTS_PATH = "data/documents"
    VECTOR_DB_PATH = "data/vector_db"
    
    # Model Settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama3.2:latest"  # Ollama model name
    
    # RAG Parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5
    
    # Ollama Settings
    OLLAMA_BASE_URL = "http://localhost:11434"

settings = Settings()