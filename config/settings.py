import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Paths
    DOCUMENTS_PATH = "data/documents"
    VECTOR_DB_PATH = "data/vector_db"
    
    # Model Settings
    EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
    # EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    # EMBEDDING_MODEL = "text-embedding-3-small"
    
    LLM_MODEL = "qwen3:8b"
    
    # RAG Parameters
    CHUNK_SIZE = 900
    CHUNK_OVERLAP = 120
    TOP_K_RESULTS = 10
    
    # Ollama Settings
    OLLAMA_BASE_URL = "http://localhost:11434"

settings = Settings()