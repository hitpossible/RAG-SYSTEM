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
    
    LLM_MODEL = "Qwen/Qwen3-8B"
    # LLM_MODEL = "openai/gpt-oss-20b"
    SLM_MODEL = "qwen3:0.6b"
    # LLM_MODEL = "Qwen3-8B-GGUF:Q4_K_M"
    
    # RAG Parameters
    CHUNK_SIZE = 900
    CHUNK_OVERLAP = 120
    TOP_K_RESULTS = 10
    
    # Ollama Settings
    LLM_BASE_URL = "http://172.21.83.10:11436/v1"

settings = Settings()