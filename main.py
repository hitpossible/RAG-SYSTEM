import os
from src.rag_system import RAGSystem

def setup_directories():
    """Create necessary directories"""
    os.makedirs("data/documents", exist_ok=True)
    os.makedirs("data/vector_db", exist_ok=True)

def main():
    setup_directories()
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Check system info
    print("System Information:")
    info = rag.get_system_info()
    print(f"Vector store: {info['vector_store']}")
    print(f"Embedding model: {info['embedding_model']}")
    print(f"LLM model: {info['llm_model']}")
    
    # Ingest documents (run this once)
    print("\n" + "="*50)
    print("DOCUMENT INGESTION")
    print("="*50)
    rag.ingest_documents()
    
    # Interactive query loop
    print("\n" + "="*50)
    print("RAG SYSTEM READY - Ask your questions!")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit']:
            break
        
        if not question:
            continue
        
        result = rag.query(question)
        
        print(f"\nAnswer: {result['answer']}")
        # print(f"\nSources used ({result['retrieved_docs_count']} documents):")
        # for i, source in enumerate(result['sources'], 1):
        #     print(f"{i}. {source['source']} (similarity: {source['similarity']:.2f})")
        #     print(f"   Preview: {source['content_preview']}")

if __name__ == "__main__":
    main()