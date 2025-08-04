import streamlit as st
from src.rag_system import RAGSystem
import os

# Page config
st.set_page_config(
    page_title="TBKK Knowledge Base",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

# Sidebar
st.sidebar.title("RAG System Control")

# Initialize system button
if st.sidebar.button("Initialize RAG System"):
    with st.spinner("Initializing RAG system..."):
        try:
            os.makedirs("data/documents", exist_ok=True)
            os.makedirs("data/vector_db", exist_ok=True)
            st.session_state.rag_system = RAGSystem()
            st.sidebar.success("RAG system initialized!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# Document ingestion
if st.session_state.rag_system is not None:
    if st.sidebar.button("Ingest Documents"):
        with st.spinner("Ingesting documents..."):
            try:
                st.session_state.rag_system.ingest_documents()
                st.sidebar.success("Documents ingested!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
    
    # System info
    if st.sidebar.button("Show System Info"):
        info = st.session_state.rag_system.get_system_info()
        st.sidebar.json(info)

# Main interface
st.title("🤖 TBKK Knowledge Base")
st.markdown("Ask questions about your documents!")

# File uploader
uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT)",
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"Uploaded {len(uploaded_files)} files. Click 'Ingest Documents' to process them.")

# Query interface
if st.session_state.rag_system is not None:
    question = st.text_input("Ask a question:")
    
    if st.button("Get Answer") and question:
        with st.spinner("Generating answer..."):
            try:
                result = st.session_state.rag_system.query(question)
                
                st.subheader("Answer:")
                st.write(result['answer'])
                
                st.subheader("Sources:")
                for i, source in enumerate(result['sources'], 1):
                    with st.expander(f"Source {i}: {source['source']} (Similarity: {source['similarity']:.2f})"):
                        st.write(source['content_preview'])
                        
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.warning("Please initialize the RAG system first using the sidebar.")