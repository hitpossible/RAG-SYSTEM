from config.settings import settings
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.llm_client import LlamaClient
from langchain_community.chat_message_histories import SQLChatMessageHistory
import pymysql

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
        
        # สร้าง directory สำหรับเก็บ database ถ้ายังไม่มี
        os.makedirs("data", exist_ok=True)
        self.memory_db_path = "sqlite:///data/memories.sqlite"

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
    
    def insert_message(self, session_id, role, content):
        conn = pymysql.connect(
            host='localhost',
            port=8889,
            user='root',
            password='root',
            database='ai_db',
            charset='utf8mb4'
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                    (session_id, role, content)
                )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def insert_file_meta(self, message_id, role, filepath, filename=None):
        conn = pymysql.connect(
            host='localhost',
            port=8889,
            user='root',
            password='root',
            database='ai_db',
            charset='utf8mb4'
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO files (message_id, role, file, file_name, created_at) VALUES (%s, %s, %s, %s, NOW())",
                    (message_id, role, filepath, filename)
                )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    # ---- เปลี่ยน method signature ----
    def query(self, question: str, session_id: str = "default", user_id: str = 'anonymouse',
            uploaded_docs: list[dict] | None = None) -> dict:
        print(f"Processing query: {question}")

        use_memory = session_id != "default"
        memory = None
        history = []

        if use_memory:
            from langchain_community.chat_message_histories import SQLChatMessageHistory
            memory = SQLChatMessageHistory(
                connection=self.memory_db_path,
                session_id=session_id,
                table_name="message_store"
            )
            memory.add_user_message(question)
            message_id = self.insert_message(session_id, "user", question)
            for msg in memory.messages:
                if msg.type == "human":
                    history.append({"role": "user", "content": msg.content})
                elif msg.type == "ai":
                    history.append({"role": "assistant", "content": msg.content})

            if uploaded_docs:
                for ud in uploaded_docs:
                    self.insert_file_meta(message_id, "user", ud['filepath'], ud['filename'])

        # Retrieve จากเวกเตอร์สโตร์
        retrieved_docs = self.vector_store.similarity_search(
            question,
            k=settings.TOP_K_RESULTS
        )

        # กรองระยะห่าง
        filtered_docs = [doc for doc in retrieved_docs if -0.3 <= 1 - doc.get('distance', 1) <= 0.3]

        # --- แนบไฟล์อัปโหลด (ถ้ามี) เข้าบริบทด้วย ---
        if uploaded_docs:
            for ud in uploaded_docs:
                content = ud["content"]
                preview = content[:200] + ("..." if len(content) > 200 else "")
                filtered_docs.append({
                    "content": content[:4000],  # กันยาวเกินบริบท
                    "metadata": {
                        "source": f"Uploaded:{ud.get('filename','unknown')}",
                        "type": ud.get("kind", "upload"),
                        "role": "user"
                    },
                    "distance": 0.0  # treat as top relevance
                })
        
        # เรียก LLM พร้อม history + docs
        answer = self.llm_client.generate_response(
            question, filtered_docs, history=history if use_memory else None
        )

        if use_memory and memory is not None:
            memory.add_ai_message(answer)
            message_id = self.insert_message(session_id, "assistant", answer)

            unique_sources = set()
            for doc in filtered_docs:
                meta = (doc.get('metadata') or {})
                role = (meta.get('role') or '').lower()
                if role == 'user':
                    continue  # ข้ามเอกสารที่มาจาก user

                raw_src  = meta.get('source') or doc.get('source', '')
                link_src = (raw_src or '').replace('\\', '/').replace('data/documents/', '/files/')

                if link_src and link_src not in unique_sources:
                    unique_sources.add(link_src)
                    self.insert_file_meta(message_id, "assistant", link_src, link_src.split('/')[-1])

        sources = []
        for doc in filtered_docs:
            meta = (doc.get('metadata') or {})
            role = (meta.get('role') or '').lower()
            if role == 'user':
                continue

            raw_src  = meta.get('source') or doc.get('source', '')
            link_src = (raw_src or '').replace('\\', '/').replace('data/documents/', '/files/')

            sources.append({
                "source": link_src or "Unknown",
                "similarity": 1 - abs(doc.get('distance', 0.0)),
                "content_preview": (doc.get('content','')[:200] + "...") if doc.get('content') else ""
            })


        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs_count": len(filtered_docs)
        }

    def get_chat_history(self, session_id: str = "default") -> list:
        """ดึงประวัติการสนทนา"""
        memory = SQLChatMessageHistory(
            connection=self.memory_db_path,
            session_id=session_id,
            table_name="message_store"
        )
        
        history = []
        for msg in memory.messages:
            history.append({
                "type": msg.type,
                "content": msg.content,
                "timestamp": getattr(msg, 'additional_kwargs', {}).get('timestamp', None)
            })
        return history

    def clear_chat_history(self, session_id: str = "default"):
        """ลบประวัติการสนทนา"""
        memory = SQLChatMessageHistory(
            connection=self.memory_db_path,
            session_id=session_id,
            table_name="message_store"
        )
        memory.clear()
        print(f"Chat history for session '{session_id}' cleared!")

    def get_system_info(self):
        """Get system information"""
        return {
            "vector_store": self.vector_store.get_collection_info(),
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL,
            "memory_db": self.memory_db_path
        }