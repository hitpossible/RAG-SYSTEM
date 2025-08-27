from config.settings import settings
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.llm_client import LlamaClient
from langchain_community.chat_message_histories import SQLChatMessageHistory
import pymysql
import os
from typing import List, Dict, Any, Optional

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

        os.makedirs("data", exist_ok=True)
        self.memory_db_path = "sqlite:///data/memories.sqlite"

    def ingest_documents(self, directory_path: str = None):
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
            host='localhost', port=8889, user='root', password='root',
            database='ai_db', charset='utf8mb4'
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
            host='localhost', port=8889, user='root', password='root',
            database='ai_db', charset='utf8mb4'
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

    def query(
        self,
        question: str,
        session_id: str = "default",
        user_id: str = 'anonymouse',
        uploaded_docs: Optional[List[Dict[str, Any]]] = None
    ) -> dict:
        print(f"Processing query: {question}")

        use_memory = session_id != "default"
        memory = None
        history = []

        if use_memory:
            memory = SQLChatMessageHistory(
                connection=self.memory_db_path,
                session_id=session_id,
                table_name="message_store"
            )
            memory.add_user_message(question)
            msg_id_user = self.insert_message(session_id, "user", question)

            # เผื่อไฟล์ที่อัปโหลดมากับคำถาม
            if uploaded_docs:
                for ud in uploaded_docs:
                    vpath = ud.get('filepath') or f"/uploads/{ud.get('filename','unknown')}"
                    self.insert_file_meta(msg_id_user, "user", vpath, ud.get('filename'))

            # สร้าง history ให้ LLM
            for msg in memory.messages:
                if msg.type == "human":
                    history.append({"role": "user", "content": msg.content})
                elif msg.type == "ai":
                    history.append({"role": "assistant", "content": msg.content})

        # ----- Retrieve จากเวกเตอร์สโตร์ -----
                # ----- Retrieve จากเวกเตอร์สโตร์ -----
        retrieved_docs = self.vector_store._enhanced_similarity_search(
            question,
            k=max(10, settings.TOP_K_RESULTS)  # ดึงมาเยอะขึ้นนิด เพื่อเปิดทาง rerank
        )

        # 🔧 คำนวณ similarity (cosine) และเตรียมค่าไว้
        scored = []
        for d in retrieved_docs:
            dist = float(d.get('distance', 1.0))
            sim = 1.0 - dist
            d['similarity'] = sim
            # บางเวอร์ชันของคุณมี 'rerank_score' อยู่แล้วจาก cross-encoder
            d['rerank_score'] = float(d.get('rerank_score', 0.0))
            if sim >= 0.35:
                scored.append(d)

        if not scored:
            filtered_docs = []
        else:
            # ----- Adaptive threshold + similarity band -----
            scored.sort(key=lambda x: x['similarity'], reverse=True)
            best_sim = scored[0]['similarity']
            band = 0.08  # ปรับได้ 0.06–0.12
            min_band_sim = max(0.40, best_sim - band)

            band_docs = [d for d in scored if d['similarity'] >= min_band_sim]

            # ----- Diversity by source (anti-duplicate) -----
            # จำกัดไม่ให้แหล่งใดแหล่งหนึ่งกินสัดส่วนเกินไป
            max_per_source = 3
            picked = []
            per_source = {}
            for d in band_docs:
                meta = (d.get('metadata') or {})
                raw_src = meta.get('source') or d.get('source', '') or 'Unknown'
                src = str(raw_src).replace('\\', '/')
                if per_source.get(src, 0) < max_per_source:
                    picked.append(d)
                    per_source[src] = per_source.get(src, 0) + 1

            # ถ้ารวมน้อยไปให้เสริมจากลิสต์เดิม
            need = max(settings.TOP_K_RESULTS, 6)
            if len(picked) < need:
                for d in scored:
                    if d in picked: 
                        continue
                    meta = (d.get('metadata') or {})
                    raw_src = meta.get('source') or d.get('source', '') or 'Unknown'
                    src = str(raw_src).replace('\\', '/')
                    if per_source.get(src, 0) < max_per_source:
                        picked.append(d)
                        per_source[src] = per_source.get(src, 0) + 1
                    if len(picked) >= need:
                        break

            # ----- Stitch by source (รวมชิ้นจากแหล่งเดียวกัน) -----
            # รวมเนื้อหา 1–3 ชิ้นแรกของแต่ละแหล่ง เป็นบล็อคเดียว
            from collections import defaultdict
            group = defaultdict(list)
            for d in picked:
                meta = (d.get('metadata') or {})
                raw_src = meta.get('source') or d.get('source', '') or 'Unknown'
                src = str(raw_src).replace('\\', '/')
                group[src].append(d)

            stitched_docs = []
            for src, items in group.items():
                # เรียงตาม (rerank_score, similarity)
                items.sort(key=lambda x: (x.get('rerank_score', 0.0), x['similarity']), reverse=True)
                # ต่อสูงสุด N ชิ้นแรกเพื่อให้ได้คอนเท็กซ์ยาวพอ แต่ไม่ยาวเกิน
                MAX_JOIN = 3
                MAX_CHARS = 1800
                contents = []
                total = 0
                take = 0
                for it in items:
                    c = (it.get('content') or '')
                    if not c:
                        continue
                    # กันยาวเกิน
                    to_add = c[:MAX_CHARS - total]
                    if to_add.strip():
                        contents.append(to_add)
                        total += len(to_add)
                        take += 1
                    if take >= MAX_JOIN or total >= MAX_CHARS:
                        break
                if not contents:
                    continue

                # สร้างเอนทรีใหม่แบบ stitched
                base = items[0]
                stitched_docs.append({
                    "content": "\n\n".join(contents),
                    "metadata": base.get('metadata') or {},
                    "distance": float(base.get('distance', 1.0)),
                    "similarity": float(base.get('similarity', 0.0)),
                    "rerank_score": float(base.get('rerank_score', 0.0)),
                    "source_key": src
                })

            # ----- Final rerank (ออปชัน: กรณีต้องการจัดลำดับรอบสุดท้าย) -----
            # ถ้าอยากเข้มขึ้นและมี cross-encoder พร้อมใช้งานผ่าน vector_store:
            try:
                ce = self.vector_store.rerank_model  # CrossEncoder พร้อมแล้ว
                pairs = [(question, d['content']) for d in stitched_docs]
                if pairs:
                    scores = ce.predict(pairs)
                    for d, s in zip(stitched_docs, scores):
                        d['final_score'] = 0.5 * float(s) + 0.3 * d.get('similarity', 0.0) + 0.2 * max(0.0, 1.0 - float(d.get('distance', 1.0)))
                else:
                    for d in stitched_docs:
                        d['final_score'] = 0.3 * d.get('similarity', 0.0)
            except Exception:
                # fallback: ใช้ similarity เป็นหลัก
                for d in stitched_docs:
                    d['final_score'] = 0.7 * d.get('similarity', 0.0) + 0.3 * max(0.0, 1.0 - float(d.get('distance', 1.0)))

            stitched_docs.sort(key=lambda x: x['final_score'], reverse=True)

            # คัดจำนวนสุดท้ายที่จะส่งให้ LLM
            MAX_CONTEXT_BLOCKS = 6
            filtered_docs = stitched_docs[:MAX_CONTEXT_BLOCKS]


        # แนบไฟล์อัปโหลด (ให้ความสำคัญสูงสุด)
        if uploaded_docs:
            for ud in uploaded_docs:
                content = (ud.get("content") or "")
                if not content.strip():
                    continue
                filtered_docs.insert(0, {  # ดันขึ้นต้น
                    "content": content[:4000],
                    "metadata": {
                        "source": f"Uploaded:{ud.get('filename','unknown')}",
                        "type": ud.get("kind", "upload"),
                        "role": "user"
                    },
                    "distance": 0.0,
                    "similarity": 1.0,
                    "final_score": 999.0
                })

        # ----- Guardrail: ถ้าความมั่นใจต่ำให้ตอบแบบไม่เดา -----
        avg_sim = sum(d.get('similarity', 0.0) for d in filtered_docs) / max(1, len(filtered_docs))
        avg_final = sum(d.get('final_score', 0.0) for d in filtered_docs) / max(1, len(filtered_docs))
        low_confidence = (avg_sim < 0.50) and (avg_final < 0.60)

        # ----- เรียก LLM พร้อมบริบท -----
        answer = self.llm_client.generate_response(
            question if not low_confidence else f"""\
คำถาม: {question}

ข้อกำหนดความปลอดภัย:
- หากบริบทที่ให้มาไม่ตรงกับคำถามหรือไม่เพียงพอ ห้ามเดา
- ให้ตอบว่า "ยังไม่พบข้อมูลเพียงพอในเอกสาร" และสรุปว่าขาดอะไร
""",
            filtered_docs,
            history=history if use_memory else None
        )

        # ----- บันทึกคำตอบ + ไฟล์อ้างอิงจาก assistant -----
        if use_memory and memory is not None:
            memory.add_ai_message(answer)
            msg_id_ai = self.insert_message(session_id, "assistant", answer)

            unique_sources = set()
            for doc in filtered_docs:
                meta = (doc.get('metadata') or {})
                role = (meta.get('role') or '').lower()
                if role == 'user':
                    continue  # ข้ามไฟล์ของ user

                raw_src = meta.get('source') or doc.get('source', '') or doc.get('source_key', '')
                link_src = (str(raw_src) or '').replace('\\', '/').replace('data/documents/', '/files/')
                if link_src and link_src not in unique_sources:
                    unique_sources.add(link_src)
                    self.insert_file_meta(msg_id_ai, "assistant", link_src, link_src.split('/')[-1])

        # ----- สร้าง sources สำหรับตอบกลับ -----
        sources = []
        for doc in filtered_docs:
            meta = (doc.get('metadata') or {})
            role = (meta.get('role') or '').lower()
            if role == 'user':
                continue  # ไม่โชว์ไฟล์ที่ผู้ใช้อัปโหลดเป็นแหล่งอ้างอิง

            raw_src = meta.get('source') or doc.get('source', '') or doc.get('source_key', '')
            link_src = (str(raw_src) or '').replace('\\', '/').replace('data/documents/', '/files/')

            sources.append({
                "source": link_src or "Unknown",
                "similarity": float(doc.get('similarity', 0.0)),
                "content_preview": (doc.get('content','')[:200] + "...") if doc.get('content') else ""
            })


        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs_count": len(filtered_docs)
        }

    def get_chat_history(self, session_id: str = "default") -> list:
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
        memory = SQLChatMessageHistory(
            connection=self.memory_db_path,
            session_id=session_id,
            table_name="message_store"
        )
        memory.clear()
        print(f"Chat history for session '{session_id}' cleared!")

    def get_system_info(self):
        return {
            "vector_store": self.vector_store.get_collection_info(),
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL,
            "memory_db": self.memory_db_path
        }
    
