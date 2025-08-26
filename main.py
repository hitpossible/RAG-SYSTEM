import os
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from src.rag_system import RAGSystem
import asyncio
import json
import uuid
from typing import List, Optional
import base64
from src.file_extractors import sniff_and_extract

from helpers.helpers_storage import (
    compute_sha256, build_storage_key, save_bytes_local,
    insert_file_meta, upsert_file_text, storage_key_to_url
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# --- Setup directories ---
def setup_directories():
    os.makedirs("data/documents", exist_ok=True)
    os.makedirs("data/vector_db", exist_ok=True)

# --- Init ---
setup_directories()
rag = RAGSystem()

# --- FastAPI app ---
app = FastAPI(title="RAG API")

app.mount("/files", StaticFiles(directory="data/documents"), name="files")

# --- CORS (ให้เชื่อมจาก frontend ได้) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict by domain here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ClientFile(BaseModel):
    name: str
    type: Optional[str] = ""
    size: Optional[int] = 0
    data: str 

# --- Pydantic Model ---
class QueryRequest(BaseModel):
    text: str
    session_id: str = "default"
    user_id: str = "anonymous"
    files: Optional[List[ClientFile]] = None

# --- Endpoint: System Info ---
@app.get("/system-info")
def get_system_info():
    info = rag.get_system_info()
    return {
        "vector_store": info["vector_store"],
        "embedding_model": info["embedding_model"],
        "llm_model": info["llm_model"]
    }

async def event_stream_v2(answer: str):
    item_id = str(uuid.uuid4())
    output_index = 0

    # เปิด stream แบบ structured event format ของ Vercel AI SDK
    yield f'data: {json.dumps({"type": "response.output_text.delta", "item_id": item_id, "delta": "🤖 "})}\n\n'
    await asyncio.sleep(0.1)

    for word in answer.split():
        yield f'data: {json.dumps({"type": "response.output_text.delta", "item_id": item_id, "delta": word + " "})}\n\n'
        await asyncio.sleep(0.05)

    # ปิดรายการตอบกลับ
    yield f'data: {json.dumps({"type": "response.output_item.done", "output_index": output_index, "item": {"type": "text", "text": answer}})}\n\n'

    # แจ้งว่าเสร็จสิ้นทั้งหมด
    yield f'data: {json.dumps({"type": "response.completed"})}\n\n'

# --- Endpoint: Query (Streaming Response) ---
@app.post("/query/stream/responses")
async def query_stream(request: Request):
    try:
        body = await request.json()
        input_messages = body.get("input", [])
        question_parts = []

        for msg in input_messages:
            if msg.get("role") == "user":
                for content in msg.get("content", []):
                    if content.get("type") == "input_text":
                        question_parts.append(content.get("text", ""))

        question = " ".join(question_parts).strip()

        if not question:
            return JSONResponse(content={"error": "Empty input"}, status_code=400)

        # ดึงคำตอบจาก model ของคุณ
        result = rag.query(question)
        answer = result["answer"]

        return StreamingResponse(event_stream_v2(answer), media_type="text/event-stream")

    except Exception as e:
        print("Error:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --- Endpoint: Query (Normal JSON) ---
@app.post("/query")
# async def query_json(request: QueryRequest):
#     question = request.text.strip()
#     session_id = request.session_id
#     user_id = request.user_id

#     if not question:
#         return JSONResponse(content={"error": "Empty question"}, status_code=400)

#     result = rag.query(question, session_id, user_id)

#     return {
#         "answer": result["answer"],
#         "retrieved_docs_count": result["retrieved_docs_count"],
#         "sources": result["sources"]
#     }
async def query_json(request: QueryRequest):
    question = (request.text or "").strip()
    if not question:
        return JSONResponse({"error": "Empty question"}, status_code=400)

    uploaded_docs = []
    for f in (request.files or []):
        try:
            data_bytes = base64.b64decode(f.data)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid base64 for file: {f.name}")

        try:
            extracted, kind = sniff_and_extract(f.name, f.type or "", data_bytes)
        except ValueError as ve:
            # ประเภทไฟล์ไม่รองรับ
            raise HTTPException(status_code=415, detail=str(ve))
        except RuntimeError as re:
            # ขาดไลบรารีที่จำเป็นของ extractor
            raise HTTPException(status_code=500, detail=str(re))
        
        storage_key = build_storage_key(f.name)
        save_bytes_local(data_bytes, storage_key)

        if extracted.strip():
            uploaded_docs.append({
                "filename": f.name,
                "filepath": storage_key_to_url(storage_key),
                "kind": kind,
                "content": extracted
            })

    # ยิงเข้า RAG พร้อมไฟล์ที่แตกข้อความแล้ว
    result = rag.query(
        question,
        session_id=request.session_id,
        user_id=request.user_id,
        uploaded_docs=uploaded_docs
    )

    return {
        "answer": result["answer"],
        "retrieved_docs_count": result["retrieved_docs_count"],
        "sources": result["sources"]
    }


# --- Endpoint: Ingest Documents ---
@app.post("/ingest")
async def ingest_documents():
    try:
        rag.ingest_documents()
        return {"status": "success", "message": "Documents ingested successfully."}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --- Endpoint: Clear Chat History ---
@app.post("/chat/clear")
async def clear_chat_history(session_id: str = Query("default", description="Session ID to clear")):
    try:
        rag.clear_chat_history(session_id)
        return {"status": "success", "message": f"Chat history for session '{session_id}' cleared."}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --- Entry point ---
if __name__ == "__main__":
    rag.ingest_documents() 
    print("RAG system initialized and documents ingested.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
