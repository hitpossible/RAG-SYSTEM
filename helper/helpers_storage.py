# helpers_storage.py (หรือใส่บน main.py เลยก็ได้)
import os, datetime, re, hashlib, pymysql
from pathlib import Path
from typing import Optional
from db_connect import get_connection

conn = get_connection()

STORAGE_ROOT = Path("data/uploads").resolve()
SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

def safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return SAFE_RE.sub("_", name)

def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def build_storage_key(filename: str) -> str:
    now = datetime.datetime.utcnow()
    return f"{now.year}/{now.month:02d}/{os.urandom(8).hex()}_{safe_filename(filename)}"

def save_bytes_local(data: bytes, storage_key: str) -> Path:
    path = STORAGE_ROOT / storage_key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path

def insert_file_meta(*, owner_user_id: str, role: str, filename: str, mime_type: str,
                     byte_size: int, sha256_hex: str, storage_key: str,
                     session_id: Optional[str] = None, message_id: Optional[str] = None) -> int:
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO files (owner_user_id, role, filename, mime_type, byte_size,
                                   sha256, storage_provider, storage_key, session_id, message_id)
                VALUES (%s,%s,%s,%s,%s,%s,'local',%s,%s,%s)
            """, (owner_user_id, role, filename, mime_type, byte_size,
                  sha256_hex, storage_key, session_id, message_id))
            conn.commit()
            return cur.lastrowid
    finally:
        conn.close()

def upsert_file_text(*, file_id: int, text: str, extractor: str):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO file_texts (file_id, extractor, text)
                VALUES (%s,%s,%s)
                ON DUPLICATE KEY UPDATE extractor=VALUES(extractor), text=VALUES(text)
            """, (file_id, extractor, text))
            conn.commit()
    finally:
        conn.close()

def storage_key_to_url(storage_key: str) -> str:
    return "/uploads/" + storage_key.replace("\\", "/")
