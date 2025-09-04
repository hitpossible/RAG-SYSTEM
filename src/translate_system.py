import re
from typing import List, Dict, Any, Optional
from src.llm_client import LlamaClient
from config.settings import settings
from db_connect import get_connection

class Translate():
    def __init__(self):
        self.llm_client = LlamaClient(
            model_name=settings.LLM_MODEL,
            base_url=settings.LLM_BASE_URL
        )
    
    def insert_message(self, session_id, role, content):
        conn = get_connection()
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

    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "en", session_id: str = None) -> str:
        if session_id:
            self.insert_message(session_id, 'user', text)

        try:
            answer = self.llm_client.translate_response(
                prompt=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            if session_id:
                self.insert_message(session_id, 'assistant', answer)
            return answer
        except Exception as e:
            return f"Error generating response: {e}"