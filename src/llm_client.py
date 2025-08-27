import os
import ollama
from typing import List, Dict, Any, Optional
import re

THINK_PATTERNS = [
    r"(?is)<think>.*?</think>",
    r"(?is)<thought>.*?</thought>",
    r"(?is)<think_start>.*?<think_end>",
    r"(?is)^\s*(?:Thought|Reasoning|Chain-of-thought|思考|推理)\s*:\s*.*?(?=\n\S|\Z)"
]

def strip_think(text: str) -> str:
    out = text or ""
    for pat in THINK_PATTERNS:
        out = re.sub(pat, "", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()

def _basename(p: str) -> str:
    try:
        return os.path.basename(p.replace("\\", "/"))
    except Exception:
        return p or "doc"

def _citation_id(meta: Dict[str, Any]) -> str:
    src = (meta or {}).get("source") or (meta or {}).get("file_name") or "doc"
    base = _basename(str(src))
    # รองรับ page/chunk index ถ้ามี
    idx = (meta or {}).get("chunk_index")
    page = (meta or {}).get("page")
    if idx is not None:
        return f"{base}#{idx}"
    if page is not None:
        return f"{base}@p{page}"
    return base

def _trim_chars(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    # ตัดให้จบประโยคสวย ๆ ถ้าเจอ .!? หรือขึ้นบรรทัดใหม่
    cut = s[:max_chars]
    for sep in [". ", "。", "!", "?", "\n"]:
        j = cut.rfind(sep)
        if j >= int(max_chars*0.7):
            return cut[:j+len(sep)] + " …"
    return cut + " …"

class LlamaClient:
    def __init__(self, model_name: str = "llama3.2:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)

        # ตรวจว่ามีโมเดลไหม
        try:
            models = self.client.list()
            available = [m["model"] for m in models.get("models", [])]
            if model_name not in available:
                print(f"[LlamaClient] Model '{model_name}' not found. Available: {available}")
                print(f"Run: ollama pull {model_name}")
        except Exception as e:
            print(f"[LlamaClient] Error connecting to Ollama: {e}. Make sure 'ollama serve' is running.")

    # ---------- core ----------
    def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[dict]] = None,
        *,
        strict_citation: bool = True,
        ctx_budget_chars: int = 8000,   # งบตัวอักษรสำหรับ context รวม (ประมาณการง่าย ๆ)
        per_chunk_chars: int = 1200,    # จำกัดความยาวต่อชิ้น
        num_predict: int = 1024,        # ใช้กับ Ollama (แทน max_tokens)
        num_ctx: int = 8192,            # context window ของโมเดล (ถ้ารองรับ)
        temperature: float = 0.25
    ) -> str:

        ctx_blocks = []
        if context:
            # เรียง context ตามคะแนนถ้ามี (final_score, rerank_score, similarity…)
            def _score(d):
                return d.get("final_score") or d.get("rerank_score") or d.get("similarity") or (1.0 - float(d.get("distance", 1.0)))
            sorted_ctx = sorted(context, key=_score, reverse=True)

            total = 0
            for d in sorted_ctx:
                meta = d.get("metadata") or {}
                cid = _citation_id(meta)
                raw = d.get("content") or ""
                # block = f"[{cid}] " + _trim_chars(raw, per_chunk_chars)
                block = f"[{cid}] " + raw
                # กัน context ยาวเกิน
                if total + len(block) > ctx_budget_chars:
                    break
                ctx_blocks.append(block)
                total += len(block)
                
        # ---------- prompts ----------
        if ctx_blocks:
            # RAG + บังคับ citation แบบ [<cid>]
            system_prompt = (
                "You are a precise RAG assistant. Use only the provided CONTEXT unless the question "
                "is general and the context is irrelevant. When you use a claim from the context, "
                "If the context is insufficient, say 'Insufficient context' and explain what's missing. "
                "Do not invent citations."
                "ห้ามแสดงกระบวนการคิด ไม่ใส่ <think> หรือคำอธิบายเบื้องหลังใด ๆ"
            )
            user_prompt = (
                "BEGIN CONTEXT\n" +
                "\n\n".join(ctx_blocks) +
                "\nEND CONTEXT\n\n" +
                f"QUESTION:\n{prompt}\n\n"
                "INSTRUCTIONS:\n"
                "- Answer concisely.\n"
                "- Prefer Thai if the question is Thai; otherwise match the user's language.\n"
            )


        else:
            # ไม่มี context: ตอบจากความรู้ทั่วไป แต่ให้ระบุความไม่แน่ใจถ้าจำเป็น
            system_prompt = (
                "You are a helpful assistant. Be accurate and well-reasoned. "
                "If uncertain, say so explicitly."
                "ห้ามแสดงกระบวนการคิด ไม่ใส่ <think> หรือคำอธิบายเบื้องหลังใด ๆ"
            )
            user_prompt = (
                f"USER INPUT:\n{prompt}\n\n"
                "INSTRUCTIONS:\n"
                "- If this looks like a question → answer as QA.\n"
                "- If this looks like a translation request → translate accordingly.\n"
                "- Answer concisely in the expected language."
            )

        messages = [{'role': 'system', 'content': system_prompt}]
        if history:
            # เก็บแค่ท้าย ๆ กัน context ล้น
            messages.extend(history[-10:])
        messages.append({'role': 'user', 'content': user_prompt})

        try:
            resp = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': float(temperature),
                    'top_p': 0.9,
                    'repeat_penalty': 1.1,
                    'num_predict': int(num_predict), 
                    'num_ctx': int(num_ctx), 
                }
            )

            return strip_think(resp['message']['content'])
        except Exception as e:
            return f"Error generating response: {e}"

    # ---------- optional: ฟอร์แมต context พร้อม metadata ----------
    def format_context_with_metadata(self, context: List[Dict[str, Any]]) -> str:
        if not context:
            return ""
        parts = []
        for i, d in enumerate(context):
            meta = d.get('metadata') or {}
            src = meta.get('source') or meta.get('file_name') or d.get('source') or "Unknown"
            cid = _citation_id(meta)
            score = d.get('final_score') or d.get('rerank_score') or d.get('similarity') or (1.0 - float(d.get('distance', 1.0)))
            content = d.get('content') or ''
            parts.append(f"[{cid}] (score: {score:.4f})\nSource: {src}\n---\n{_trim_chars(content, 800)}")
        return "\n\n".join(parts)
    
    def translate_response(self, prompt: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        system_prompt = (
            "You are a helpful translation assistant. Translate the user's input text from the source language to the target language. "
            "If the source language is 'auto', detect the language automatically. "
            "Respond only with the translated text without any additional commentary."
        )
        
        user_prompt = (
            f"Translate the following text from {source_lang} to {target_lang}:\n\n{prompt}\n\n"
            "Provide a clear and accurate translation."
        )

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        try:
            resp = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1,
                    'num_predict': 1024, 
                    'num_ctx': 8192, 
                }
            )
            return strip_think(resp['message']['content'])
        except Exception as e:
            return f"Error generating translation: {e}"
