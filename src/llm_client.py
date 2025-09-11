import os
import ollama
from typing import List, Dict, Any, Optional
import re
from openai import OpenAI

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
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", base_url: str = "http://172.21.83.10:11436/v1"):
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

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
            system_prompt = (
                "คุณเป็นผู้ช่วย RAG ที่ตอบอย่างแม่นยำ ใช้เฉพาะข้อมูลจาก CONTEXT เท่านั้น "
                "ยกเว้นคำถามเป็นเรื่องทั่วไปที่ไม่เกี่ยวกับ context ถึงตอบจากความรู้ทั่วไปได้ "
                "หาก context ไม่เพียงพอ ให้ตอบว่า 'ไม่มีข้อมูลเพียงพอ' "
                "และอธิบายว่าขาดอะไร ห้ามเดาหรือแต่งเอง "
                "***do not think "

            )

            user_prompt = (
                "BEGIN CONTEXT\n"
                + "\n\n".join(ctx_blocks)
                + "\nEND CONTEXT\n\n"
                f"คำถาม:\n{prompt}\n\n"
                "โปรดตอบอย่างกระชับ ตรงประเด็น"
            )

        else:
            system_prompt = (
                "คุณเป็นผู้ช่วยที่เป็นมิตร ตอบด้วยความรู้ทั่วไปให้ถูกต้อง กระชับ และชัดเจน "
                "ให้คำตอบตามสามัญความรู้ของโลกและความเข้าใจทั่วไปได้ตามปกติ "
                "หากเป็นเพียงการทักทาย เช่น 'สวัสดี' หรือ 'ดีครับ' ให้ตอบกลับด้วยคำทักทายที่เหมาะสม "
                "หากคำถามเป็นความรู้ทั่วไป ให้ตอบตามความรู้ที่มี "
                "หากคำถามเป็นการขอแปลภาษา เขียนอีเมล หรือจัดรูปแบบข้อความ "
                # "ให้ทำตามคำขอทันที โดยไม่ต้องพึ่ง context "
                # "ทุกคำตอบที่ไม่ได้อ้างอิง context ของบริษัท ต้องระบุด้วยว่า "
                # "'คำตอบนี้ไม่ได้อ้างอิงข้อมูลจากบริษัท เป็นเพียงความรู้ทั่วไป/การแปลข้อความ' "
                "รายละเอียดเชิงตัวเลขหรือวันที่ล่าสุด ซึ่งคุณไม่สามารถทราบได้จากความรู้ทั่วไป "
                "ตอบเฉพาะคำตอบสุดท้าย ไม่ต้องอธิบายขั้นตอนการคิด"
                "***do not think "
            )

            user_prompt = (
                f"คำถาม:\n{prompt}\n\n"
                "โปรดตอบอย่างกระชับ ตรงประเด็น"
            )

        messages = [{'role': 'system', 'content': system_prompt}]
        if history:
            messages.extend(history[-10:])
        messages.append({'role': 'user', 'content': user_prompt})

        print(messages)

        try:
            import datetime
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=float(temperature),
                top_p=0.9,
                frequency_penalty=1.1,
                max_tokens=int(num_predict),
            )
            return strip_think(resp.choices[0].message.content)
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
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=float(0.3),
                top_p=0.9,
                frequency_penalty=1.1,
            )
            return strip_think(resp.choices[0].message.content)
        except Exception as e:
            return f"Error generating translation: {e}"
