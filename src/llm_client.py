import json
import os
import ollama
from typing import List, Dict, Any, Optional
import re

from requests import JSONDecodeError
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
        ctx_budget_chars: int = 8000,   
        per_chunk_chars: int = 1200,    
        num_predict: int = 4096,        
        num_ctx: int = 8192,           
        temperature: float = 0.3,
        predict_followups: int = 2
    ) -> str:
        import json

        def _score(d):
            return d.get("final_score") or d.get("rerank_score") or d.get("similarity") or (1.0 - float(d.get("distance", 1.0)))

        def _citation_id(meta):
            # ใส่ตามของเดิมคุณ
            return meta.get("cid") or meta.get("id") or meta.get("source") or "C"

        # ---------------- build context blocks ----------------
        ctx_blocks = []
        if context:
            sorted_ctx = sorted(context, key=_score, reverse=True)

            total = 0
            for d in sorted_ctx:
                meta = d.get("metadata") or {}
                cid = _citation_id(meta)
                raw = d.get("content") or ""
                block = f"[{cid}] " + raw
                if total + len(block) > ctx_budget_chars:
                    break
                ctx_blocks.append(block)
                total += len(block)

        print(context)

        # ---------------- original system/user prompt (UNTOUCHED) ----------------
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
                "โปรดตอบให้ละเอียด ครอบคลุมทุกประเด็นในคำถาม "
            )

        else:
            system_prompt = (
                "คุณเป็นผู้ช่วยที่เป็นมิตร ตอบด้วยความรู้ทั่วไปให้ถูกต้อง กระชับ และชัดเจน "
                "ให้คำตอบตามสามัญความรู้ของโลกและความเข้าใจทั่วไปได้ตามปกติ "
                "หากเป็นเพียงการทักทาย เช่น 'สวัสดี' หรือ 'ดีครับ' ให้ตอบกลับด้วยคำทักทายที่เหมาะสม "
                "หากคำถามเป็นความรู้ทั่วไป ให้ตอบตามความรู้ที่มี "
                "หากคำถามเป็นการขอแปลภาษา เขียนอีเมล หรือจัดรูปแบบข้อความ "
                "รายละเอียดเชิงตัวเลขหรือวันที่ล่าสุด ซึ่งคุณไม่สามารถทราบได้จากความรู้ทั่วไป "
                # "ตอบเฉพาะคำตอบสุดท้าย ไม่ต้องอธิบายขั้นตอนการคิด"
                "***do not think "
            )

            user_prompt = (
                f"คำถาม:\n{prompt}\n\n"
                "โปรดตอบให้ละเอียด ครอบคลุมทุกประเด็นในคำถาม "
            )

        # ---------------- ADD-ONLY: JSON + follow-ups directives ----------------
        # (เพิ่มเข้าไปหลังจากสร้างข้อความเดิม เพื่อไม่แก้/ลบ system prompt เดิม)
        followup_n = max(0, int(predict_followups or 0))
        schema = '{"answer":"string","followups":[{"q":"string"}]}'
        
        format_directive = (
            "รูปแบบเอาต์พุต: ตอบกลับเป็น JSON ล้วน ๆ เท่านั้น ห้ามมี Markdown/โค้ดบล็อก/คำอธิบายอื่นนอก JSON\n"
            "ข้อกำหนดสำคัญ:\n"
            "- คีย์ต้องมีแค่ answer (string) และ followups (array ของ object ที่มีคีย์ q เป็น string) เท่านั้น\n"
            "- ห้ามห่อ answer เป็น object หรือมีคีย์อื่นเพิ่ม เด็ดขาด\n"
            f"- จำนวน followups: {followup_n}\n"
            "สคีมาที่ต้องยึด:\n" + schema + "\n"
            "ตัวอย่างที่ถูกต้อง:\n"
            '{"answer":"สรุปคำตอบ","followups":[{"q":"อยากให้ขยายความเรื่อง X?"},{"q":"มีตัวอย่างการใช้งานไหม?"}]}\n'
            "ตัวอย่างที่ผิด (ห้าม):\n"
            '{"answer":{"answer":""}}\n'
        )
        system_prompt = system_prompt + "\n\n" + format_directive
        user_prompt = user_prompt + "\n\n" + format_directive

        messages = [{'role': 'system', 'content': system_prompt}]
        if history:
            messages.extend(history[-10:])
        messages.append({'role': 'user', 'content': user_prompt})

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=float(temperature),
                top_p=0.9,
                frequency_penalty=1.1,
                max_tokens=int(num_predict),
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content

            try:
                raw = strip_think(raw)
            except Exception:
                pass

            try:
                obj = json.loads(raw)
            except TypeError as e:
                print("TypeError:", type(raw), e)  
                obj = raw if isinstance(raw, (dict, list)) else {"answer": str(raw), "followups": []}
            except JSONDecodeError as e:
                print("JSONDecodeError:", e) 
                obj = {"answer": raw, "followups": []}

            return obj

        except Exception as e:
            # error ก็ยังคืน JSON เสมอ
            err_obj = {
                "answer": f"Error generating response: {e}",
                "followups": [],
            }
    
        return err_obj

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
