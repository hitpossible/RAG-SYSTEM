import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any, Optional
from langchain.schema import Document
import uuid
from pythainlp import word_tokenize
import re
from pythainlp.util import normalize
from collections import Counter
import math

class VectorStore:
    def __init__(self, db_path: str, embedding_model: str):
        # --- ใช้ Chroma เป็น cosine space เพื่อเข้าคู่กับ normalized embeddings ---
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = "documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # <<< สำคัญ
        )

        # --- โมเดลฝั่ง embedding: Jina v3 ---
        # แนะนำให้ส่งค่า embedding_model = "jinaai/jina-embeddings-v3"
        self.embedding_model_id = embedding_model or "jinaai/jina-embeddings-v3"
        self.embedding_model = SentenceTransformer(
            self.embedding_model_id,
            trust_remote_code=True
        )

        # --- Cross-encoder reranker (เดิม) ---
        self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # สถิติเพื่อ TF-IDF (เดิม)
        self.doc_freq: Dict[str, int] = {}
        self.total_docs = 0

        # Thai stopwords (เดิม)
        self.thai_stopwords = {
            'และ', 'หรือ', 'แต่', 'ที่', 'ใน', 'จาก', 'ของ', 'เป็น', 'มี', 'จะ', 'ได้', 'แล้ว',
            'กับ', 'ไป', 'มา', 'ให้', 'ถึง', 'คือ', 'อะไร', 'ไหม', 'มั้ย', 'นะ', 'ครับ', 'ค่ะ',
            'เอ่อ', 'อื่ม', 'อา', 'เออ', 'โอ้', 'อ๋อ', 'เฮ้ย', 'นั่น', 'นี่', 'โน่น', 'เหล่า',
            'ผู้', 'คน', 'บุคคล', 'ตัว', 'อัน', 'สิ่ง', 'การ', 'ความ', 'เรื่อง'
        }

    # -------------------- Utils (เหมือนเดิม + เล็กน้อย) --------------------
    def normalize_thai(self, text: str) -> str:
        text = normalize(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize_thai(self, text: str) -> str:
        tokens = word_tokenize(text, engine="newmm")
        return " ".join(tokens)

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        tokens = word_tokenize(text, engine="newmm")
        filtered = [
            t.lower() for t in tokens
            if len(t) > 1 and t.lower() not in self.thai_stopwords
            and not re.match(r'^[0-9\s\.\,\!\?\-\(\)]+$', t)
        ]
        return [tok for tok, _ in Counter(filtered).most_common(top_k)]

    def build_document_frequency(self, documents: List[Document]):
        self.doc_freq.clear()
        self.total_docs = len(documents)
        if self.total_docs == 0:
            return
        for doc in documents:
            if not getattr(doc, "page_content", None):
                continue
            toks = set(word_tokenize(doc.page_content.lower(), engine="newmm"))
            for t in toks:
                if len(t) > 1 and t not in self.thai_stopwords:
                    self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
        print(f"Built document frequency for {self.total_docs} documents, {len(self.doc_freq)} unique tokens")

    def calculate_tfidf_score(self, query_tokens: List[str], doc_content: str) -> float:
        if self.total_docs == 0 or not query_tokens:
            return 0.0
        dtoks = word_tokenize(doc_content.lower(), engine="newmm")
        if not dtoks:
            return 0.0
        dfreq = Counter(dtoks)
        score = 0.0
        for tok in query_tokens:
            tf = dfreq.get(tok, 0) / len(dtoks)
            if tf == 0: 
                continue
            df = self.doc_freq.get(tok, 1)
            idf = math.log((self.total_docs + 1) / (df + 1)) if df > 0 else 0.0
            score += tf * idf
        return score

    # -------------------- NEW: ตัวช่วย encode แบบฉลาด --------------------
    def _encode_texts(self, texts: List[str], is_query: bool) -> List[List[float]]:
        """
        พยายามใช้พารามิเตอร์ที่เหมาะกับ Instruct-embeddings เช่น Jina v3:
        - query:  task='retrieval.query' / prompt_name='retrieval.query'
        - passage: task='retrieval.passage' / prompt_name='retrieval.passage'
        ถ้าไม่รองรับ → fallback เป็น encode ปกติ
        """
        name = "retrieval.query" if is_query else "retrieval.passage"
        # ลองแบบต่าง ๆ ตามลำดับ
        trials = [
            {"task": name, "normalize_embeddings": True},
            {"prompt_name": name, "normalize_embeddings": True},
            {"normalize_embeddings": True},
        ]
        for kw in trials:
            try:
                embs = self.embedding_model.encode(
                    texts,
                    batch_size=64,
                    **kw
                )
                # sentence-transformers อาจคืนเป็น np.ndarray หรือ list
                return embs.tolist() if hasattr(embs, "tolist") else embs
            except TypeError:
                # โมเดลไม่รับพารามิเตอร์นี้ → ลองแบบถัดไป
                continue
        # fallback สุดท้าย
        embs = self.embedding_model.encode(texts, batch_size=64)
        return embs.tolist() if hasattr(embs, "tolist") else embs

    # -------------------- Ingest --------------------
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        for k, v in metadata.items():
            if v is None or isinstance(v, (str, int, float, bool)):
                cleaned[k] = v
            elif isinstance(v, list):
                cleaned[k] = ", ".join(str(x) for x in v)
            elif isinstance(v, dict):
                cleaned[k] = str(v)
            else:
                cleaned[k] = str(v)
        return cleaned

    def add_documents(self, documents: List[Document]):
        """
        แนะนำ chunk_size ≈ 900–1100 tokens, overlap ≈ 120–200 (คุณตั้ง 1000/200 ไว้แล้วดี)
        เพิ่ม boost ด้วย title/section ในข้อความฝั่ง embeddings
        """
        if not documents:
            return

        self.build_document_frequency(documents)

        texts: List[str] = []
        embed_texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for i, doc in enumerate(documents):
            raw = doc.page_content or ""
            meta = dict(doc.metadata or {})

            # --- ทำความสะอาด + tokenize ภาษาไทย ---
            norm = self.normalize_thai(raw)
            tokenized = self.tokenize_thai(norm)

            # --- boost จาก title/section ---
            title = (meta.get("title") or meta.get("file_name") or "").strip()
            section = (meta.get("section") or meta.get("heading") or "").strip()
            prefix = ""
            if title:
                prefix += f"[TITLE] {title}\n"
            if section:
                prefix += f"[SECTION] {section}\n"

            embed_txt = (prefix + tokenized).strip()

            texts.append(raw)
            embed_texts.append(embed_txt)

            # enrich metadata
            enriched = meta.copy()
            keywords = self.extract_keywords(raw)
            enriched["keywords"] = ", ".join(keywords) if keywords else ""
            enriched["word_count"] = len(word_tokenize(raw, engine="newmm"))
            enriched["char_count"] = len(raw)
            enriched["full_content"] = prefix + raw   # ✅ เก็บเนื้อหาจริง + title/section
            metadatas.append(self._clean_metadata(enriched))


        # --- สร้าง embeddings (passage mode + normalize) ---
        embeddings = self._encode_texts(embed_texts, is_query=False)

        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        self.collection.add(
            embeddings=embeddings,
            documents=texts,         # เก็บข้อความดิบ
            metadatas=metadatas,
            ids=ids,
        )
        print(f"Added {len(documents)} documents to vector store (model={self.embedding_model_id}, cosine+normalized)")

    # -------------------- Retrieval (ใช้ query-mode + normalize) --------------------
    def _query_collection(self, query_embedding, n_results: int):
        return self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

    def _basic_similarity_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        q_emb = self._encode_texts([query], is_query=True)
        results = self._query_collection(q_emb, n_results=k)

        candidate_docs = []
        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i] or {}
            candidate_text = meta.get("full_content") or results['documents'][0][i]

            candidate_docs.append({
                'content': candidate_text,
                'metadata': meta,
                'distance': results['distances'][0][i],
                'id': results['ids'][0][i],
            })

        # rerank
        pairs = [(query, d['content']) for d in candidate_docs]
        rr = self.rerank_model.predict(pairs) if pairs else []
        reranked = sorted(zip(candidate_docs, rr), key=lambda x: x[1], reverse=True)

        return [{
            **doc,
            'rerank_score': float(score)
        } for doc, score in reranked]


    def _enhanced_similarity_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        enhanced_queries = self.enhance_query(query)
        query_tokens = self.extract_keywords(query)

        all_candidates: Dict[str, Dict[str, Any]] = {}
        for eq in enhanced_queries[:3]:
            q_emb = self._encode_texts([eq], is_query=True)
            results = self._query_collection(q_emb, n_results=min(k * 2, 15))
            for i in range(len(results['documents'][0])):
                doc_id = results['ids'][0][i]
                meta = results['metadatas'][0][i] or {}
                # ✅ ใช้ full_content ถ้ามี
                candidate_text = meta.get("content") or results['documents'][0][i]

                if doc_id not in all_candidates:
                    all_candidates[doc_id] = {
                        'content': candidate_text,
                        'metadata': meta,
                        'distance': results['distances'][0][i],
                        'doc_id': doc_id
                    }

        candidates = list(all_candidates.values())

        # semantic score
        enhanced_candidates = []
        for doc in candidates:
            semantic_score = max(0.0, 1.0 - float(doc['distance']))
            tfidf_score = self.calculate_tfidf_score(query_tokens, doc['content']) if (query_tokens and self.total_docs > 0) else 0.0

            doc_keywords = set(self.extract_keywords(doc['content']))
            q_keywords = set(query_tokens)
            keyword_overlap = len(doc_keywords & q_keywords)
            keyword_score = (keyword_overlap / len(q_keywords)) if q_keywords else 0.0

            metadata = doc.get('metadata', {}) or {}
            metadata_score = 0.0
            if 'source' in metadata and any(kw in str(metadata['source']).lower() for kw in query_tokens):
                metadata_score += 0.1
            if 'keywords' in metadata and metadata['keywords']:
                meta_keywords = {kw.strip().lower() for kw in str(metadata['keywords']).split(',') if kw.strip()}
                metadata_score += len(meta_keywords & {kw.lower() for kw in query_tokens}) * 0.05

            wc = metadata.get('word_count', len(doc['content'].split()))
            length_penalty = 1.0 if wc < 500 else 0.9 if wc < 1000 else 0.8

            enhanced_candidates.append({
                **doc,
                'semantic_score': semantic_score,
                'tfidf_score': tfidf_score,
                'keyword_score': keyword_score,
                'metadata_score': metadata_score,
                'length_penalty': length_penalty,
            })

        # cross-encoder rerank (เพิ่ม context จาก metadata เหมือนเดิม)
        pairs = []
        for d in enhanced_candidates:
            ctx = ""
            if 'source' in d['metadata']: ctx += f"[{d['metadata']['source']}] "
            if 'section' in d['metadata']: ctx += f"หัวข้อ: {d['metadata']['section']} "
            pairs.append((query, f"{ctx}{d['content']}"))

        rr = self.rerank_model.predict(pairs) if pairs else []
        final_candidates = []
        for d, r in zip(enhanced_candidates, rr):
            final_score = (
                0.35 * float(r) +
                0.25 * d['semantic_score'] +
                0.20 * d['tfidf_score'] +
                0.10 * d['keyword_score'] +
                0.05 * d['metadata_score'] +
                0.05 * d['length_penalty']
            )
            final_candidates.append({**d, 'rerank_score': float(r), 'final_score': float(final_score)})

        return self._select_diverse_results(final_candidates, k, lambda_param=0.7)

    
    # -------------------- Query expansion (restored) --------------------
    def enhance_query(self, query: str) -> List[str]:
        """
        ขยาย query แบบเรียบง่ายเพื่อเพิ่ม recall:
        - ถ้าไม่มีคำถามชัดเจน เติมรูปแบบถาม-ตอบ
        - แทนคำพ้องบางคำ
        - ดึงคีย์เวิร์ดหลัก (Thai) เพื่อสร้าง variant แบบสั้น
        """
        q = (query or "").strip()
        variants: List[str] = [q]

        # 1) เพิ่มรูปแบบถาม-ตอบ ถ้าไม่ได้ขึ้นต้นแนวคำถาม
        question_markers = ['คือ', 'อะไร', 'ทำไม', 'อย่างไร', 'เมื่อไหร่', 'ที่ไหน']
        if not any(m in q for m in question_markers):
            variants.extend([
                f"คำตอบของ {q}",
                f"ข้อมูลเกี่ยวกับ {q}",
                f"รายละเอียดของ {q}",
                f"อธิบาย {q}"
            ])

        # 2) คำพ้องความหมายพื้นฐาน (ไทย)
        synonyms = {
            'วิธี': ['วิธีการ', 'ขั้นตอน', 'กระบวนการ'],
            'ใช้': ['ใช้งาน', 'นำไปใช้', 'ประยุกต์ใช้'],
            'คือ': ['หมายถึง', 'มีความหมายว่า'],
            'นโยบาย': ['ข้อกำหนด', 'ระเบียบ', 'กฎ']
        }
        for word, syns in synonyms.items():
            if word in q:
                for syn in syns:
                    variants.append(q.replace(word, syn))

        # 3) เพิ่มเวอร์ชันคีย์เวิร์ดล้วน (สั้น กระชับ)
        kws = self.extract_keywords(q, top_k=3)
        if len(kws) >= 2:
            variants.append(" ".join(kws))

        # ลบซ้ำ + คืนรายการ
        out = list(dict.fromkeys([v for v in variants if v and v.strip()]))
        return out

    # -------------------- Diversity helper (restored) --------------------
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Jaccard similarity บน token ภาษาไทย (newmm)
        """
        t1 = set(word_tokenize((text1 or "").lower(), engine="newmm"))
        t2 = set(word_tokenize((text2 or "").lower(), engine="newmm"))
        if not t1 or not t2:
            return 0.0
        inter = t1.intersection(t2)
        union = t1.union(t2)
        return len(inter) / len(union) if union else 0.0

    def _select_diverse_results(self, candidates: List[Dict[str, Any]], k: int, lambda_param: float = 0.7) -> List[Dict[str, Any]]:
        """
        เลือกผลแบบหลากหลาย (MMR-like)
        - lambda_param สูง → เน้นความเกี่ยวข้องมากกว่า
        - lambda_param ต่ำ → เน้นความหลากหลายมากกว่า
        """
        if not candidates:
            return []

        # เรียงเบื้องต้นตาม final_score (ถ้าไม่มีให้ใช้ rerank_score)
        def base_score(c):
            return float(c.get('final_score', c.get('rerank_score', 0.0)))

        pool = sorted(candidates, key=base_score, reverse=True)
        selected: List[Dict[str, Any]] = [pool[0]]
        remaining = pool[1:]

        while len(selected) < k and remaining:
            best_idx = -1
            best_mmr = -1e9
            for i, cand in enumerate(remaining):
                relevance = base_score(cand)
                # ความคล้ายสูงสุดกับสิ่งที่เลือกไปแล้ว
                max_sim = 0.0
                for s in selected:
                    sim = self._calculate_jaccard_similarity(cand.get('content', ''), s.get('content', ''))
                    if sim > max_sim:
                        max_sim = sim
                # MMR
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break

        return selected[:k]
