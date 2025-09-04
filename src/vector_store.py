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
import hashlib

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

        self.max_aug_queries = 2            
        self.per_query_fetch_cap = 10      
        self.rerank_top_m = 12          
        self.use_cross_encoder = True 

    def make_id(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
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
        if not documents:
            return

        self.build_document_frequency(documents)

        texts: List[str] = []
        embed_texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for i, doc in enumerate(documents):
            raw = doc.page_content or ""
            meta = dict(doc.metadata or {})

            norm = self.normalize_thai(raw)
            tokenized = self.tokenize_thai(norm)

            title = (meta.get("title") or meta.get("file_name") or "").strip()
            section = (meta.get("section") or meta.get("heading") or "").strip()
            prefix = ""
            if title:   prefix += f"[TITLE] {title}\n"
            if section: prefix += f"[SECTION] {section}\n"

            embed_txt = (prefix + tokenized).strip()

            texts.append(raw)
            embed_texts.append(embed_txt)

            toks_list = word_tokenize(raw.lower(), engine="newmm")
            keywords = [tok for tok, _ in Counter(
                [t for t in toks_list if len(t) > 1 and t not in self.thai_stopwords]
            ).most_common(10)]

            enriched = meta.copy()
            enriched["keywords"] = ", ".join(keywords) if keywords else ""
            enriched["keywords_set"] = set(keywords)          
            enriched["tokens"] = toks_list                    
            enriched["word_count"] = len(toks_list)
            enriched["char_count"] = len(raw)
            enriched["full_content"] = prefix + raw
            metadatas.append(self._clean_metadata(enriched))


        # --- สร้าง embeddings (passage mode + normalize) ---
        embeddings = self._encode_texts(embed_texts, is_query=False)
        

        ids = [self.make_id(embed_txt) for embed_txt in embed_texts]

        self.collection.add(
            embeddings=embeddings,
            documents=texts,        
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
        # เหลือแค่ 2 เวอร์ชันพอ
        enhanced_queries = enhanced_queries[:self.max_aug_queries]

        # คีย์เวิร์ดของ query (ทำครั้งเดียว)
        query_tokens = self.extract_keywords(query)
        q_keywords = set(query_tokens)

        all_candidates: Dict[str, Dict[str, Any]] = {}
        per_fetch = min(k * 2, self.per_query_fetch_cap)

        for eq in enhanced_queries:
            q_emb = self._encode_texts([eq], is_query=True)
            results = self._query_collection(q_emb, n_results=per_fetch)

            # รวมผู้สมัครแบบ unique
            docs0 = results['documents'][0]
            metas0 = results['metadatas'][0]
            dists0 = results['distances'][0]
            ids0   = results['ids'][0]

            for i in range(len(docs0)):
                doc_id = ids0[i]
                meta = metas0[i] or {}
                # <<< FIX BUG: ใช้ full_content ให้ตรงกับตอน ingest >>>
                candidate_text = meta.get("full_content") or docs0[i]

                if doc_id not in all_candidates:
                    all_candidates[doc_id] = {
                        'content': candidate_text,
                        'metadata': meta,
                        'distance': dists0[i],
                        'doc_id': doc_id
                    }

        candidates = list(all_candidates.values())
        if not candidates:
            return []

        # ===== คำนวณคะแนนแบบไม่ tokenize ซ้ำ =====
        enhanced_candidates = []
        for doc in candidates:
            meta = doc.get('metadata') or {}

            # semantic distance -> score
            semantic_score = max(0.0, 1.0 - float(doc['distance']))

            # ใช้ tokens ที่ precompute แล้ว
            dtoks = meta.get('tokens') or word_tokenize(doc['content'].lower(), engine="newmm")
            dfreq = Counter(dtoks)

            # TF-IDF แบบเร็ว: ไม่ต้องวน extract ใหม่
            tfidf_score = 0.0
            if self.total_docs > 0 and q_keywords:
                N = self.total_docs
                for tok in q_keywords:
                    tf = dfreq.get(tok, 0) / (len(dtoks) or 1)
                    if tf > 0:
                        df = self.doc_freq.get(tok, 1)
                        tfidf_score += math.log((N + 1) / (df + 1)) * tf

            # keyword overlap แบบ set
            meta_kw = meta.get('keywords_set')
            if isinstance(meta_kw, list):
                meta_kw = set(meta_kw)
            if not isinstance(meta_kw, set):
                meta_kw = set(str(meta.get('keywords','')).split(',')) if meta.get('keywords') else set()
                meta_kw = {x.strip().lower() for x in meta_kw if x.strip()}

            overlap = len(q_keywords & meta_kw)
            keyword_score = (overlap / (len(q_keywords) or 1))

            metadata_score = 0.0
            if 'source' in meta and any(kw in str(meta['source']).lower() for kw in q_keywords):
                metadata_score += 0.1
            if meta_kw:
                metadata_score += len(meta_kw & q_keywords) * 0.05

            wc = meta.get('word_count', len(dtoks))
            length_penalty = 1.0 if wc < 500 else 0.9 if wc < 1000 else 0.8

            enhanced_candidates.append({
                **doc,
                'semantic_score': semantic_score,
                'tfidf_score': tfidf_score,
                'keyword_score': keyword_score,
                'metadata_score': metadata_score,
                'length_penalty': length_penalty,
            })

        # ===== Pre-filter ก่อนเข้า cross-encoder =====
        # เอา top ตาม semantic+tfidf ก่อน (ถูกกว่า)
        prelim_sorted = sorted(
            enhanced_candidates,
            key=lambda d: (0.6*d['semantic_score'] + 0.4*d['tfidf_score']),
            reverse=True
        )
        top_for_rerank = prelim_sorted[:min(self.rerank_top_m, len(prelim_sorted))]

        # ===== Cross-encoder (optional) =====
        final_candidates = []
        if self.use_cross_encoder and top_for_rerank:
            pairs = []
            for d in top_for_rerank:
                meta = d.get('metadata') or {}
                ctx = ""
                if 'source' in meta:  ctx += f"[{meta['source']}] "
                if 'section' in meta: ctx += f"หัวข้อ: {meta['section']} "
                pairs.append((query, f"{ctx}{d['content']}"))

            rr = self.rerank_model.predict(pairs)
            for d, r in zip(top_for_rerank, rr):
                final_score = (
                    0.50 * float(r) +
                    0.25 * d['semantic_score'] +
                    0.15 * d['tfidf_score'] +
                    0.05 * d['keyword_score'] +
                    0.05 * d['length_penalty']
                )
                final_candidates.append({**d, 'rerank_score': float(r), 'final_score': float(final_score)})
        else:
            # ไม่มี cross-encoder ก็ถ่วงน้ำหนักจากสกอร์ที่มี
            for d in top_for_rerank:
                final_score = (
                    0.65 * d['semantic_score'] +
                    0.30 * d['tfidf_score'] +
                    0.05 * d['keyword_score']
                )
                final_candidates.append({**d, 'final_score': float(final_score)})

        # ===== Diversify แบบไม่ tokenize ซ้ำ =====
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
    def _calculate_jaccard_similarity(self, text1: str, text2: str, toks1=None, toks2=None) -> float:
        if toks1 is None:
            toks1 = set(word_tokenize((text1 or "").lower(), engine="newmm"))
        else:
            toks1 = set(toks1)
        if toks2 is None:
            toks2 = set(word_tokenize((text2 or "").lower(), engine="newmm"))
        else:
            toks2 = set(toks2)
        if not toks1 or not toks2:
            return 0.0
        inter = toks1 & toks2
        union = toks1 | toks2
        return len(inter) / len(union) if union else 0.0


    def _select_diverse_results(self, candidates, k, lambda_param=0.7):
        if not candidates: return []

        def base_score(c):
            return float(c.get('final_score', c.get('rerank_score', 0.0)))

        pool = sorted(candidates, key=base_score, reverse=True)
        selected = [pool[0]]
        remaining = pool[1:]

        # เตรียม tokens ล่วงหน้า
        def get_toks(d):
            meta = d.get('metadata') or {}
            toks = meta.get('tokens')
            if toks: return toks
            # fallback ครั้งเดียว
            return word_tokenize(d.get('content','').lower(), engine="newmm")

        selected_toks = [get_toks(pool[0])]

        while len(selected) < k and remaining:
            best_idx = -1
            best_mmr = -1e9
            for i, cand in enumerate(remaining):
                relevance = base_score(cand)
                cand_toks = get_toks(cand)
                max_sim = 0.0
                for s, s_toks in zip(selected, selected_toks):
                    sim = self._calculate_jaccard_similarity(
                        cand.get('content',''), s.get('content',''),
                        toks1=cand_toks, toks2=s_toks
                    )
                    if sim > max_sim:
                        max_sim = sim
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            if best_idx >= 0:
                chosen = remaining.pop(best_idx)
                selected.append(chosen)
                selected_toks.append(get_toks(chosen))
            else:
                break

        return selected[:k]

