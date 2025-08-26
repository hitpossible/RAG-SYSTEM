import os

# เพิ่มการตั้งค่า environment variable ที่ตอนเริ่มต้น
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
import uuid
from pythainlp import word_tokenize
import re
from pythainlp.util import normalize
import numpy as np
from collections import Counter
import math
# from openai import OpenAI

class VectorStore:
    def __init__(self, db_path: str, embedding_model: str):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.collection_name = "documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        
        # เก็บ document frequency สำหรับ TF-IDF
        self.doc_freq = {}
        self.total_docs = 0
        
        # Thai stop words
        self.thai_stopwords = {
            'และ', 'หรือ', 'แต่', 'ที่', 'ใน', 'จาก', 'ของ', 'เป็น', 'มี', 'จะ', 'ได้', 'แล้ว', 
            'กับ', 'ไป', 'มา', 'ให้', 'ถึง', 'คือ', 'อะไร', 'ไหม', 'มั้ย', 'นะ', 'ครับ', 'ค่ะ',
            'เอ่อ', 'อื่ม', 'อา', 'เออ', 'โอ้', 'อ๋อ', 'เฮ้ย', 'นั่น', 'นี่', 'โน่น', 'เหล่า',
            'ผู้', 'คน', 'บุคคล', 'ตัว', 'อัน', 'สิ่ง', 'การ', 'ความ', 'เรื่อง'
        }

    def normalize_thai(self, text: str) -> str:
        text = normalize(text)              # Normalize วรรณยุกต์และสระ
        text = re.sub(r"\s+", " ", text)     # ลบช่องว่างเกิน
        return text.strip()

    def tokenize_thai(self, text: str) -> str:
        tokens = word_tokenize(text, engine="newmm")
        return " ".join(tokens)
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """ดึงคำสำคัญจากข้อความ"""
        # Tokenize และกรอง stop words
        tokens = word_tokenize(text, engine="newmm")
        filtered_tokens = [
            token.lower() for token in tokens 
            if len(token) > 1 and token.lower() not in self.thai_stopwords
            and not re.match(r'^[0-9\s\.\,\!\?\-\(\)]+$', token)
        ]
        
        # นับความถี่
        token_freq = Counter(filtered_tokens)
        return [token for token, freq in token_freq.most_common(top_k)]
    
    def calculate_tfidf_score(self, query_tokens: List[str], doc_content: str) -> float:
        """คำนวณ TF-IDF score"""
        if self.total_docs == 0 or not query_tokens:
            return 0.0
            
        doc_tokens = word_tokenize(doc_content.lower(), engine="newmm")
        if not doc_tokens:
            return 0.0
            
        doc_token_freq = Counter(doc_tokens)
        
        tfidf_score = 0
        for token in query_tokens:
            # TF (Term Frequency)
            tf = doc_token_freq.get(token, 0) / len(doc_tokens)
            
            if tf == 0:  # ถ้าไม่มี token นี้ในเอกสาร ข้ามไป
                continue
            
            # IDF (Inverse Document Frequency)
            df = self.doc_freq.get(token, 1)
            
            # ป้องกัน math domain error
            if df > 0 and self.total_docs > 0 and self.total_docs >= df:
                idf = math.log((self.total_docs + 1) / (df + 1))  # เพิ่ม 1 เพื่อป้องกัน log(0)
            else:
                idf = 0.0
            
            tfidf_score += tf * idf
        
        return tfidf_score
    
    def build_document_frequency(self, documents: List[Document]):
        """สร้าง document frequency สำหรับ TF-IDF"""
        self.doc_freq.clear()
        self.total_docs = len(documents)
        
        if self.total_docs == 0:
            return
        
        for doc in documents:
            if not doc.page_content:  # ตรวจสอบเอกสารว่าง
                continue
                
            tokens = set(word_tokenize(doc.page_content.lower(), engine="newmm"))
            for token in tokens:
                if len(token) > 1 and token not in self.thai_stopwords:
                    self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        
        # ตรวจสอบผลลัพธ์
        print(f"Built document frequency for {self.total_docs} documents, {len(self.doc_freq)} unique tokens")
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ทำความสะอาด metadata ให้เป็น type ที่ ChromaDB รองรับ"""
        cleaned = {}
        
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = None
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list):
                # แปลง list เป็น string
                cleaned[key] = ', '.join(str(item) for item in value)
            elif isinstance(value, dict):
                # แปลง dict เป็น string
                cleaned[key] = str(value)
            else:
                # แปลง type อื่นๆ เป็น string
                cleaned[key] = str(value)
        
        return cleaned
    
    def _parse_keywords_from_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """ดึง keywords กลับมาจาก metadata"""
        if 'keywords' in metadata and metadata['keywords']:
            return [kw.strip() for kw in metadata['keywords'].split(',') if kw.strip()]
        return []
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store with enhanced processing"""
        # สร้าง document frequency
        self.build_document_frequency(documents)
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings with enhanced preprocessing
        processed_texts = []
        enhanced_metadatas = []
        
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            # ปรับปรุงข้อความ
            norm = self.normalize_thai(text)
            tokenized = self.tokenize_thai(norm)
            processed_texts.append(tokenized)
            
            # เพิ่มข้อมูล metadata
            enhanced_metadata = metadata.copy()
            
            # แปลง keywords เป็น string เพื่อให้ ChromaDB รองรับ
            keywords = self.extract_keywords(text)
            enhanced_metadata['keywords'] = ', '.join(keywords) if keywords else ""
            enhanced_metadata['keywords_list'] = str(keywords)  # เก็บเป็น string representation
            enhanced_metadata['word_count'] = len(word_tokenize(text, engine="newmm"))
            enhanced_metadata['char_count'] = len(text)
            
            # ตรวจสอบและแปลงค่าใน metadata ให้เป็น type ที่ ChromaDB รองรับ
            cleaned_metadata = self._clean_metadata(enhanced_metadata)
            enhanced_metadatas.append(cleaned_metadata)

        task = "retrieval.query"
        embeddings = self.embedding_model.encode(processed_texts, task=task, prompt_name=task).tolist()
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in range(len(processed_texts))]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=enhanced_metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} documents to vector store")
    
    def enhance_query(self, query: str) -> List[str]:
        """ขยาย query ให้ครอบคลุมมากขึ้น"""
        enhanced_queries = [query]
        
        # สร้าง query variants
        variants = []
        
        # เพิ่มรูปแบบคำถาม
        if not any(q_word in query for q_word in ['คือ', 'อะไร', 'ทำไม', 'อย่างไร', 'เมื่อไหร่', 'ที่ไหน']):
            variants.extend([
                f"คำตอบของ {query}",
                f"ข้อมูลเกี่ยวกับ {query}",
                f"รายละเอียดของ {query}",
                f"การอธิบาย {query}"
            ])
        
        # แทนที่คำพ้อง
        synonyms = {
            'วิธี': ['วิธีการ', 'ขั้นตอน', 'กระบวนการ'],
            'ทำ': ['จัดทำ', 'สร้าง', 'ดำเนินการ'],
            'ใช้': ['ใช้งาน', 'นำไปใช้', 'ประยุกต์ใช้'],
            'คือ': ['หมายถึง', 'มีความหมายว่า', 'คือการ']
        }
        
        for word, syns in synonyms.items():
            if word in query:
                for syn in syns:
                    variants.append(query.replace(word, syn))
        
        # ดึงคำสำคัญ
        keywords = self.extract_keywords(query, top_k=3)
        if len(keywords) >= 2:
            variants.append(' '.join(keywords))
        
        enhanced_queries.extend(variants)
        return list(set(enhanced_queries))  # ลบซ้ำ
    
    def similarity_search(self, query: str, k: int = 5, use_enhanced: bool = True) -> List[Dict[str, Any]]:
        """Enhanced similarity search with multiple techniques"""
        
        if use_enhanced:
            return self._enhanced_similarity_search(query, k)
        else:
            return self._basic_similarity_search(query, k)
    
    def _basic_similarity_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Original similarity search method"""
        task = "retrieval.query"
        query_embedding = self.embedding_model.encode([query], task=task, prompt_name=task).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        
        candidate_docs = []
        for i in range(len(results['documents'][0])):
            candidate_docs.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })

        pairs = [(query, doc['content']) for doc in candidate_docs]
        rerank_scores = self.rerank_model.predict(pairs)
        
        reranked_results = sorted(
            zip(candidate_docs, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        final_results = []
        for doc, score in reranked_results:
            final_results.append({
                'content': doc['content'],
                'metadata': doc['metadata'],
                'distance': doc['distance'],
                'rerank_score': float(score)
            })
        
        return final_results
    
    def _enhanced_similarity_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Enhanced search with multiple techniques"""
        
        # 🔥 1. Query Enhancement
        enhanced_queries = self.enhance_query(query)
        query_tokens = self.extract_keywords(query)
        
        all_candidates = {}  # ใช้ dict เพื่อหลีกเลี่ยงซ้ำ
        
        # ค้นหาด้วย enhanced queries
        for enhanced_query in enhanced_queries[:3]:  # จำกัดไว้ 3 queries
            task = "retrieval.query"
            query_embedding = self.embedding_model.encode([enhanced_query], task=task, prompt_name=task).tolist()
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(k * 2, 15)  # เพิ่มตัวเลือก
            )
            
            for i in range(len(results['documents'][0])):
                doc_id = results['ids'][0][i]
                if doc_id not in all_candidates:
                    all_candidates[doc_id] = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'doc_id': doc_id
                    }
        
        candidates = list(all_candidates.values())
        
        # 🔥 2. Multi-Score Calculation
        enhanced_candidates = []
        for doc in candidates:
            # Semantic similarity score (จาก distance)
            semantic_score = max(0, 1 - doc['distance'])
            
            # TF-IDF score (ตรวจสอบก่อนคำนวณ)
            tfidf_score = 0.0
            if query_tokens and self.total_docs > 0:
                try:
                    tfidf_score = self.calculate_tfidf_score(query_tokens, doc['content'])
                except Exception as e:
                    print(f"Warning: TF-IDF calculation failed: {e}")
                    tfidf_score = 0.0
            
            # Keyword matching score
            doc_keywords = set(self.extract_keywords(doc['content']))
            query_keywords = set(query_tokens)
            keyword_overlap = len(doc_keywords.intersection(query_keywords))
            keyword_score = keyword_overlap / len(query_keywords) if query_keywords else 0
            
            # Metadata boost
            metadata_score = 0
            metadata = doc.get('metadata', {})
            if 'source' in metadata and any(keyword in metadata['source'].lower() for keyword in query_tokens):
                metadata_score += 0.1
            
            # ดึง keywords จาก metadata string
            if 'keywords' in metadata and metadata['keywords']:
                meta_keywords = set([kw.lower().strip() for kw in metadata['keywords'].split(',') if kw.strip()])
                meta_overlap = len(meta_keywords.intersection(set([kw.lower() for kw in query_tokens])))
                metadata_score += meta_overlap * 0.05
            
            # Length penalty (เอกสารยาวเกินไปอาจไม่เฉพาะเจาะจง)
            word_count = metadata.get('word_count', len(doc['content'].split()))
            length_penalty = 1.0 if word_count < 500 else 0.9 if word_count < 1000 else 0.8
            
            enhanced_candidates.append({
                'content': doc['content'],
                'metadata': doc['metadata'],
                'distance': doc['distance'],
                'semantic_score': semantic_score,
                'tfidf_score': tfidf_score,
                'keyword_score': keyword_score,
                'metadata_score': metadata_score,
                'length_penalty': length_penalty,
                'doc_id': doc['doc_id']
            })
        
        # 🔥 3. Cross-Encoder Re-ranking with context
        rerank_pairs = []
        for doc in enhanced_candidates:
            # เพิ่ม context จาก metadata
            context = ""
            if 'source' in doc['metadata']:
                context += f"[{doc['metadata']['source']}] "
            if 'section' in doc['metadata']:
                context += f"หัวข้อ: {doc['metadata']['section']} "
            
            enhanced_content = f"{context}{doc['content']}"
            rerank_pairs.append((query, enhanced_content))
        
        rerank_scores = self.rerank_model.predict(rerank_pairs) if rerank_pairs else []
        
        # 🔥 4. Score Fusion
        final_candidates = []
        for i, (doc, rerank_score) in enumerate(zip(enhanced_candidates, rerank_scores)):
            # Weighted fusion
            final_score = (
                0.35 * float(rerank_score) +           # Cross-encoder score
                0.25 * doc['semantic_score'] +         # Semantic similarity
                0.20 * doc['tfidf_score'] +           # TF-IDF
                0.10 * doc['keyword_score'] +         # Keyword matching
                0.05 * doc['metadata_score'] +        # Metadata boost
                0.05 * doc['length_penalty']          # Length penalty
            )
            
            final_candidates.append({
                'content': doc['content'],
                'metadata': doc['metadata'],
                'distance': doc['distance'],
                'rerank_score': float(rerank_score),
                'semantic_score': doc['semantic_score'],
                'tfidf_score': doc['tfidf_score'],
                'keyword_score': doc['keyword_score'],
                'metadata_score': doc['metadata_score'],
                'final_score': final_score,
                'doc_id': doc['doc_id']
            })
        
        # 🔥 5. Diversity-aware ranking (MMR-like)
        diverse_results = self._select_diverse_results(final_candidates, k, lambda_param=0.7)
        
        return diverse_results
    
    def _select_diverse_results(self, candidates: List[Dict], k: int, lambda_param: float = 0.7) -> List[Dict]:
        """Select diverse results using MMR-like algorithm"""
        if not candidates:
            return []
        
        # เรียงตาม final_score
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        selected = [candidates[0]]  # เลือกผลลัพธ์แรก (คะแนนสูงสุด)
        remaining = candidates[1:]
        
        while len(selected) < k and remaining:
            best_score = -float('inf')
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                relevance = candidate['final_score']
                
                # คำนวณ diversity (ความแตกต่างจากที่เลือกแล้ว)
                max_similarity = 0
                for selected_doc in selected:
                    similarity = self._calculate_jaccard_similarity(
                        candidate['content'], 
                        selected_doc['content']
                    )
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break
        
        return selected[:k]
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """คำนวณ Jaccard similarity"""
        tokens1 = set(word_tokenize(text1.lower(), engine="newmm"))
        tokens2 = set(word_tokenize(text2.lower(), engine="newmm"))
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0
    
    def search_with_filters(self, query: str, k: int = 5, source_filter: Optional[str] = None, 
                           min_score: float = 0.0) -> List[Dict[str, Any]]:
        """ค้นหาพร้อมกรองผลลัพธ์"""
        results = self.similarity_search(query, k=k*2, use_enhanced=True)  # ดึงมากกว่าเพื่อกรอง
        
        filtered_results = []
        for result in results:
            # กรองตาม source
            if source_filter and source_filter.lower() not in result['metadata'].get('source', '').lower():
                continue
            
            # กรองตาม minimum score
            if result.get('final_score', result.get('rerank_score', 0)) < min_score:
                continue
            
            filtered_results.append(result)
            
            if len(filtered_results) >= k:
                break
        
        return filtered_results
    
    def get_collection_info(self):
        """Get information about the collection"""
        return {
            "count": self.collection.count(),
            "name": self.collection_name,
            "doc_freq_size": len(self.doc_freq),
            "total_docs": self.total_docs
        }