#!/usr/bin/env python
# coding: utf-8

# # Stage 4 - Rag Core: Retrieval and LLM Integration with Quatization

# In[ ]:


"""
Steps 5-6: Retrieval & LLM Integration (Mac Compatible)
"""

import logging
import time
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from config import (
    LLM_MODEL, MAX_CONTEXT_LENGTH, MAX_NEW_TOKENS,
    TEMPERATURE, TOP_P, CHROMA_DB_PATH, EMBEDDING_MODEL, TOP_K
)
from embeddings import VectorStore, EmbeddingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGAgent:
    """Complete RAG Agent with Retrieval and Generation."""

    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.tokenizer = None
        self.llm_model = None
        self.initialized = False

    def initialize(self):
        """Load all components with proper error handling."""
        logger.info("Initializing RAG Agent...")

        # Load embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("✅ Embedding model loaded")

        # Load vector store
        self.vector_store = VectorStore(CHROMA_DB_PATH)
        self.vector_store.create_or_load_collection()
        logger.info("✅ Vector store loaded")

        # Load LLM
        self._load_llm()
        logger.info("✅ LLM loaded")

        self.initialized = True
        logger.info("🚀 RAG Agent initialization complete!")

    def _load_llm(self):
        """Load LLM without quantization for Mac compatibility."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                torch_dtype=torch.float32,
                device_map="cpu"
            )

        except Exception as e:
            logger.error(f"Error loading LLM: {e}")
            raise

    def retrieve(self, query: str, top_k: int = TOP_K,
                 filter_dict: Dict = None) -> Tuple[List, List, List, List]:
        """Retrieve documents with similarity scores."""
        if not self.initialized:
            raise ValueError("Agent not initialized. Call initialize() first.")

        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Query vector store
        results = self.vector_store.query(query_embedding, top_k, filter_dict)

        docs = results["documents"][0] or []
        metas = results["metadatas"][0] or []
        ids = results["ids"][0] or []
        distances = results.get("distances", [[]])[0] or []

        # Convert to similarity scores (Chroma returns distances)
        similarities = [1 - d for d in distances] if distances else [0] * len(docs)

        return docs, metas, ids, similarities

    def build_enhanced_prompt(self, query: str, docs: List[str],
                              metas: List[Dict]) -> str:
        """Better prompt with citations and clear instructions."""
        context_parts = []

        for i, (doc, meta) in enumerate(zip(docs, metas)):
            source_info = f"[{i+1}] {meta.get('title', 'Unknown')}"
            truncated_doc = doc[:600] if len(doc) > 600 else doc
            context_parts.append(f"{source_info}\n{truncated_doc}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are a research assistant specializing in AI. Based on the provided papers, answer the question accurately. If the papers don't contain enough information, clearly state what is missing.

Papers:
{context}

Question: {query}

Provide a concise, factual answer with citations [number]. If citing multiple papers, use format like [1][2]."""

        return prompt

    def generate(self, query: str, docs: List[str], metas: List[Dict]) -> Dict:
        """Generate answer with timing and quality metrics."""
        start_time = time.time()

        try:
            prompt = self.build_enhanced_prompt(query, docs, metas)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_CONTEXT_LENGTH
            )
            inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    top_p=TOP_P,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

            generation_time = time.time() - start_time

            return {
                "answer": response,
                "generation_time": generation_time,
                "prompt_tokens": inputs['input_ids'].shape[1],
                "generated_tokens": len(generated_tokens)
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "generation_time": time.time() - start_time,
                "prompt_tokens": 0,
                "generated_tokens": 0
            }

    def query(self, question: str, top_k: int = TOP_K) -> Dict:
        """Full RAG pipeline: Retrieve + Generate."""
        total_start = time.time()

        docs, metas, ids, similarities = self.retrieve(question, top_k)

        if not docs:
            return {
                "answer": "No relevant documents found for this query.",
                "sources": [],
                "retrieval_time": time.time() - total_start,
                "generation_time": 0,
                "total_time": time.time() - total_start,
                "status": "No Documents"
            }

        gen_result = self.generate(question, docs, metas)
        confidence = self._calculate_confidence(gen_result['answer'], docs, similarities)

        sources = []
        for meta, sim in zip(metas, similarities):
            source = meta.copy()
            source['similarity'] = round(sim, 4)
            sources.append(source)

        total_time = time.time() - total_start

        return {
            "answer": gen_result['answer'],
            "sources": sources,
            "retrieval_time": total_time - gen_result['generation_time'],
            "generation_time": gen_result['generation_time'],
            "total_time": total_time,
            "prompt_tokens": gen_result['prompt_tokens'],
            "generated_tokens": gen_result['generated_tokens'],
            "confidence": confidence,
            "status": self._classify_answer(gen_result['answer'], docs)
        }

    def _calculate_confidence(self, answer: str, docs: List[str],
                              similarities: List[float]) -> float:
        """Calculate confidence score."""
        if not docs:
            return 0.0

        avg_similarity = np.mean(similarities) if similarities else 0.0
        word_count = len(answer.split())
        length_factor = min(word_count / 50, 1.0)
        citation_count = answer.count('[')
        citation_factor = min(citation_count / 2, 1.0)
        confidence = (avg_similarity * 0.5 + length_factor * 0.3 + citation_factor * 0.2)

        return round(min(confidence, 1.0), 3)

    def _classify_answer(self, answer: str, docs: List[str]) -> str:
        """Classify answer quality."""
        if not docs:
            return "Not Supported"
        if not answer or len(answer) < 20:
            return "Not Supported"
        if "not enough information" in answer.lower() or "don't have" in answer.lower():
            return "Partially Supported"
        if any(str(i) in answer for i in range(1, 10)):
            return "Supported"
        return "Partially Supported"


def run_rag_demo():
    """Demo the RAG system."""
    agent = RAGAgent()
    agent.initialize()

    test_queries = [
        "What are recent trends in transformer architectures?",
        "What is the main focus of the paper about autonomous agents?",
        "How do recent papers address privacy in LLMs?"
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        result = agent.query(query)
        print(f"Answer: {result.get['answer'][:300]}...")
        print(f"Status: {result.get['status']}, Confidence: {result['confidence']}")
        print(f"Time: {result.get['total_time']:.2f}s")


if __name__ == "__main__":
    run_rag_demo()

