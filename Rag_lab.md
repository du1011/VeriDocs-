# RAG Pipeline Project — COMP 741

---

## Stage 1 — The RAG Pipeline

A RAG system consists of two flows:

### Indexing Flow (Offline)
- Documents → ArXiv cs.AI papers
- Chunker → splits documents into smaller text units
- Embedding model → converts chunks into vectors
- Vector store → stores embeddings (ChromaDB)

### Querying Flow (Online)
- User query → converted into embedding
- Vector store → retrieves top-k similar chunks
- Prompt builder → constructs context + question
- LLM → generates grounded answer

The vector store acts as the bridge between indexing and querying.

---

### What you implemented
- Defined full pipeline for ArXiv dataset
- Identified each component used in your system

---

### Key insight
RAG is not just an LLM — it is a **retrieval + generation system**.  
The LLM is only as good as the retrieved context.

---

## Stage 2 — Document Ingestion & Chunking

### What you built
- ArXiv API ingestion pipeline
- JSONL dataset storage
- Fixed-size chunking with overlap

### What to observe
- Small chunks → better retrieval precision
- Large chunks → more context but noisy
- Overlap → improves recall but increases storage

---

### Tasks completed
- Retrieved 500 cs.AI papers via API
- Saved to `cs_ai_recent.jsonl`
- Created chunk dataset `cs_ai_recent_chunks.jsonl`
- Inspected chunk quality manually

---

### Key insight
Chunk quality directly affects retrieval quality.  
Bad chunking = bad answers.

---

## Stage 3 — Embeddings & Similarity

### What you built
- Embedding model: `all-MiniLM-L6-v2`
- Vector representation for each chunk
- Cosine similarity search

---

### What to observe
- Same meaning → high similarity even with different words
- Embeddings capture semantics, not keywords

---

### Tasks completed
- Generated embeddings for all chunks
- Saved:
  - `chunk_embeddings.npy`
  - `chunks_metadata.parquet`
- Tested similarity queries

---

### Key insight
Semantic search solves the limitation of keyword matching.

---

## Stage 4 — Naive Vector Store

### What you built
- Manual vector store using cosine similarity
- Retrieval function: `retrieve_chunks_simple`

---

### What to observe
- Retrieval is nearest-neighbor search
- Increasing k:
  - improves recall
  - reduces precision

---

### Tasks completed
- Implemented in-memory retrieval
- Compared results for different k values

---

### Key insight
A vector store is just organized similarity search.

---

## Stage 5 — End-to-End RAG

### What you built
- Prompt builder
- Retrieval + generation pipeline
- LLM integration using Hugging Face

### LLM used
- Primary: `mistralai/Mistral-7B-Instruct-v0.3`
- Fallback: `google/flan-t5-base`

---

### What to observe
- Without context → hallucination risk
- With context → grounded answers
- Out-of-scope → system still retrieves irrelevant chunks

---

### Tasks completed
- Built `rag_answer_mistral()`
- Compared responses with/without retrieval
- Tested multiple query types

---

### Key insight
Prompt design controls whether the LLM stays grounded.

---

## Stage 6 — Real Vector Store (ChromaDB)

### What you built
- Persistent vector database using ChromaDB
- Stored embeddings + metadata

---

### What to observe
- Data persists across sessions
- Retrieval faster than naive approach
- Metadata filtering improves relevance

---

### Tasks completed
- Created collection: `arxiv_ai_chunks`
- Indexed all chunks
- Verified persistence
- Handled duplicate indexing safely

---

### Key insight
Production vector stores add:
- persistence
- scalability
- metadata filtering

---

## Stage 7 — Retrieval Evaluation

### What you built
- Benchmark dataset (20 questions)
- Manual expected answers
- Evaluation pipeline

---

### What to observe
- Retrieval and generation fail differently
- More chunks ≠ better answers

---

### Tasks completed
- Created `benchmark_questions.csv`
- Filled expected answers manually
- Ran evaluation loop
- Saved `evaluation_results.csv`

---

### Metrics used
- Supported / Partially Supported / Not Supported
- Number of citations
- Manual correctness

---

### Key insight
Good retrieval is required before good generation.

---

## Stage 8 — Extension (Verification & Analysis)

### What you built
- Verification layer for answers
- Failure analysis workflow

---

### What to observe
Failures come from:
- Retrieval issues
- LLM hallucination
- Weak chunking

---

### Tasks completed
- Classified answers:
  - Supported
  - Partially Supported
  - Not Supported
- Identified failure cases

---

### Key insight
RAG systems fail in two places:
- retrieval
- generation

Fixing the right one matters.

---

## Final Reflection

This project demonstrates:

- End-to-end RAG system design
- Real dataset ingestion
- Embedding-based retrieval
- LLM integration with Hugging Face
- Evaluation using benchmark dataset
- Extension with verification

---

## Conclusion

The system successfully:
- retrieves relevant scientific content
- generates grounded answers
- evaluates performance using structured metrics

Future improvements include:
- reranking
- hybrid search
- larger models
- UI integration (Streamlit)