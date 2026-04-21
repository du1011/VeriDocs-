#!/usr/bin/env python
# coding: utf-8

# 

# # Stage 1 : API Intergration (ArXiv)

# In[2]:


"""
Configuration settings for ArXiv RAG System
"""

import os

# Paths
BASE_DIR = "arxiv_cs_ai"
os.makedirs(BASE_DIR, exist_ok=True)

METADATA_PATH = os.path.join(BASE_DIR, "cs_ai_recent.jsonl")
CHUNKS_PATH = os.path.join(BASE_DIR, "cs_ai_recent_chunks.jsonl")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "chunk_embeddings.npy")
METADATA_PARQUET_PATH = os.path.join(BASE_DIR, "chunks_metadata.parquet")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
BENCHMARK_PATH = os.path.join(BASE_DIR, "benchmark_questions.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "evaluation_results.csv")

# API Settings
ARXIV_API_URL = "http://export.arxiv.org/api/query"
CATEGORY = "cs.AI"
MAX_RESULTS = 500
SLEEP_TIME = 3.0

# Chunking Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 100

# Model Settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_CONTEXT_LENGTH = 4096
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9

# Retrieval Settings
TOP_K = 5
BATCH_SIZE = 32

# Evaluation
BENCHMARK_SAMPLES = 20
RANDOM_SEED = 42

def run_ingestion():
    run_pipeline()  # or whatever the main function is called in that file