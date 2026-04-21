# config.py

# ArXiv API
ARXIV_API_URL = "http://export.arxiv.org/api/query"
CATEGORY = "cs.AI"
MAX_RESULTS = 200
SLEEP_TIME = 3

# Chunking
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30

# Paths
BASE_DIR = "arxiv_cs_ai"
METADATA_PATH = "arxiv_cs_ai/papers_metadata.jsonl"
CHUNKS_PATH = "arxiv_cs_ai/chunks.jsonl"
EMBEDDINGS_PATH = "arxiv_cs_ai/chunk_embeddings.npy"
METADATA_PARQUET_PATH = "arxiv_cs_ai/chunks_metadata.parquet"
CHROMA_DB_PATH = "arxiv_cs_ai/chroma_db"

# ChromaDB
CHROMA_PATH = "arxiv_cs_ai/chroma_db"
COLLECTION_NAME = "arxiv_ai_chunks"

# Embedding & LLM
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TOP_K = 5

# Batch processing
BATCH_SIZE = 64

# LLM Generation
MAX_CONTEXT_LENGTH = 4096
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.95

# Evaluation
BENCHMARK_PATH = "arxiv_cs_ai/benchmark.csv"
RESULTS_PATH = "arxiv_cs_ai/evaluation_results.csv"
BENCHMARK_SAMPLES = 20
RANDOM_SEED = 42