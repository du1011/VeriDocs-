# Veridocs
### ArXiv AI Research RAG Pipeline

Veridocs is an end-to-end Retrieval-Augmented Generation (RAG) system that ingests recent AI research papers from ArXiv, indexes them semantically, and answers natural language queries with cited, grounded responses.

---

## Overview

Veridocs fetches papers from the ArXiv `cs.AI` category, chunks and embeds them, stores them in a vector database, and uses a language model to generate answers backed by retrieved sources. A Streamlit interface makes the system accessible without any code.

---

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| 1–2 | **Data Ingestion & Chunking** — Fetch papers from ArXiv API, semantic chunking with sentence boundaries |
| 3–4 | **Embeddings & Vector Store** — Batch embedding with `all-MiniLM-L6-v2`, ChromaDB persistence |
| 5–6 | **Retrieval & Generation** — Cosine similarity retrieval, LLM answer generation with citations |
| 7–8 | **Evaluation & Verification** — Benchmarking, confidence scoring, failure analysis |

---

## Project Structure

```
FinalProject/
├── config.py               # All configuration settings
├── data_ingestion.py       # Stages 1-2: ArXiv fetch + semantic chunking
├── embeddings.py           # Stages 3-4: Embedding generation + ChromaDB indexing
├── rag_core.py             # Stages 5-6: RAG agent (retrieval + generation)
├── evaluation.py           # Stages 7-8: Benchmarking + failure analysis
├── main.py                 # Orchestration script
├── app.py                  # Streamlit UI
├── arxiv_cs_ai/
│   ├── papers_metadata.jsonl       # Raw paper metadata
│   ├── chunks.jsonl                # Semantic chunks
│   ├── chunk_embeddings.npy        # Embedding vectors
│   ├── chunks_metadata.parquet     # Chunk metadata
│   ├── chroma_db/                  # Persistent vector store
│   ├── benchmark.csv               # Evaluation questions
│   └── evaluation_results.csv      # Evaluation output
└── README.md
```

---

## Setup

### 1. Clone and create virtual environment
```bash
git clone <your-repo-url>
cd FinalProject
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install feedparser pandas numpy sentence-transformers chromadb \
            transformers torch accelerate pyarrow nltk tenacity \
            scikit-learn streamlit
```

---

## Usage

### Run the full pipeline
```bash
python main.py --step all
```

### Run individual steps
```bash
python main.py --step ingest       # Fetch and chunk papers
python main.py --step embeddings   # Generate embeddings and index
python main.py --step rag          # Run RAG demo
python main.py --step eval         # Run evaluation
python main.py --step ui           # Launch Streamlit UI
```

### Launch the UI directly
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CATEGORY` | `cs.AI` | ArXiv category to fetch |
| `MAX_RESULTS` | `200` | Number of papers to fetch |
| `CHUNK_SIZE` | `150` | Token size per chunk |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `LLM_MODEL` | `TinyLlama-1.1B-Chat` | Language model for generation |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `BENCHMARK_SAMPLES` | `20` | Questions used for evaluation |

---

## Models

| Component | Model |
|-----------|-------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM (default) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| LLM (high-end) | `mistralai/Mistral-7B-Instruct-v0.3` *(requires 14GB+ RAM)* |

---

## Evaluation

Veridocs includes a built-in evaluation framework that:
- Auto-generates benchmark questions from ingested papers
- Measures response similarity, confidence, and citation quality
- Classifies answers as `Supported`, `Partially Supported`, or `Not Supported`
- Identifies failure cases and suggests improvements

---

## Requirements

- Python 3.10+
- macOS / Linux
- 8GB+ RAM (16GB recommended for larger models)
- Internet connection for initial paper fetch and model download

---

## License

MIT License


