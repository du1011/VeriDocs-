# embeddings.py

from typing import List
import json
import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

from config import (
    CHUNKS_PATH, EMBEDDINGS_PATH, METADATA_PARQUET_PATH,
    CHROMA_DB_PATH, EMBEDDING_MODEL, BATCH_SIZE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manage embeddings generation and storage."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def generate_embeddings(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Batch processing for memory efficiency."""
        logger.info(f"Generating embeddings for {len(texts)} texts...")

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            all_embeddings.append(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        embeddings = np.vstack(all_embeddings)
        logger.info(f"Embeddings shape: {embeddings.shape}")

        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        """Save embeddings and metadata."""
        np.save(EMBEDDINGS_PATH, embeddings)
        metadata.to_parquet(METADATA_PARQUET_PATH, index=False)
        logger.info(f"Saved embeddings to {EMBEDDINGS_PATH}")
        logger.info(f"Saved metadata to {METADATA_PARQUET_PATH}")

    def load_embeddings(self) -> tuple:
        """Load embeddings and metadata."""
        embeddings = np.load(EMBEDDINGS_PATH)
        metadata = pd.read_parquet(METADATA_PARQUET_PATH)
        return embeddings, metadata


class VectorStore:
    """ChromaDB vector store with persistence."""

    def __init__(self, persist_directory: str = CHROMA_DB_PATH):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
        logger.info(f"Initialized ChromaDB at {persist_directory}")

    def create_or_load_collection(self, name: str = "arxiv_ai_chunks"):
        """Get or create collection."""
        try:
            self.collection = self.client.get_collection(name=name)
            logger.info(f"Loaded existing collection: {name}")
        except:
            self.collection = self.client.create_collection(name=name)
            logger.info(f"Created new collection: {name}")

        return self.collection

    def index_documents(self, chunks_df: pd.DataFrame, embeddings: np.ndarray,
                        batch_size: int = 500):
        """Index documents in batches with proper metadata handling."""
        if self.collection is None:
            raise ValueError("Collection not initialized")

        if self.collection.count() > 0:
            logger.info("Collection already indexed. Skipping...")
            return

        logger.info(f"Indexing {len(chunks_df)} documents...")

        # Prepare metadata - convert lists to strings for ChromaDB
        metadatas = []
        for _, row in chunks_df.iterrows():
            meta = {
                "doc_id": str(row["doc_id"]),
                "chunk_id": str(row["chunk_id"]),
                "title": str(row["title"]),
                "source": str(row["source"]),
                "published": str(row["published"]),
                "categories": ", ".join(row["categories"]) if isinstance(row["categories"], list) else str(row["categories"]),
                "token_count": int(row["token_count"]),
                "chunk_index": int(row["chunk_index"])
            }
            metadatas.append(meta)

        # Batch insert
        ids = chunks_df["chunk_id"].astype(str).tolist()
        documents = chunks_df["text"].astype(str).tolist()

        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                metadatas=metadatas[i:end_idx]
            )
            logger.info(f"Indexed batch {i//batch_size + 1}")

        logger.info(f"Indexing complete. Total documents: {self.collection.count()}")

    def query(self, query_embedding: np.ndarray, top_k: int = 5,
              filter_dict: dict = None) -> dict:
        """Query with filtering support and similarity scores."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filter_dict,
                include=["documents", "metadatas", "distances", "ids"]
            )
            return results
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}


def run_embeddingsAndVectorStore():
    """Execute Steps 3-4."""
    logger.info("=" * 60)
    logger.info("STEP 3: Generating Embeddings")
    logger.info("=" * 60)

    # Load chunks
    records = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    chunks_df = pd.DataFrame(records)

    # Generate embeddings with batch processing
    embed_manager = EmbeddingManager()
    embeddings = embed_manager.generate_embeddings(chunks_df["text"].tolist())
    embed_manager.save_embeddings(embeddings, chunks_df)

    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Building Vector Store")
    logger.info("=" * 60)

    # Build vector store
    vector_store = VectorStore()
    vector_store.create_or_load_collection()
    vector_store.index_documents(chunks_df, embeddings)

    logger.info("\n✅ Embeddings and Vector Store ready!")


if __name__ == "__main__":
    run_embeddingsAndVectorStore()