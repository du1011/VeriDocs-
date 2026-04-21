"""
Main orchestration script for ArXiv RAG Pipeline
Run this to execute all steps sequentially
"""

import logging
import sys
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def run_pipeline(step: str = "all"):
    """
    Run pipeline steps.
    
    Args:
        step: 'all', 'ingest', 'embeddings', 'rag', 'eval', or 'ui'
    """
    logger.info("=" * 60)
    logger.info("ARXIV RAG PIPELINE")
    logger.info("=" * 60)
    
    if step in ["all", "ingest"]:
        logger.info("\n>>> Running Steps 1-2: Data Ingestion")
        from data_ingestion import run_ingestion
        run_ingestion()
    
    if step in ["all", "embeddings"]:
        logger.info("\n>>> Running Steps 3-4: Embeddings & Vector Store")
        from embeddings import run_embeddingsAndVectorStore
        run_embeddingsAndVectorStore()
    
    if step in ["all", "rag"]:
        logger.info("\n>>> Running Steps 5-6: RAG Demo")
        from rag_core import run_rag_demo
        run_rag_demo()
    
    if step in ["all", "eval"]:
        logger.info("\n>>> Running Steps 7-8: Evaluation")
        from evaluation import run_full_evaluation
        run_full_evaluation()
    
    if step == "ui":
        logger.info("\n>>> Starting Streamlit UI")
        import subprocess
        subprocess.run(["streamlit", "run", "app.py"])
    
    if step == "all":
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 60)
        logger.info("To start the UI, run: python main.py --step ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArXiv RAG Pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "ingest", "embeddings", "rag", "eval", "ui"],
        default="all",
        help="Which step to run (default: all)"
    )
    
    args = parser.parse_args()
    run_pipeline(args.step)