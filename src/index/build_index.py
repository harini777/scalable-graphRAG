import os
import json
import argparse
import logging
from pathlib import Path
from typing import List

from src.config import settings
from src.models import Paper, Chunk
from src.ingest.arxiv import create_seed_papers_dataset, fetch_arxiv_papers
from src.index.chunker import chunk_papers
from src.index.embeddings import EmbeddingModel
from src.index.vector_store import VectorIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_vector_index(
    papers_path: Path,
    output_dir: Path,
    index_type: str = "hnsw",
    sample_size: int = 5000
) -> VectorIndex:
    logger.info(f"Loading papers from {papers_path}...")
    papers: List[Paper] = []
    with open(papers_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(Paper.model_validate_json(line))
                if len(papers) >= sample_size:
                    break

    logger.info(f"Loaded {len(papers)} papers. Chunking...")
    chunks = chunk_papers(papers)
    logger.info(f"Generated {len(chunks)} chunks.")

    logger.info("Initializing embedding model...")
    embedder = EmbeddingModel()

    chunk_texts = [c.text for c in chunks]
    logger.info(f"Computing embeddings for {len(chunk_texts)} chunks...")
    embeddings = embedder.encode(chunk_texts, batch_size=64, show_progress_bar=True)

    metadata = [
        {
            "chunk_id": c.chunk_id,
            "paper_id": c.paper_id,
            "text": c.text,
            **c.metadata
        }
        for c in chunks
    ]

    logger.info(f"Building {index_type.upper()} index (dim={embedder.dimension})...")
    vector_index = VectorIndex(dimension=embedder.dimension, index_type=index_type)
    vector_index.add(embeddings, metadata)

    logger.info(f"Saving index to {output_dir}...")
    vector_index.save(output_dir, prefix="plain_rag")
    logger.info("Index build complete!")
    return vector_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build baseline vector index")
    parser.add_argument("--source", type=str, default="seed", choices=["seed", "arxiv"], help="Source of papers")
    parser.add_argument("--count", type=int, default=5000, help="Number of papers to process")
    parser.add_argument("--index-type", type=str, default="hnsw", choices=["flat", "hnsw", "sq8"])
    args = parser.parse_args()

    data_dir = settings.DATA_DIR
    if args.source == "seed":
        seed_file = data_dir / "papers_seed.jsonl"
        if not seed_file.exists():
            create_seed_papers_dataset(seed_file)
        target_file = seed_file
    else:
        arxiv_file = data_dir / f"papers_{args.count}.jsonl"
        if not arxiv_file.exists():
            fetch_arxiv_papers(max_results=args.count, output_path=arxiv_file)
        target_file = arxiv_file

    out_dir = data_dir / "indices" / "plain_rag"
    build_vector_index(target_file, out_dir, index_type=args.index_type, sample_size=args.count)
