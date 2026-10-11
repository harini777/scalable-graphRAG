import pytest
import numpy as np
from pathlib import Path
from src.models import Paper, Chunk
from src.index.chunker import chunk_paper
from src.index.vector_store import VectorIndex
from src.retrieve.baseline import PlainRAGPipeline
from src.answer.generator import AnswerGenerator

def test_paper_chunking():
    paper = Paper(
        arxiv_id="2106.09685",
        title="LoRA: Low-Rank Adaptation of Large Language Models",
        year=2021,
        abstract="We propose LoRA...",
        categories=["cs.CL"],
        authors=["Edward J. Hu"],
        citations=[]
    )
    chunks = chunk_paper(paper, chunk_size=100)
    assert len(chunks) >= 1
    assert chunks[0].paper_id == "2106.09685"
    assert "LoRA" in chunks[0].text

def test_vector_index_add_and_search():
    dim = 64
    index = VectorIndex(dimension=dim, index_type="hnsw")
    
    # Generate deterministic vectors
    np.random.seed(42)
    vectors = np.random.randn(10, dim).astype(np.float32)
    # Normalize for cosine inner product
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    
    metadata = [{"chunk_id": f"c_{i}", "paper_id": f"p_{i}", "text": f"Sample text {i}"} for i in range(10)]
    index.add(vectors, metadata)
    assert index.total_vectors() == 10
    
    query_vec = vectors[0]
    hits = index.search(query_vec, top_k=3)
    assert len(hits) == 3
    # Top-1 must be the exact vector (score ~ 1.0)
    assert hits[0].chunk_id == "c_0"
    assert pytest.approx(hits[0].score, abs=1e-4) == 1.0

def test_mock_answer_generation():
    generator = AnswerGenerator(provider="mock")
    from src.models import RetrievalResult
    results = [
        RetrievalResult(
            chunk_id="c_1",
            paper_id="2106.09685",
            text="Paper [2106.09685]: LoRA reduces memory footprint.",
            score=0.9,
            metadata={"title": "LoRA Paper"}
        )
    ]
    resp = generator.generate_answer(
        query="What does LoRA do?",
        retrieved_results=results,
        system_name="Test System"
    )
    assert resp.answer != ""
    assert len(resp.citations) >= 1
    assert resp.citations[0].paper_id == "2106.09685"
