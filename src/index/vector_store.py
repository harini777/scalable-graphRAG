import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from src.models import Chunk, RetrievalResult

class VectorIndex:
    """
    Vector index wrapper around FAISS supporting Flat (exact), HNSW, and Quantized variants.
    """
    def __init__(
        self,
        dimension: int,
        index_type: str = "hnsw",  # "flat", "hnsw", "sq8"
        m: int = 32,
        ef_construction: int = 64,
        ef_search: int = 32,
        metric: str = "cosine"  # "cosine" or "l2"
    ):
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = metric
        self.faiss_metric = faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2

        self.index: faiss.Index = self._build_empty_index()
        self.metadata: List[Dict[str, Any]] = []

    def _build_empty_index(self) -> faiss.Index:
        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.dimension) if self.faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dimension, self.m, self.faiss_metric)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            return index
        elif self.index_type == "sq8":
            # Scalar quantization (int8)
            index = faiss.IndexScalarQuantizer(self.dimension, faiss.ScalarQuantizer.QT_8bit, self.faiss_metric)
            return index
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def set_ef_search(self, ef_search: int) -> None:
        """Update runtime efSearch parameter for HNSW."""
        self.ef_search = ef_search
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = ef_search

    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add embeddings and their corresponding metadata."""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if not self.index.is_trained:
            self.index.train(embeddings)

        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """
        Query nearest neighbors for a single vector.
        """
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = self.ef_search

        distances, indices = self.index.search(query_vector, top_k)

        results: List[RetrievalResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append(
                RetrievalResult(
                    chunk_id=meta.get("chunk_id", str(idx)),
                    paper_id=meta.get("paper_id", ""),
                    text=meta.get("text", ""),
                    score=float(dist),
                    metadata=meta
                )
            )
        return results

    def save(self, directory: Path, prefix: str = "vector_index") -> None:
        """Persist index and metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss_path = directory / f"{prefix}.faiss"
        meta_path = directory / f"{prefix}_meta.json"

        faiss.write_index(self.index, str(faiss_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "dimension": self.dimension,
                "index_type": self.index_type,
                "m": self.m,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "metric": self.metric,
                "metadata": self.metadata
            }, f)

    @classmethod
    def load(cls, directory: Path, prefix: str = "vector_index") -> "VectorIndex":
        """Load index and metadata from disk."""
        directory = Path(directory)
        faiss_path = directory / f"{prefix}.faiss"
        meta_path = directory / f"{prefix}_meta.json"

        with open(meta_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        instance = cls(
            dimension=info["dimension"],
            index_type=info["index_type"],
            m=info["m"],
            ef_construction=info["ef_construction"],
            ef_search=info["ef_search"],
            metric=info["metric"]
        )
        instance.index = faiss.read_index(str(faiss_path))
        instance.metadata = info["metadata"]
        return instance

    def total_vectors(self) -> int:
        return self.index.ntotal
