import os
import hashlib
from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path

from src.config import settings

class EmbeddingModel:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL, device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        self.cache_dir = settings.CACHE_DIR / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, texts: List[str]) -> str:
        h = hashlib.sha256()
        h.update(self.model_name.encode("utf-8"))
        for t in texts:
            h.update(t.encode("utf-8"))
        return h.hexdigest()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Encode text or list of texts into float32 numpy embeddings.
        Supports automatic disk caching for repeatable experiments.
        """
        if isinstance(texts, str):
            single = True
            text_list = [texts]
        else:
            single = False
            text_list = texts

        if not text_list:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        cache_key = self._get_cache_key(text_list)
        cache_file = self.cache_dir / f"{cache_key}.npy"

        if use_cache and cache_file.exists():
            embeddings = np.load(cache_file)
            return embeddings[0] if single else embeddings

        embeddings = self.model.encode(
            text_list,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        ).astype(np.float32)

        if use_cache:
            try:
                np.save(cache_file, embeddings)
            except Exception:
                pass

        return embeddings[0] if single else embeddings

    @property
    def dimension(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()
