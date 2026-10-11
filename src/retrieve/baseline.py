from typing import List, Optional
from src.index.embeddings import EmbeddingModel
from src.index.vector_store import VectorIndex
from src.answer.generator import AnswerGenerator
from src.models import AnswerResponse, RetrievalResult
from src.config import settings

class PlainRAGPipeline:
    """
    System A: Plain RAG baseline.
    Vector search over chunk embeddings, top-k chunks passed directly to Answer LLM.
    """
    def __init__(
        self,
        vector_index: VectorIndex,
        embedding_model: Optional[EmbeddingModel] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        top_k: int = settings.TOP_K
    ):
        self.vector_index = vector_index
        self.embedding_model = embedding_model or EmbeddingModel()
        self.answer_generator = answer_generator or AnswerGenerator()
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Perform dense vector retrieval for a query."""
        k = top_k or self.top_k
        query_vec = self.embedding_model.encode(query, use_cache=True)
        return self.vector_index.search(query_vec, top_k=k)

    def query(self, query: str, top_k: Optional[int] = None) -> AnswerResponse:
        """End-to-end question answering pipeline with citation attribution."""
        retrieved = self.retrieve(query, top_k=top_k)
        response = self.answer_generator.generate_answer(
            query=query,
            retrieved_results=retrieved,
            system_name="System A: Plain RAG"
        )
        return response
