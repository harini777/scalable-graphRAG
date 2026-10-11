from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

NodeType = Literal["Paper", "Method", "Dataset", "Task", "Metric", "Author"]
RelationType = Literal[
    "CITES",
    "PROPOSES",
    "USES",
    "IMPROVES_ON",
    "EVALUATED_ON",
    "FOR_TASK",
    "AUTHORED_BY",
]

class Paper(BaseModel):
    arxiv_id: str
    title: str
    year: int
    abstract: str
    categories: List[str] = Field(default_factory=list)
    authors: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)  # list of cited arxiv_ids or titles

class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GraphNode(BaseModel):
    node_id: str
    node_type: NodeType
    name: str
    aliases: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)

class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    relation: RelationType
    source_paper_id: str
    supporting_sentence: str
    confidence: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)

class Citation(BaseModel):
    paper_id: str
    paper_title: str
    supporting_text: str

class RetrievalResult(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AnswerResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    retrieved_results: List[RetrievalResult] = Field(default_factory=list)
    subgraph: Optional[Dict[str, Any]] = None
    system_name: str
    latency_ms: float
    total_tokens: int = 0
