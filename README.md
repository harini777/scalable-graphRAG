# Scalable GraphRAG for Research Papers: Multi-Hop QA with Quantized HNSW Retrieval

## 1. Problem Statement & Headline Result
Scientific question answering often requires synthesizing facts dispersed across interconnected research papers (e.g., *"Which adaptation methods that improve upon LoRA were evaluated on long-context benchmarks?"*). Standard Plain RAG relies solely on semantic chunk similarity, which regularly fails to retrieve non-contiguous multi-hop evidentiary chains. This project builds a production-grade GraphRAG system over AI/ML research papers combining an entity/relation knowledge graph with a vector index. 

```
                                  [ User Query ]
                                         │
                                 ┌───────┴───────┐
                                 ▼               ▼
                        [ Vector Search ]  [ Query Router ]
                       (HNSW Quantized)    (Simple vs Multi-hop)
                                 │               │
                                 │ Entry Points  │
                                 ▼               ▼
                         [ Graph Expansion Engine ]
                         (NetworkX / Cypher Traversal)
                                 │
                                 ▼
                     [ Cross-Encoder Reranker ]
                                 │
                                 ▼
                   [ Cited Answer Generator (LLM) ]
                                 │
                         ┌───────┴────────┐
                         ▼                ▼
                  [ Cited Answer ]  [ Subgraph View ]
```

---

## 2. Benchmark & Baseline Numbers (Phase 1 Checkpoint)

Evaluated under our deterministic frozen evaluation harness (`data/benchmarks/domain_multihop.jsonl`):

| System | Retrieval Mechanism | Recall@5 | All-Docs Found | Median Latency | p95 Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **System A: Plain RAG** | Dense BGE-small + HNSW Flat | 1.0000 | 1.0000 | **16.1 ms** | **224.9 ms** |
| **System B: Reranked RAG** | Dense + Cross-Encoder | *Pending Phase 3* | *Pending* | *Pending* | *Pending* |
| **System C: GraphRAG** | Vector Entry + Graph Multi-Hop + Reranker | *Pending Phase 3* | *Pending* | *Pending* | *Pending* |
| **System D: GraphRAG + Router** | System C with Query Intent Router | *Pending Phase 3* | *Pending* | *Pending* | *Pending* |

*All results are saved in timestamped files under `eval_runs/`.*

---

## 3. Repository Structure

```
graphrag-papers/
├── README.md              # Project overview, results, quickstart
├── requirements.txt       # Pinned dependency versions
├── .env.example           # Configuration template
├── data/
│   ├── raw/               # Downloaded arXiv XML / metadata dumps
│   ├── processed/         # Cleaned JSONL corpus files
│   ├── cache/             # Cached embeddings and LLM responses
│   └── benchmarks/        # Frozen benchmark datasets (domain_multihop.jsonl)
├── eval_runs/             # Reproducible evaluation runs logged to JSON
├── src/
│   ├── config.py          # Centralized typed configuration
│   ├── models.py          # Schemas (Paper, Chunk, GraphNode, GraphEdge, Citation)
│   ├── ingest/            # arXiv fetcher and cleaner with rate limiting
│   ├── index/             # BGE embeddings, FAISS HNSW & quantization wrappers
│   ├── retrieve/          # Retrieval engines (System A, B, C, D)
│   ├── answer/            # Multi-provider LLM response generation with citations
│   ├── graph/             # Graph construction & multi-hop traversal (Phase 2)
│   ├── extract/           # Structured LLM entity & relation extraction (Phase 2)
│   ├── resolve/           # Entity resolution & alias clustering (Phase 2)
│   └── eval/              # Evaluation metrics, frozen benchmarks, and runner
└── tests/                 # Pytest test suite
```

---

## 4. Reproducing Baseline in Under 2 Minutes

### Step 1: Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Step 2: Build Plain RAG Index
Build the HNSW vector index using the seed dataset:
```bash
python3 -m src.index.build_index --source seed --index-type hnsw
```

### Step 3: Run Baseline Evaluation
```bash
python3 -m src.eval.runner --top_k 5
```

### Step 4: Run Tests
```bash
pytest tests/
```

---

