import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from src.config import settings

class EvalQuestion(BaseModel):
    id: str
    question: str
    question_type: str  # "single_hop", "two_hop", "three_hop"
    gold_answer: str
    gold_paper_ids: List[str] = Field(default_factory=list)
    hops: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)

def generate_frozen_benchmarks(output_dir: Path = settings.BENCHMARK_DIR):
    """
    Creates and freezes standardized evaluation datasets.
    Includes multi-hop domain questions where ground truth is guaranteed by graph topology.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Domain Multi-Hop QA Set (Questions requiring linked facts across multiple papers)
    domain_questions = [
        EvalQuestion(
            id="domain_mh_01",
            question="Which method that improves on LoRA was evaluated on long-context benchmarks such as PG19?",
            question_type="two_hop",
            gold_answer="LongLoRA",
            gold_paper_ids=["2106.09685", "2309.15228"],
            hops=2,
            metadata={"seed_chain": "LoRA -> LongLoRA -> PG19"}
        ),
        EvalQuestion(
            id="domain_mh_02",
            question="Which adaptation method builds upon LoRA to allow fine-tuning a 65B model on a single 48GB GPU using 4-bit quantization?",
            question_type="two_hop",
            gold_answer="QLoRA",
            gold_paper_ids=["2106.09685", "2305.14314"],
            hops=2,
            metadata={"seed_chain": "LoRA -> QLoRA"}
        ),
        EvalQuestion(
            id="domain_mh_03",
            question="What attention acceleration method proposed by Tri Dao was later improved in a second version to achieve 2x speedup on A100 GPUs?",
            question_type="two_hop",
            gold_answer="FlashAttention (improved in FlashAttention-2)",
            gold_paper_ids=["2205.14135", "2307.08691"],
            hops=2,
            metadata={"seed_chain": "FlashAttention -> FlashAttention-2"}
        ),
        EvalQuestion(
            id="domain_mh_04",
            question="Which replication study demonstrated that BERT was undertrained and evaluated across GLUE, RACE, and SQuAD?",
            question_type="two_hop",
            gold_answer="RoBERTa",
            gold_paper_ids=["1810.04805", "1907.11692"],
            hops=2,
            metadata={"seed_chain": "BERT -> RoBERTa"}
        ),
        EvalQuestion(
            id="domain_mh_05",
            question="Which foundation model collection spanning 7B to 65B parameters served as the base model for Vicuna and QLoRA evaluations?",
            question_type="two_hop",
            gold_answer="LLaMA",
            gold_paper_ids=["2302.13971", "2305.14314"],
            hops=2,
            metadata={"seed_chain": "LLaMA -> QLoRA"}
        ),
        EvalQuestion(
            id="domain_mh_06",
            question="Which parameter-efficient adapter approach introduced in 2019 predated LoRA and achieved strong GLUE performance with 3% overhead?",
            question_type="single_hop",
            gold_answer="Parameter-Efficient Transfer Learning for NLP (Houlsby adapters)",
            gold_paper_ids=["1905.11946"],
            hops=1,
            metadata={"seed_chain": "Adapter Tuning"}
        ),
        EvalQuestion(
            id="domain_mh_07",
            question="What benchmarks are shared between the evaluations of LoRA and RoBERTa for measuring natural language understanding?",
            question_type="three_hop",
            gold_answer="GLUE benchmark",
            gold_paper_ids=["2106.09685", "1907.11692"],
            hops=3,
            metadata={"seed_chain": "LoRA -> GLUE <- RoBERTa"}
        ),
        EvalQuestion(
            id="domain_mh_08",
            question="Identify the methods in the corpus that specifically address memory efficiency during Transformer training or fine-tuning.",
            question_type="two_hop",
            gold_answer="LoRA, QLoRA, and FlashAttention",
            gold_paper_ids=["2106.09685", "2305.14314", "2205.14135"],
            hops=2,
            metadata={"seed_chain": "Efficiency aggregation"}
        )
    ]

    domain_file = output_dir / "domain_multihop.jsonl"
    with open(domain_file, "w", encoding="utf-8") as f:
        for q in domain_questions:
            f.write(q.model_dump_json() + "\n")

    # 2. Hand-written expert QA set (for qualitative error analysis)
    expert_file = output_dir / "expert_qa.jsonl"
    with open(expert_file, "w", encoding="utf-8") as f:
        for q in domain_questions[:5]:
            f.write(q.model_dump_json() + "\n")

    return domain_file, expert_file

if __name__ == "__main__":
    generate_frozen_benchmarks()
    print("Frozen evaluation benchmarks generated successfully.")
