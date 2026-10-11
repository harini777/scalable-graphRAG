import os
import json
import time
import datetime
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import settings
from src.models import AnswerResponse
from src.eval.benchmarks import EvalQuestion, generate_frozen_benchmarks
from src.eval.metrics import (
    recall_at_k,
    all_supporting_found_rate,
    exact_match_score,
    f1_score,
    compute_system_metrics
)
from src.index.vector_store import VectorIndex
from src.retrieve.baseline import PlainRAGPipeline

class EvaluationHarness:
    def __init__(self, run_dir: Optional[Path] = None):
        self.run_dir = run_dir or (settings.BASE_DIR / "eval_runs")
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def load_benchmark(self, benchmark_file: Path) -> List[EvalQuestion]:
        questions = []
        with open(benchmark_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(EvalQuestion.model_validate_json(line))
        return questions

    def evaluate_pipeline(
        self,
        pipeline: Any,
        system_name: str,
        benchmark_file: Path,
        top_k: int = 5
    ) -> Dict[str, Any]:
        questions = self.load_benchmark(benchmark_file)
        results = []
        latencies = []
        recalls_at_k = []
        all_found = []
        f1_scores = []
        em_scores = []

        print(f"\n=======================================================")
        print(f"Evaluating {system_name} on {benchmark_file.name} ({len(questions)} items, top_k={top_k})")
        print(f"=======================================================")

        for q in questions:
            start_t = time.time()
            resp: AnswerResponse = pipeline.query(q.question, top_k=top_k)
            duration_ms = (time.time() - start_t) * 1000

            # Gather retrieved paper IDs
            retrieved_paper_ids = [
                r.paper_id or r.metadata.get("arxiv_id", "")
                for r in resp.retrieved_results
            ]

            r_k = recall_at_k(q.gold_paper_ids, retrieved_paper_ids, k=top_k)
            all_f = all_supporting_found_rate(q.gold_paper_ids, retrieved_paper_ids, k=top_k)
            em = exact_match_score(resp.answer, q.gold_answer)
            f1 = f1_score(resp.answer, q.gold_answer)

            recalls_at_k.append(r_k)
            all_found.append(all_f)
            em_scores.append(em)
            f1_scores.append(f1)
            latencies.append(duration_ms)

            results.append({
                "question_id": q.id,
                "question": q.question,
                "question_type": q.question_type,
                "gold_answer": q.gold_answer,
                "gold_paper_ids": q.gold_paper_ids,
                "predicted_answer": resp.answer,
                "retrieved_paper_ids": retrieved_paper_ids,
                "citations": [c.paper_id for c in resp.citations],
                "recall_at_k": r_k,
                "all_supporting_found": all_f,
                "exact_match": em,
                "f1": f1,
                "latency_ms": duration_ms
            })

            print(f"[{q.id}] {q.question_type} | Recall@{top_k}: {r_k:.2f} | All-Found: {all_f} | Latency: {duration_ms:.1f}ms")

        system_timing = compute_system_metrics(latencies)

        summary = {
            "system_name": system_name,
            "benchmark_dataset": benchmark_file.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "total_questions": len(questions),
            "top_k": top_k,
            "mean_recall_at_k": float(sum(recalls_at_k) / len(recalls_at_k)) if recalls_at_k else 0.0,
            "all_supporting_found_rate": float(sum(all_found) / len(all_found)) if all_found else 0.0,
            "mean_f1": float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0,
            "mean_exact_match": float(sum(em_scores) / len(em_scores)) if em_scores else 0.0,
            "latency_median_ms": system_timing["median_ms"],
            "latency_p95_ms": system_timing["p95_ms"],
            "latency_mean_ms": system_timing["mean_ms"],
            "detailed_results": results
        }

        # Save to file
        safe_name = system_name.lower().replace(" ", "_").replace(":", "")
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.run_dir / f"{safe_name}_{timestamp_str}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n--- Summary for {system_name} ---")
        print(f"Mean Recall@{top_k}: {summary['mean_recall_at_k']:.4f}")
        print(f"All-Supporting-Found Rate: {summary['all_supporting_found_rate']:.4f}")
        print(f"Median Latency: {summary['latency_median_ms']:.2f} ms")
        print(f"p95 Latency: {summary['latency_p95_ms']:.2f} ms")
        print(f"Results saved to: {log_file}\n")

        return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # Ensure benchmarks exist
    bench_file = settings.BENCHMARK_DIR / "domain_multihop.jsonl"
    if not bench_file.exists():
        bench_file, _ = generate_frozen_benchmarks()

    # Load baseline index
    index_dir = settings.DATA_DIR / "indices" / "plain_rag"
    if not (index_dir / "plain_rag.faiss").exists():
        print("Plain RAG index not found. Building index first...")
        from src.index.build_index import build_vector_index
        from src.ingest.arxiv import create_seed_papers_dataset
        seed_f = settings.DATA_DIR / "papers_seed.jsonl"
        create_seed_papers_dataset(seed_f)
        build_vector_index(seed_f, index_dir, "hnsw")

    index = VectorIndex.load(index_dir, prefix="plain_rag")
    pipeline = PlainRAGPipeline(vector_index=index, top_k=args.top_k)

    harness = EvaluationHarness()
    harness.evaluate_pipeline(pipeline, "System A: Plain RAG", bench_file, top_k=args.top_k)
