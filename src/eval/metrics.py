import re
import string
from typing import List, Set, Dict, Any
import numpy as np

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def recall_at_k(gold_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    """Recall@k of the gold supporting documents."""
    if not gold_ids:
        return 1.0
    top_k_retrieved = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    hits = len(gold_set.intersection(top_k_retrieved))
    return hits / len(gold_set)

def all_supporting_found_rate(gold_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    """1.0 if ALL gold supporting documents are retrieved within top-k, else 0.0."""
    if not gold_ids:
        return 1.0
    top_k_retrieved = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    return 1.0 if gold_set.issubset(top_k_retrieved) else 0.0

def compute_system_metrics(latencies: List[float]) -> Dict[str, float]:
    """Compute median, p95, mean latency."""
    if not latencies:
        return {"median_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    arr = np.array(latencies)
    return {
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(np.mean(arr))
    }
