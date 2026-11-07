"""Retrieval-Layer and Generation Metrics Module for Regulatory RAG Evaluation.

Implements standard IR (Information Retrieval) metrics:
- Hit@k (Precision at rank k: Hit@1, Hit@3, Hit@5)
- Mean Reciprocal Rank (MRR)
- Category-stratified aggregation for regulatory compliance benchmarks
"""

import re
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field


def normalize_clause_tag(clause: str) -> str:
    """Normalize clause strings (e.g., 'Paragraph 13', 'paragraph 13(1)', 'p13') to a clean standard form."""
    if not clause or clause.lower() == "none":
        return "none"
    norm = clause.lower().strip()
    norm = re.sub(r'[^a-z0-9]', '', norm)
    # Convert 'paragraph13' -> 'para13', 'p13' -> 'para13'
    norm = re.sub(r'^(paragraph|para|p)', 'para', norm)
    return norm


def is_clause_match(retrieved_clause: str, target_clause: str, retrieved_text: str = "") -> bool:
    """Check whether a retrieved chunk corresponds to the target regulatory clause."""
    norm_target = normalize_clause_tag(target_clause)
    if norm_target == "none":
        return False

    norm_retrieved = normalize_clause_tag(retrieved_clause)
    if norm_target == norm_retrieved:
        return True

    # Extract paragraph number digits: e.g. 'Paragraph 13' -> '13'
    target_num_match = re.search(r'\d+', target_clause)
    if target_num_match:
        target_num = target_num_match.group(0)
        retrieved_num_match = re.search(r'\d+', retrieved_clause)
        if retrieved_num_match and retrieved_num_match.group(0) == target_num:
            return True

        # Check if paragraph heading occurs prominently at the start of chunk text
        if re.search(rf'^\s*\[?[^\]]*\bParagraph\s+{target_num}\b', retrieved_text, re.IGNORECASE):
            return True
        if re.search(rf'^\s*{target_num}\.\s+', retrieved_text):
            return True

    return False


def hit_at_k(retrieved_chunks: List[Dict[str, Any]], target_clause: str, k: int = 3) -> float:
    """Calculate Hit@k (1.0 if target clause appears within top k retrieved chunks, else 0.0)."""
    if not target_clause or target_clause.lower() == "none":
        # For unanswerable questions, a 'hit' means the retriever correctly returned low similarity or none
        return 1.0 if not retrieved_chunks else 0.0

    top_k_chunks = retrieved_chunks[:k]
    for chunk in top_k_chunks:
        ret_clause = chunk.get("metadata", {}).get("clause_number", "")
        ret_text = chunk.get("text", "")
        if is_clause_match(ret_clause, target_clause, ret_text):
            return 1.0
    return 0.0


def reciprocal_rank(retrieved_chunks: List[Dict[str, Any]], target_clause: str) -> float:
    """Calculate Reciprocal Rank (1/rank of first relevant chunk, or 0.0 if not found)."""
    if not target_clause or target_clause.lower() == "none":
        return 1.0 if not retrieved_chunks else 0.0

    for rank_idx, chunk in enumerate(retrieved_chunks, start=1):
        ret_clause = chunk.get("metadata", {}).get("clause_number", "")
        ret_text = chunk.get("text", "")
        if is_clause_match(ret_clause, target_clause, ret_text):
            return 1.0 / rank_idx
    return 0.0


class RetrievalMetricsSummary(BaseModel):
    """Structured container for aggregate retrieval metrics."""
    total_queries: int
    mean_hit_at_1: float
    mean_hit_at_3: float
    mean_hit_at_5: float
    mrr: float
    by_category: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "mean_hit_at_1": round(self.mean_hit_at_1, 4),
            "mean_hit_at_3": round(self.mean_hit_at_3, 4),
            "mean_hit_at_5": round(self.mean_hit_at_5, 4),
            "mrr": round(self.mrr, 4),
            "by_category": self.by_category,
        }


def compute_aggregate_retrieval_metrics(
    eval_results: List[Dict[str, Any]]
) -> RetrievalMetricsSummary:
    """Compute aggregate Hit@1, Hit@3, Hit@5 and MRR across questions and categories."""
    if not eval_results:
        return RetrievalMetricsSummary(
            total_queries=0,
            mean_hit_at_1=0.0,
            mean_hit_at_3=0.0,
            mean_hit_at_5=0.0,
            mrr=0.0,
        )

    # Filter out unanswerable queries for standard clause hit calculation,
    # or score them appropriately
    clause_evals = [r for r in eval_results if r.get("type") != "unanswerable_refusal"]

    hit_1_list = [r["retrieval_metrics"]["hit_at_1"] for r in clause_evals]
    hit_3_list = [r["retrieval_metrics"]["hit_at_3"] for r in clause_evals]
    hit_5_list = [r["retrieval_metrics"]["hit_at_5"] for r in clause_evals]
    mrr_list = [r["retrieval_metrics"]["reciprocal_rank"] for r in clause_evals]

    categories: Dict[str, List[Dict[str, Any]]] = {}
    for r in eval_results:
        cat = r.get("type", "unknown")
        categories.setdefault(cat, []).append(r)

    by_category_summary = {}
    for cat, items in categories.items():
        if cat == "unanswerable_refusal":
            # For unanswerable, evaluate abstention / refusal metrics
            refusal_rate = sum(
                1.0 for item in items if item.get("judge_metrics", {}).get("refusal_correctness", False)
            ) / len(items) if items else 0.0
            by_category_summary[cat] = {
                "count": len(items),
                "refusal_accuracy": round(refusal_rate, 4),
            }
        else:
            cat_hit3 = [x["retrieval_metrics"]["hit_at_3"] for x in items]
            cat_mrr = [x["retrieval_metrics"]["reciprocal_rank"] for x in items]
            by_category_summary[cat] = {
                "count": len(items),
                "mean_hit_at_1": round(sum(x["retrieval_metrics"]["hit_at_1"] for x in items) / len(items), 4),
                "mean_hit_at_3": round(sum(cat_hit3) / len(items), 4),
                "mean_hit_at_5": round(sum(x["retrieval_metrics"]["hit_at_5"] for x in items) / len(items), 4),
                "mrr": round(sum(cat_mrr) / len(items), 4),
            }

    n = len(clause_evals) if clause_evals else 1
    return RetrievalMetricsSummary(
        total_queries=len(eval_results),
        mean_hit_at_1=sum(hit_1_list) / n,
        mean_hit_at_3=sum(hit_3_list) / n,
        mean_hit_at_5=sum(hit_5_list) / n,
        mrr=sum(mrr_list) / n,
        by_category=by_category_summary,
    )
