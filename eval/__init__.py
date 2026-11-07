"""Evaluation Harness Package for RBI Regulatory Corpora RAG Benchmarks.

Modules:
- metrics: Hit@k (Hit@1, Hit@3, Hit@5), MRR, and stratified category metrics.
- judge: LLM-as-a-Judge, Faithfulness, Relevance, Citation Precision, and 20-sample calibration.
- eval_runner: Execution harness across the 60-question golden dataset.
"""

from eval.metrics import (
    hit_at_k,
    reciprocal_rank,
    compute_aggregate_retrieval_metrics,
    RetrievalMetricsSummary,
)
from eval.judge import (
    LLMJudge,
    JudgeResult,
    JudgeCalibration,
)
from eval.eval_runner import EvaluationRunner

__all__ = [
    "hit_at_k",
    "reciprocal_rank",
    "compute_aggregate_retrieval_metrics",
    "RetrievalMetricsSummary",
    "LLMJudge",
    "JudgeResult",
    "JudgeCalibration",
    "EvaluationRunner",
]
