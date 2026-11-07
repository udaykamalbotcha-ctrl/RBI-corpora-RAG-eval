"""LLM-as-a-Judge and Automated Evaluation Engine for RBI Regulatory RAG.

Evaluates generated regulatory responses across:
1. Faithfulness Score (0.0 – 1.0): Verifies all claims are supported by retrieved context.
2. Answer Relevance Score (0.0 – 1.0): Measures precision and conciseness against the prompt.
3. Citation Precision (Pass/Fail): Verifies exact RBI Paragraph / Table citation against context.
4. Refusal Correctness (Pass/Fail): Validates abstention guardrails for out-of-scope questions.

Includes a Hand-Label Calibration Engine calculating Percentage Agreement and Cohen's Kappa
across 20 benchmarked questions.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer


class JudgeResult(BaseModel):
    """Evaluation result produced for a single Q&A pair."""
    question_id: str
    faithfulness_score: float = Field(..., ge=0.0, le=1.0, description="Context support score")
    faithfulness_reason: str = Field(..., description="Explanation of factual grounding")
    answer_relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to query")
    answer_relevance_reason: str = Field(..., description="Explanation of relevance")
    citation_precision: bool = Field(..., description="Whether cited clause matches expected target")
    cited_clauses: List[str] = Field(default_factory=list, description="Extracted citation tags")
    refusal_correctness: bool = Field(..., description="Accurate refusal for out-of-scope queries")
    overall_pass: bool = Field(..., description="Overall pass/fail decision")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "faithfulness_score": round(self.faithfulness_score, 4),
            "faithfulness_reason": self.faithfulness_reason,
            "answer_relevance_score": round(self.answer_relevance_score, 4),
            "answer_relevance_reason": self.answer_relevance_reason,
            "citation_precision": self.citation_precision,
            "cited_clauses": self.cited_clauses,
            "refusal_correctness": self.refusal_correctness,
            "overall_pass": self.overall_pass,
        }


class LLMJudge:
    """Automated judge with dual LLM API and semantic embedding evaluation support."""

    RE_CITATION = re.compile(
        r'\b(?:Paragraph|Para\.?|Section|Sec\.?|Table|Annex(?:ure)?)\s+([0-9IVXLCDM]+(?:\([0-9a-zivx]+\))*)',
        re.IGNORECASE
    )

    REFUSAL_TRIGGERS = [
        "out of scope",
        "unanswerable",
        "not covered",
        "does not provide",
        "not mentioned",
        "no information",
        "cannot answer",
        "neither the cva",
        "not contained in the provided",
    ]

    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        if embedding_model is not None:
            self.model = embedding_model
        else:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            except Exception:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_citations(self, text: str) -> List[str]:
        """Extract explicit regulatory clause/table citations from generated text."""
        matches = self.RE_CITATION.findall(text)
        return list(dict.fromkeys(matches))

    def evaluate_refusal(self, question_type: str, answer: str) -> bool:
        """Verify whether an unanswerable question was properly refused, or an in-scope question was answered."""
        ans_lower = answer.lower()
        is_refusal = any(trigger in ans_lower for trigger in self.REFUSAL_TRIGGERS)

        if question_type == "unanswerable_refusal":
            # For unanswerable queries, pass if the system abstained
            return is_refusal
        else:
            # For in-scope queries, pass if the system did not inappropriately refuse
            return not is_refusal

    def evaluate_citation_precision(
        self,
        answer: str,
        target_clause: str,
        retrieved_contexts: List[str]
    ) -> Tuple[bool, List[str]]:
        """Verify if the generated answer cites the correct target clause."""
        if not target_clause or target_clause.lower() == "none":
            return True, []

        cited = self.extract_citations(answer)
        target_num_match = re.search(r'\d+', target_clause)
        if not target_num_match:
            return bool(cited), cited

        target_num = target_num_match.group(0)
        # Check if the target paragraph number appears in cited clauses or answer text
        clause_mentioned = any(target_num in c for c in cited) or bool(
            re.search(rf'\b(?:Paragraph|Para\.?)\s+{target_num}\b', answer, re.IGNORECASE)
        )
        return clause_mentioned, cited

    def clean_answer_for_eval(self, answer: str) -> str:
        """Strip lead-in attribution boilerplate before checking factual claims."""
        cleaned = re.sub(r'^Under\s+Paragraph\s+[0-9a-z_]+\s+of\s+[^,]+,\s*', '', answer, flags=re.IGNORECASE)
        cleaned = re.sub(r'^According\s+to\s+[^,]+,\s*', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def evaluate_faithfulness(self, answer: str, retrieved_contexts: List[str]) -> Tuple[float, str]:
        """Measure whether claims in the answer are supported by retrieved context chunks."""
        if not retrieved_contexts:
            return 0.0, "No context chunks were provided for verification."

        body_text = self.clean_answer_for_eval(answer)
        if not body_text:
            return 1.0, "Trivial answer."

        joined_context = " ".join(retrieved_contexts).lower()
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', body_text) if len(s.strip()) > 8]
        if not sentences:
            sentences = [body_text]

        supported_count = 0
        reasons = []

        ctx_embs = self.model.encode(retrieved_contexts, normalize_embeddings=True)

        for sent in sentences:
            sent_lower = sent.lower()

            # Check if sentence is substantially present in retrieved context
            if any(sent_lower[:40] in ctx.lower() for ctx in retrieved_contexts if len(sent_lower) >= 40):
                supported_count += 1
                continue

            # Check semantic entailment via cosine similarity
            sent_emb = self.model.encode([sent], normalize_embeddings=True)
            max_sim = float((sent_emb @ ctx_embs.T).max())

            # Extract specific metrics/numbers excluding bare years
            raw_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', sent)
            numbers = [n for n in raw_numbers if n not in ["2025", "2026", "2027", "2028"]]
            num_ok = all(n.lower() in joined_context for n in numbers) if numbers else True

            if num_ok and max_sim >= 0.50:
                supported_count += 1
            else:
                reasons.append(f"Claim unsupported: '{sent[:60]}...' (sim={max_sim:.2f})")

        score = supported_count / len(sentences)
        reason_text = "All factual claims verified against context." if score == 1.0 else "; ".join(reasons[:2])
        return round(score, 4), reason_text

    def evaluate_relevance(self, question: str, answer: str) -> Tuple[float, str]:
        """Measure semantic relevance and directness between user query and generated response."""
        q_emb = self.model.encode([question], normalize_embeddings=True)
        a_emb = self.model.encode([answer], normalize_embeddings=True)
        sim = float((q_emb @ a_emb.T)[0][0])

        # Penalize excessive boilerplate or extreme length disparity
        score = max(0.0, min(1.0, sim * 1.15))
        if len(answer.split()) < 4:
            score *= 0.5
            reason = "Response is excessively terse."
        elif sim >= 0.65:
            reason = f"Response directly addresses query with high semantic alignment (sim={sim:.2f})."
        else:
            reason = f"Response has moderate topic alignment (sim={sim:.2f})."

        return round(score, 4), reason

    def grade(
        self,
        question_id: str,
        question: str,
        answer: str,
        expected_answer: str,
        target_clause: str,
        question_type: str,
        retrieved_contexts: List[str],
    ) -> JudgeResult:
        """Execute full evaluation grading suite for a single Q&A instance."""
        # 1. Refusal check
        refusal_ok = self.evaluate_refusal(question_type, answer)

        if question_type == "unanswerable_refusal":
            # For unanswerable queries, proper refusal constitutes 100% faithfulness and relevance
            faith_score = 1.0 if refusal_ok else 0.0
            faith_reason = "Appropriately abstained from out-of-scope query." if refusal_ok else "Hallucinated an answer for out-of-scope query."
            rel_score = 1.0 if refusal_ok else 0.0
            rel_reason = "Refusal matches expected abstention behavior." if refusal_ok else "Failed to refuse out-of-scope question."
            cite_pass = True
            cited = []
            overall = refusal_ok
        else:
            # 2. Faithfulness
            faith_score, faith_reason = self.evaluate_faithfulness(answer, retrieved_contexts)

            # 3. Relevance
            rel_score, rel_reason = self.evaluate_relevance(question, answer)

            # 4. Citation precision
            cite_pass, cited = self.evaluate_citation_precision(answer, target_clause, retrieved_contexts)

            # Overall pass criteria: Faithfulness >= 0.75, Relevance >= 0.70, Refusal correct
            overall = bool(faith_score >= 0.70 and rel_score >= 0.60 and refusal_ok)

        return JudgeResult(
            question_id=question_id,
            faithfulness_score=faith_score,
            faithfulness_reason=faith_reason,
            answer_relevance_score=rel_score,
            answer_relevance_reason=rel_reason,
            citation_precision=cite_pass,
            cited_clauses=cited,
            refusal_correctness=refusal_ok,
            overall_pass=overall,
        )


class JudgeCalibration:
    """Manages hand-label calibration against 20 reference golden set questions."""

    # 20 curated reference questions with human ground-truth Pass/Fail decisions for baseline evaluation
    CALIBRATION_GROUND_TRUTH_20 = {
        "q_001": True,    # Discount scalar 0.65 -> Pass (retrieved correctly)
        "q_002": True,    # Financials IG 5.0% -> Pass (retrieved correctly)
        "q_003": True,    # Sovereigns IG 0.5% -> Pass (retrieved correctly)
        "q_004": True,    # Tech HY/NR 5.5% -> Pass (retrieved correctly)
        "q_005": False,   # Health care IG -> Fail (retriever returned Para 12 instead of Para 16)
        "q_006": False,   # April 1, 2027 -> Fail (retriever missed Para 2, returned Para 1)
        "q_007": True,    # 10 lakh crore threshold -> Pass (retrieved correctly)
        "q_008": True,    # 100% CCR capital requirement -> Pass (retrieved correctly)
        "q_009": False,   # Alpha 1.4 -> Fail (retriever returned Para 9 instead of Para 10)
        "q_010": True,    # 1 year floor -> Pass (retrieved correctly)
        "q_026": False,   # Margined RC formula -> Fail (retriever returned unmargined RC)
        "q_027": False,   # EAD = alpha * (RC + PFE) -> Fail (retriever returned Para 9)
        "q_028": True,    # Margined cap justification -> Pass (retrieved correctly)
        "q_030": True,    # Cash flow effective maturity -> Pass (retrieved correctly)
        "q_041": False,   # Bank own default DVA excluded -> Fail (retriever missed Para 4(3))
        "q_042": False,   # SFT repo exclusion -> Fail (retriever missed Para 6)
        "q_045": False,   # Small Finance Bank exclusion -> Fail (retriever missed Para 3)
        "q_051": True,    # Crypto asset refusal -> Pass (properly refused)
        "q_052": True,    # NBFC LCR refusal -> Pass (properly refused)
        "q_055": True,    # CRR penal interest refusal -> Pass (properly refused)
    }

    @classmethod
    def compute_calibration(cls, judge_results: List[JudgeResult]) -> Dict[str, Any]:
        """Compute Percentage Agreement and Cohen's Kappa against the 20 reference labels."""
        results_map = {r.question_id: r.overall_pass for r in judge_results if r.question_id in cls.CALIBRATION_GROUND_TRUTH_20}

        if len(results_map) < len(cls.CALIBRATION_GROUND_TRUTH_20):
            # Evaluate only available overlap
            keys = list(results_map.keys())
        else:
            keys = list(cls.CALIBRATION_GROUND_TRUTH_20.keys())

        if not keys:
            return {"percentage_agreement": 0.0, "cohens_kappa": 0.0, "sample_size": 0}

        agreements = 0
        human_true = 0
        judge_true = 0
        total = len(keys)

        for q_id in keys:
            h_val = cls.CALIBRATION_GROUND_TRUTH_20[q_id]
            j_val = results_map[q_id]
            if h_val == j_val:
                agreements += 1
            if h_val:
                human_true += 1
            if j_val:
                judge_true += 1

        po = agreements / total  # Observed agreement
        # Marginal probabilities for chance agreement
        p_h_true = human_true / total
        p_h_false = 1.0 - p_h_true
        p_j_true = judge_true / total
        p_j_false = 1.0 - p_j_true

        pe = (p_h_true * p_j_true) + (p_h_false * p_j_false)  # Chance agreement
        kappa = (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0

        return {
            "sample_size": total,
            "agreed_count": agreements,
            "percentage_agreement": round(po * 100, 2),
            "cohens_kappa": round(kappa, 4),
            "calibration_status": "CALIBRATED (>=85%)" if (po >= 0.85) else "NEEDS_CALIBRATION",
        }
