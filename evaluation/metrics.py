"""
Quantitative metrics and evaluation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

import numpy as np


VERDICT_MAP = {
    "TRUE": 0,
    "FALSE": 1,
    "PARTIALLY_TRUE": 2,
    "MISLEADING": 3,
}
ID_TO_VERDICT = {v: k for k, v in VERDICT_MAP.items()}


@dataclass
class EvaluationRecord:
    claim_id: str
    category: str
    ground_truth: str
    system_verdict: str
    baseline_verdict: str
    system_correct: bool
    baseline_correct: bool
    deliberation_changed: bool
    deliberation_changed_outcome: bool
    system_confidence: float
    num_sub_claims: int
    had_any_disagreement: bool
    error_analysis: str = ""


@dataclass
class EvaluationReport:
    records: List[EvaluationRecord] = field(default_factory=list)

    #aggregate metrics

    @property
    def system_accuracy(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.system_correct for r in self.records) / len(self.records)

    @property
    def baseline_accuracy(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.baseline_correct for r in self.records) / len(self.records)

    @property
    def deliberation_change_rate(self) -> float:
        """% of claims where at least one verifier changed their verdict."""
        if not self.records:
            return 0.0
        return sum(r.deliberation_changed for r in self.records) / len(self.records)

    @property
    def deliberation_outcome_change_rate(self) -> float:
        """% of claims where deliberation changed the final outcome verdict."""
        if not self.records:
            return 0.0
        return sum(r.deliberation_changed_outcome for r in self.records) / len(self.records)

    @property
    def per_category_accuracy(self) -> Dict[str, Dict[str, float]]:
        from collections import defaultdict
        cat_records: Dict[str, List[EvaluationRecord]] = defaultdict(list)
        for r in self.records:
            cat_records[r.category].append(r)
        result = {}
        for cat, recs in cat_records.items():
            result[cat] = {
                "system_accuracy": sum(r.system_correct for r in recs) / len(recs),
                "baseline_accuracy": sum(r.baseline_correct for r in recs) / len(recs),
                "n": len(recs),
            }
        return result

    @property
    def confusion_matrix_system(self) -> Tuple[np.ndarray, List[str]]:
        labels = ["TRUE", "FALSE", "PARTIALLY_TRUE", "MISLEADING"]
        n = len(labels)
        matrix = np.zeros((n, n), dtype=int)
        for r in self.records:
            true_idx = VERDICT_MAP.get(r.ground_truth, 1)
            pred_idx = VERDICT_MAP.get(r.system_verdict, 1)
            matrix[true_idx][pred_idx] += 1
        return matrix, labels

    @property
    def confusion_matrix_baseline(self) -> Tuple[np.ndarray, List[str]]:
        labels = ["TRUE", "FALSE", "PARTIALLY_TRUE", "MISLEADING"]
        n = len(labels)
        matrix = np.zeros((n, n), dtype=int)
        for r in self.records:
            true_idx = VERDICT_MAP.get(r.ground_truth, 1)
            pred_idx = VERDICT_MAP.get(r.baseline_verdict, 1)
            matrix[true_idx][pred_idx] += 1
        return matrix, labels

    def print_summary(self) -> None:
        print("\n" + "═" * 60)
        print("EVALUATION SUMMARY")
        print("═" * 60)
        print(f"Total claims evaluated : {len(self.records)}")
        print(f"System accuracy        : {self.system_accuracy:.1%}")
        print(f"Baseline accuracy      : {self.baseline_accuracy:.1%}")
        print(f"Improvement            : {(self.system_accuracy - self.baseline_accuracy):+.1%}")
        print(f"Deliberation change rate    : {self.deliberation_change_rate:.1%}")
        print(f"Deliberation outcome change : {self.deliberation_outcome_change_rate:.1%}")
        print("\nPer-category accuracy:")
        for cat, stats in self.per_category_accuracy.items():
            print(
                f"  {cat:<35} system={stats['system_accuracy']:.1%}  "
                f"baseline={stats['baseline_accuracy']:.1%}  (n={stats['n']})"
            )
        print("═" * 60)

    def to_dict(self) -> dict:
        return {
            "system_accuracy": self.system_accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "improvement": self.system_accuracy - self.baseline_accuracy,
            "deliberation_change_rate": self.deliberation_change_rate,
            "deliberation_outcome_change_rate": self.deliberation_outcome_change_rate,
            "per_category_accuracy": self.per_category_accuracy,
            "records": [
                {
                    "claim_id": r.claim_id,
                    "category": r.category,
                    "ground_truth": r.ground_truth,
                    "system_verdict": r.system_verdict,
                    "baseline_verdict": r.baseline_verdict,
                    "system_correct": r.system_correct,
                    "baseline_correct": r.baseline_correct,
                    "deliberation_changed": r.deliberation_changed,
                    "deliberation_changed_outcome": r.deliberation_changed_outcome,
                    "error_analysis": r.error_analysis,
                }
                for r in self.records
            ],
        }


def compute_record(
    claim: Dict[str, Any],
    system_verdict: str,
    baseline_verdict: str,
    system_confidence: float,
    num_sub_claims: int,
    had_any_disagreement: bool,
    deliberation_changed: bool,
    pre_deliberation_system_verdict: str,
) -> EvaluationRecord:
    ground_truth = claim["ground_truth"]
    system_correct = system_verdict == ground_truth
    baseline_correct = baseline_verdict == ground_truth
    deliberation_changed_outcome = (
        deliberation_changed and pre_deliberation_system_verdict != system_verdict
    )

    # Simple error analysis
    error_analysis = ""
    if not system_correct:
        if system_verdict == "FALSE" and ground_truth == "PARTIALLY_TRUE":
            error_analysis = "Over-rejection: system treated partial truth as false."
        elif system_verdict == "TRUE" and ground_truth != "TRUE":
            error_analysis = "False acceptance: system treated non-true claim as true."
        elif system_verdict == "PARTIALLY_TRUE" and ground_truth == "FALSE":
            error_analysis = "Under-rejection: system too lenient on false claim."
        else:
            error_analysis = f"Misclassified {ground_truth} as {system_verdict}."

    return EvaluationRecord(
        claim_id=claim["id"],
        category=claim["category"],
        ground_truth=ground_truth,
        system_verdict=system_verdict,
        baseline_verdict=baseline_verdict,
        system_correct=system_correct,
        baseline_correct=baseline_correct,
        deliberation_changed=deliberation_changed,
        deliberation_changed_outcome=deliberation_changed_outcome,
        system_confidence=system_confidence,
        num_sub_claims=num_sub_claims,
        had_any_disagreement=had_any_disagreement,
        error_analysis=error_analysis,
    )