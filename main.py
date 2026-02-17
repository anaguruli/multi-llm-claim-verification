"""
Main orchestrator for the Multi-LLM Fact-Checking Pipeline.

Stages:
  1. Claim Decomposition
  2. Evidence Retrieval (RAG)
  3. Independent Verification (3 verifiers)
  4. Cross-Verification Deliberation
  5. Final Synthesis
  6. Baseline evaluation
  7. Metrics & Visualisation
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from tqdm import tqdm

import config
from data.build_knowledge_base import build_knowledge_base
from data.claims import CLAIMS
from evaluation.metrics import EvaluationReport, compute_record
from evaluation.visualizer import generate_all_plots
from pipeline.baseline import BaselineEvaluator
from pipeline.decomposer import ClaimDecomposer
from pipeline.deliberation import DeliberationEngine
from pipeline.retriever import RAGRetriever
from pipeline.synthesizer import ClaimSynthesizer
from pipeline.verifier import VerifierPanel

# ── Helpers ───────────────────────────────────────────────────────────────────

RESULTS_FILE = os.path.join(config.RESULTS_DIR, "full_results.json")
EVAL_FILE = os.path.join(config.RESULTS_DIR, "evaluation_report.json")


def _sleep(seconds: float = 1.5) -> None:
    """Polite pause between API calls to avoid rate limits."""
    time.sleep(seconds)


def _save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[IO] Saved → {path}")


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_decomposition(decomposer: ClaimDecomposer, claim: Dict[str, Any]) -> Dict[str, Any]:
    result = decomposer.decompose(claim["id"], claim["text"])
    _sleep()
    return result.to_dict()


def run_retrieval(retriever: RAGRetriever, decomposition: Dict[str, Any]) -> Dict[str, Any]:
    results = retriever.retrieve_batch(decomposition["sub_claims"])
    return {
        r.sub_claim_id: r.to_dict()["retrieved_evidence"]
        for r in results
    }


def run_verification(
    panel: VerifierPanel,
    sub_claims: List[Dict[str, Any]],
    evidence_map: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Returns {sc_id: {v1: VerifierResult.to_dict(), v2: ..., v3: ...}}"""
    verification_results: Dict[str, Dict[str, Any]] = {}
    for sc in sub_claims:
        sc_id = sc["id"]
        evidence = evidence_map.get(sc_id, [])
        verifier_results = panel.verify_sub_claim(sc_id, sc["text"], evidence)
        verification_results[sc_id] = {
            vid: vr.to_dict() for vid, vr in verifier_results.items()
        }
        _sleep(1.0)
    return verification_results


def run_deliberation(
    engine: DeliberationEngine,
    sub_claims: List[Dict[str, Any]],
    verification_results: Dict[str, Dict[str, Any]],
    evidence_map: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    from pipeline.verifier import VerifierResult

    deliberation_results = []
    for sc in sub_claims:
        sc_id = sc["id"]
        raw_vr = verification_results.get(sc_id, {})

        # Reconstruct VerifierResult objects from dicts
        initial_results = {}
        for vid, vd in raw_vr.items():
            initial_results[vid] = VerifierResult(
                verifier_id=vid,
                sub_claim_id=sc_id,
                verdict=vd["verdict"],
                confidence=vd["confidence"],
                reasoning=vd["reasoning"],
                evidence_sufficiency=vd["evidence_sufficiency"],
            )

        evidence = evidence_map.get(sc_id, [])
        delib = engine.deliberate(sc_id, sc["text"], initial_results, evidence)
        deliberation_results.append(delib.to_dict())
        _sleep(1.0)

    return deliberation_results


def run_synthesis(
    synthesizer: ClaimSynthesizer,
    claim: Dict[str, Any],
    decomposition: Dict[str, Any],
    evidence_map: Dict[str, List[Dict[str, Any]]],
    deliberation_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    result = synthesizer.synthesize(
        claim_id=claim["id"],
        original_claim=claim["text"],
        sub_claims=decomposition["sub_claims"],
        evidence_map=evidence_map,
        deliberation_results=deliberation_results,
    )
    _sleep()
    return result.to_dict()


def run_baseline(evaluator: BaselineEvaluator, claim: Dict[str, Any]) -> Dict[str, Any]:
    result = evaluator.evaluate(claim["id"], claim["text"])
    _sleep()
    return result.to_dict()


# ── Pre-deliberation verdict helper ──────────────────────────────────────────

def _majority_verdict_from_verification(
    verification_results: Dict[str, Dict[str, Any]],
    sub_claims: List[Dict[str, Any]],
) -> str:
    """Compute what the synthesizer would likely produce before deliberation."""
    all_verdicts: List[str] = []
    for sc in sub_claims:
        sc_id = sc["id"]
        for vid in ["v1", "v2", "v3"]:
            vd = verification_results.get(sc_id, {}).get(vid, {})
            if vd:
                all_verdicts.append(vd.get("verdict", "FALSE"))
    if not all_verdicts:
        return "FALSE"
    from collections import Counter
    return Counter(all_verdicts).most_common(1)[0][0]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 60)
    print("  MULTI-LLM FACT-CHECKING PIPELINE")
    print("═" * 60)

    # ── Setup ──────────────────────────────────────────────────────────────
    print("\n[Setup] Building knowledge base …")
    build_knowledge_base(config.KB_DIR)

    print("[Setup] Initialising components …")
    retriever = RAGRetriever()
    decomposer = ClaimDecomposer()
    verifier_panel = VerifierPanel()
    deliberation_engine = DeliberationEngine()
    synthesizer = ClaimSynthesizer()
    baseline_evaluator = BaselineEvaluator()

    all_results: List[Dict[str, Any]] = []
    eval_report = EvaluationReport()

    # ── Process each claim ─────────────────────────────────────────────────
    for claim in tqdm(CLAIMS, desc="Processing claims", unit="claim"):
        cid = claim["id"]
        print(f"\n{'─' * 55}")
        print(f"[Claim {cid}] {claim['text'][:80]}…")
        print(f"  Category: {claim['category']}  |  Ground truth: {claim['ground_truth']}")

        claim_record: Dict[str, Any] = {
            "claim_id": cid,
            "claim_text": claim["text"],
            "category": claim["category"],
            "ground_truth": claim["ground_truth"],
            "ground_truth_justification": claim["justification"],
        }

        # Stage 1: Decomposition
        print(f"  [Stage 1] Decomposing claim …")
        decomposition = run_decomposition(decomposer, claim)
        n_sub = len(decomposition["sub_claims"])
        print(f"    → {n_sub} sub-claims generated")
        claim_record["decomposition"] = decomposition

        # Stage 2: Retrieval
        print(f"  [Stage 2] Retrieving evidence for {n_sub} sub-claims …")
        evidence_map = run_retrieval(retriever, decomposition)
        claim_record["evidence_map"] = evidence_map

        # Stage 3: Independent Verification
        print(f"  [Stage 3] Running 3 independent verifiers …")
        verification_results = run_verification(
            verifier_panel, decomposition["sub_claims"], evidence_map
        )
        claim_record["verification_results"] = verification_results

        # Pre-deliberation majority verdict (for deliberation impact tracking)
        pre_delib_verdict = _majority_verdict_from_verification(
            verification_results, decomposition["sub_claims"]
        )

        # Stage 4: Deliberation
        print(f"  [Stage 4] Running deliberation …")
        deliberation_results = run_deliberation(
            deliberation_engine,
            decomposition["sub_claims"],
            verification_results,
            evidence_map,
        )
        n_disagreements = sum(d["had_disagreement"] for d in deliberation_results)
        n_changes = sum(
            any(e["changed"] for e in d["deliberation"])
            for d in deliberation_results
            if d["deliberation"]
        )
        print(f"    → Disagreements: {n_disagreements}/{n_sub} | Mind-changes: {n_changes}")
        claim_record["deliberation_results"] = deliberation_results

        # Stage 5: Synthesis
        print(f"  [Stage 5] Synthesising final verdict …")
        synthesis = run_synthesis(
            synthesizer, claim, decomposition, evidence_map, deliberation_results
        )
        print(f"    → Final verdict: {synthesis['final_verdict']} "
              f"(confidence: {synthesis['confidence']:.2f})")
        claim_record["synthesis"] = synthesis

        # Baseline
        print(f"  [Baseline] Running single-LLM baseline …")
        baseline = run_baseline(baseline_evaluator, claim)
        print(f"    → Baseline verdict: {baseline['verdict']}")
        claim_record["baseline"] = baseline

        # ── Evaluation record ─────────────────────────────────────────────
        had_any_disagreement = any(d["had_disagreement"] for d in deliberation_results)
        deliberation_changed = any(
            any(e["changed"] for e in d["deliberation"])
            for d in deliberation_results
        )

        eval_rec = compute_record(
            claim=claim,
            system_verdict=synthesis["final_verdict"],
            baseline_verdict=baseline["verdict"],
            system_confidence=synthesis["confidence"],
            num_sub_claims=n_sub,
            had_any_disagreement=had_any_disagreement,
            deliberation_changed=deliberation_changed,
            pre_deliberation_system_verdict=pre_delib_verdict,
        )
        eval_report.records.append(eval_rec)

        # Attach eval fields to record for serialisation
        claim_record["eval"] = {
            "system_correct": eval_rec.system_correct,
            "baseline_correct": eval_rec.baseline_correct,
            "system_confidence": eval_rec.system_confidence,
            "deliberation_changed": eval_rec.deliberation_changed,
            "deliberation_changed_outcome": eval_rec.deliberation_changed_outcome,
            "had_any_disagreement": eval_rec.had_any_disagreement,
            "error_analysis": eval_rec.error_analysis,
        }

        correctness_symbol = "✓" if eval_rec.system_correct else "✗"
        print(f"  [{correctness_symbol}] System: {synthesis['final_verdict']} | "
              f"Truth: {claim['ground_truth']} | "
              f"Baseline: {baseline['verdict']} ({'✓' if eval_rec.baseline_correct else '✗'})")

        all_results.append(claim_record)

    # ── Save full results ──────────────────────────────────────────────────
    _save_json(all_results, RESULTS_FILE)

    # ── Evaluation report ──────────────────────────────────────────────────
    eval_report.print_summary()
    _save_json(eval_report.to_dict(), EVAL_FILE)

    # ── Visualisations ─────────────────────────────────────────────────────
    print("\n[Visualizer] Generating plots …")

    # Build simplified records list for plotting
    plot_records = [
        {
            "claim_id": r.claim_id,
            "category": r.category,
            "ground_truth": r.ground_truth,
            "system_verdict": r.system_verdict,
            "baseline_verdict": r.baseline_verdict,
            "system_correct": r.system_correct,
            "baseline_correct": r.baseline_correct,
            "system_confidence": r.system_confidence,
            "deliberation_changed": r.deliberation_changed,
            "deliberation_changed_outcome": r.deliberation_changed_outcome,
            "had_any_disagreement": r.had_any_disagreement,
        }
        for r in eval_report.records
    ]

    generate_all_plots(eval_report, plot_records)

    # ── Error discussion ───────────────────────────────────────────────────
    _print_error_discussion(eval_report, all_results)

    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Results  → {RESULTS_FILE}")
    print(f"  Metrics  → {EVAL_FILE}")
    print(f"  Plots    → {config.PLOTS_DIR}/")
    print("═" * 60)


def _print_error_discussion(
    report: EvaluationReport,
    all_results: List[Dict[str, Any]],
) -> None:
    failed = [r for r in report.records if not r.system_correct]
    print(f"\n{'═' * 60}")
    print("  ERROR ANALYSIS")
    print("═" * 60)
    if not failed:
        print("  All claims correctly classified!")
        return

    print(f"  {len(failed)} claim(s) incorrectly classified:\n")
    for rec in failed:
        # Find matching full result for synthesis reasoning
        full = next((r for r in all_results if r["claim_id"] == rec.claim_id), {})
        synthesis = full.get("synthesis", {})
        print(f"  [{rec.claim_id}] Ground truth: {rec.ground_truth} → "
              f"System predicted: {rec.system_verdict}")
        print(f"    Category : {rec.category}")
        print(f"    Analysis : {rec.error_analysis}")
        if synthesis.get("limitations"):
            print(f"    Limitations noted by synthesizer: {synthesis['limitations'][:150]}")
        print()
    print("═" * 60)


if __name__ == "__main__":
    main()