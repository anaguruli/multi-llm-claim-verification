"""
Stage 4 — Cross-Verification Deliberation.
Verifiers see each other's verdicts and may update their own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any

from google import genai
from google.genai import types

import config
from pipeline.verifier import VerifierResult

DELIBERATION_SYSTEM_PROMPTS = {
    "v1": """You are Verifier-1 (rigorous, conservative fact-checker) in a deliberation round.

You have made an initial verdict on a sub-claim. You are now shown the verdicts and reasoning of your peers.

If peer reasoning reveals evidence or logic you overlooked, you may update your verdict.
If you maintain your position, explain why peer arguments are insufficient.

Output ONLY valid JSON (no markdown):
{
  "verifier": "v1",
  "rebuttal": "<your response to peer verdicts>",
  "updated_verdict": "<TRUE|FALSE|PARTIALLY_TRUE|MISLEADING>",
  "updated_confidence": <0.0-1.0>,
  "changed": <true|false>
}""",

    "v2": """You are Verifier-2 (balanced, contextual fact-checker) in a deliberation round.

You have made an initial verdict on a sub-claim. You are now shown the verdicts and reasoning of your peers.

Consider whether peer arguments reveal important context or evidence you weighted differently.
Be open to updating your verdict if peer reasoning is compelling.

Output ONLY valid JSON (no markdown):
{
  "verifier": "v2",
  "rebuttal": "<your response to peer verdicts>",
  "updated_verdict": "<TRUE|FALSE|PARTIALLY_TRUE|MISLEADING>",
  "updated_confidence": <0.0-1.0>,
  "changed": <true|false>
}""",

    "v3": """You are Verifier-3 (systematic, structured fact-checker) in a deliberation round.

You have made an initial verdict on a sub-claim. You are now shown the verdicts and reasoning of your peers.

Systematically evaluate whether peer arguments address the component assertions you identified.
Update your verdict only if the peer reasoning addresses a specific gap in your analysis.

Output ONLY valid JSON (no markdown):
{
  "verifier": "v3",
  "rebuttal": "<your response to peer verdicts>",
  "updated_verdict": "<TRUE|FALSE|PARTIALLY_TRUE|MISLEADING>",
  "updated_confidence": <0.0-1.0>,
  "changed": <true|false>
}""",
}


@dataclass
class DeliberationEntry:
    verifier: str
    rebuttal: str
    updated_verdict: str
    updated_confidence: float
    changed: bool

    def to_dict(self) -> dict:
        return {
            "verifier": self.verifier,
            "rebuttal": self.rebuttal,
            "updated_verdict": self.updated_verdict,
            "updated_confidence": self.updated_confidence,
            "changed": self.changed,
        }


@dataclass
class DeliberationResult:
    sub_claim_id: str
    initial_verdicts: Dict[str, str]
    had_disagreement: bool
    deliberation: List[DeliberationEntry] = field(default_factory=list)
    final_verdicts: Dict[str, str] = field(default_factory=dict)
    final_confidences: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sub_claim_id": self.sub_claim_id,
            "initial_verdicts": self.initial_verdicts,
            "had_disagreement": self.had_disagreement,
            "deliberation": [d.to_dict() for d in self.deliberation],
            "final_verdicts": self.final_verdicts,
            "final_confidences": self.final_confidences,
        }


def _verdicts_agree(verdicts: Dict[str, str]) -> bool:
    return len(set(verdicts.values())) == 1


def _format_peer_verdicts(
    verifier_id: str,
    initial_results: Dict[str, VerifierResult],
    sub_claim_text: str,
    evidence_str: str,
) -> str:
    """Format peer verdicts for a given verifier's deliberation prompt."""
    lines = [
        f"Sub-claim: {sub_claim_text}",
        "",
        f"Your initial verdict ({verifier_id}): {initial_results[verifier_id].verdict}",
        f"Your reasoning: {initial_results[verifier_id].reasoning}",
        "",
        "Peer verdicts:",
    ]
    for vid, result in initial_results.items():
        if vid != verifier_id:
            lines.append(f"  {vid}: {result.verdict} (confidence: {result.confidence:.2f})")
            lines.append(f"    Reasoning: {result.reasoning}")
    lines.append("")
    lines.append(f"Evidence summary:\n{evidence_str[:800]}")
    return "\n".join(lines)


def _parse_deliberation_response(raw: str, verifier_id: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {
            "verifier": verifier_id,
            "rebuttal": "Parse error",
            "updated_verdict": "FALSE",
            "updated_confidence": 0.5,
            "changed": False,
        }


class DeliberationEngine:
    def __init__(self):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.VERIFIER_MODELS["v1"]  # same base model

    def deliberate(
        self,
        sub_claim_id: str,
        sub_claim_text: str,
        initial_results: Dict[str, VerifierResult],
        evidence: List[Dict[str, Any]],
    ) -> DeliberationResult:
        initial_verdicts = {vid: r.verdict for vid, r in initial_results.items()}
        had_disagreement = not _verdicts_agree(initial_verdicts)

        # Start final verdicts from initial
        final_verdicts = dict(initial_verdicts)
        final_confidences = {vid: r.confidence for vid, r in initial_results.items()}
        deliberation_entries: List[DeliberationEntry] = []

        if not had_disagreement:
            # No deliberation needed
            return DeliberationResult(
                sub_claim_id=sub_claim_id,
                initial_verdicts=initial_verdicts,
                had_disagreement=False,
                deliberation=[],
                final_verdicts=final_verdicts,
                final_confidences=final_confidences,
            )

        # Format evidence summary
        evidence_str = "\n\n".join(
            f"[{ev['source']}]: {ev['text'][:300]}" for ev in evidence
        )

        # Each verifier gets a chance to deliberate
        for verifier_id in ["v1", "v2", "v3"]:
            prompt = _format_peer_verdicts(
                verifier_id, initial_results, sub_claim_text, evidence_str
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=DELIBERATION_SYSTEM_PROMPTS[verifier_id],
                    temperature=config.VERIFIER_TEMPERATURES[verifier_id],
                    response_mime_type="application/json",
                ),
            )

            data = _parse_deliberation_response(response.text, verifier_id)

            updated_verdict = data.get("updated_verdict", initial_verdicts[verifier_id]).upper()
            changed = bool(data.get("changed", False))

            # Verify 'changed' is accurate
            if updated_verdict != initial_verdicts[verifier_id]:
                changed = True
            elif changed:
                changed = False  # override if verdict didn't actually change

            entry = DeliberationEntry(
                verifier=verifier_id,
                rebuttal=data.get("rebuttal", ""),
                updated_verdict=updated_verdict,
                updated_confidence=float(data.get("updated_confidence", final_confidences[verifier_id])),
                changed=changed,
            )
            deliberation_entries.append(entry)
            final_verdicts[verifier_id] = updated_verdict
            final_confidences[verifier_id] = entry.updated_confidence

        return DeliberationResult(
            sub_claim_id=sub_claim_id,
            initial_verdicts=initial_verdicts,
            had_disagreement=had_disagreement,
            deliberation=deliberation_entries,
            final_verdicts=final_verdicts,
            final_confidences=final_confidences,
        )