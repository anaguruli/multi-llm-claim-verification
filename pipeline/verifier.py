"""
Stage 3 — Independent Verification.
Three Verifier LLMs each independently assess sub-claims against evidence.
Different system prompts and temperatures provide perspective diversity.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

from google import genai
from google.genai import types

import config

# ── System prompts give each verifier a distinct analytical persona ───────────

VERIFIER_SYSTEM_PROMPTS = {
    "v1": """You are Verifier-1, a rigorous and conservative fact-checker.

Your approach:
- Demand strong, explicit evidence before accepting any claim as true.
- Be skeptical of vague or indirect evidence.
- If evidence is ambiguous, lean toward FALSE or PARTIALLY_TRUE.
- Focus strictly on what the evidence says, ignoring prior knowledge.

For each sub-claim, output ONLY valid JSON (no markdown):
{
  "sub_claim_id": "<id>",
  "verdict": "<TRUE|FALSE|PARTIALLY_TRUE|MISLEADING>",
  "confidence": <0.0-1.0>,
  "reasoning": "<detailed reasoning referencing the evidence>",
  "evidence_sufficiency": "<sufficient|insufficient|contradictory>"
}""",

    "v2": """You are Verifier-2, a balanced and contextual fact-checker.

Your approach:
- Weigh evidence carefully, considering both what is stated and what context implies.
- Consider nuance — a claim may be partially true if some elements are correct.
- Look for both supporting and contradicting evidence.
- Consider the spirit of the claim, not just its literal wording.

For each sub-claim, output ONLY valid JSON (no markdown):
{
  "sub_claim_id": "<id>",
  "verdict": "<TRUE|FALSE|PARTIALLY_TRUE|MISLEADING>",
  "confidence": <0.0-1.0>,
  "reasoning": "<detailed reasoning referencing the evidence>",
  "evidence_sufficiency": "<sufficient|insufficient|contradictory>"
}""",

    "v3": """You are Verifier-3, a systematic and structured fact-checker.

Your approach:
- Break down the sub-claim into its component assertions.
- Verify each component against the evidence separately.
- Synthesise component-level verdicts into an overall verdict.
- Explicitly note any evidence gaps.

For each sub-claim, output ONLY valid JSON (no markdown):
{
  "sub_claim_id": "<id>",
  "verdict": "<TRUE|FALSE|PARTIALLY_TRUE|MISLEADING>",
  "confidence": <0.0-1.0>,
  "reasoning": "<detailed reasoning referencing the evidence>",
  "evidence_sufficiency": "<sufficient|insufficient|contradictory>"
}""",
}


@dataclass
class VerifierResult:
    verifier_id: str
    sub_claim_id: str
    verdict: str
    confidence: float
    reasoning: str
    evidence_sufficiency: str

    def to_dict(self) -> dict:
        return {
            "verifier_id": self.verifier_id,
            "sub_claim_id": self.sub_claim_id,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence_sufficiency": self.evidence_sufficiency,
        }


def _format_evidence(evidence_list: List[Dict[str, Any]]) -> str:
    """Format retrieved evidence passages into a readable string."""
    parts = []
    for i, ev in enumerate(evidence_list, 1):
        parts.append(
            f"[Evidence {i} | Source: {ev['source']} | Score: {ev['relevance_score']:.3f}]\n"
            f"{ev['text']}"
        )
    return "\n\n".join(parts)


def _parse_verifier_response(raw: str, verifier_id: str, sub_claim_id: str) -> dict:
    """Parse JSON from verifier response, with fallback."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        # Fallback
        return {
            "sub_claim_id": sub_claim_id,
            "verdict": "FALSE",
            "confidence": 0.5,
            "reasoning": f"Parse error for {verifier_id}: {raw[:200]}",
            "evidence_sufficiency": "insufficient",
        }


class IndependentVerifier:
    """A single verifier agent."""

    def __init__(self, verifier_id: str):
        self.verifier_id = verifier_id
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.VERIFIER_MODELS[verifier_id]
        self.temperature = config.VERIFIER_TEMPERATURES[verifier_id]
        self.system_prompt = VERIFIER_SYSTEM_PROMPTS[verifier_id]

    def verify(
        self,
        sub_claim_id: str,
        sub_claim_text: str,
        evidence: List[Dict[str, Any]],
    ) -> VerifierResult:
        evidence_str = _format_evidence(evidence)
        prompt = (
            f"Sub-claim ID: {sub_claim_id}\n"
            f"Sub-claim: {sub_claim_text}\n\n"
            f"Retrieved Evidence:\n{evidence_str}\n\n"
            f"Assess whether the sub-claim is supported by the evidence."
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                response_mime_type="application/json",
            ),
        )

        data = _parse_verifier_response(response.text, self.verifier_id, sub_claim_id)

        return VerifierResult(
            verifier_id=self.verifier_id,
            sub_claim_id=sub_claim_id,
            verdict=data.get("verdict", "FALSE").upper(),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            evidence_sufficiency=data.get("evidence_sufficiency", "insufficient"),
        )


class VerifierPanel:
    """Manages all three verifiers and runs them in parallel (sequential here for API limits)."""

    def __init__(self):
        self.verifiers = {
            vid: IndependentVerifier(vid) for vid in config.VERIFIER_MODELS
        }

    def verify_sub_claim(
        self,
        sub_claim_id: str,
        sub_claim_text: str,
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, VerifierResult]:
        results = {}
        for vid, verifier in self.verifiers.items():
            result = verifier.verify(sub_claim_id, sub_claim_text, evidence)
            results[vid] = result
        return results