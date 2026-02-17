"""
Stage 5 — Final Synthesis.
Aggregates all sub-claim verdicts into a final ruling for the original claim.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any

from google import genai
from google.genai import types

import config

SYNTHESIZER_SYSTEM_PROMPT = """You are the Synthesizer in a multi-agent fact-checking system.

You receive:
1. The original complex claim
2. All atomic sub-claims it was decomposed into
3. Retrieved evidence for each sub-claim
4. Verdicts from three independent verifiers after deliberation

Your task:
- Integrate all sub-claim verdicts into a single final verdict for the original claim
- The final verdict must be one of: TRUE, FALSE, PARTIALLY_TRUE, MISLEADING
- Provide detailed synthesis reasoning explaining how the sub-claim verdicts combine
- Note any limitations (missing evidence, uncertain sub-claims, etc.)
- Assign an overall confidence score

Verdict guidelines:
- TRUE: All material sub-claims are true and supported
- FALSE: One or more material sub-claims are clearly false, making the overall claim false
- PARTIALLY_TRUE: Some sub-claims are true, others false or unsupported
- MISLEADING: Sub-claims are technically true but the overall framing is deceptive

Output ONLY valid JSON (no markdown):
{
  "original_claim": "<claim text>",
  "final_verdict": "<TRUE|FALSE|PARTIALLY_TRUE|MISLEADING>",
  "confidence": <0.0-1.0>,
  "synthesis_reasoning": "<detailed paragraph-level reasoning>",
  "sub_claim_summary": [
    {"id": "<sc_id>", "text": "<sub-claim>", "consensus_verdict": "<verdict>", "agreement": "<agree|disagree>"}
  ],
  "limitations": "<known limitations of this analysis>"
}"""


@dataclass
class SynthesisResult:
    claim_id: str
    original_claim: str
    final_verdict: str
    confidence: float
    synthesis_reasoning: str
    sub_claim_summary: List[Dict[str, str]]
    limitations: str

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "original_claim": self.original_claim,
            "final_verdict": self.final_verdict,
            "confidence": self.confidence,
            "synthesis_reasoning": self.synthesis_reasoning,
            "sub_claim_summary": self.sub_claim_summary,
            "limitations": self.limitations,
        }


def _build_synthesis_prompt(
    claim_id: str,
    original_claim: str,
    sub_claims: List[Dict[str, Any]],
    evidence_map: Dict[str, List[Dict[str, Any]]],
    deliberation_results: List[Dict[str, Any]],
) -> str:
    sections = [
        f"ORIGINAL CLAIM (ID: {claim_id}):\n{original_claim}\n",
        "─" * 60,
        "SUB-CLAIMS AND VERDICTS:\n",
    ]

    for sc in sub_claims:
        sc_id = sc["id"]
        delib = next((d for d in deliberation_results if d["sub_claim_id"] == sc_id), None)

        sections.append(f"\nSub-claim [{sc_id}]: {sc['text']}")
        sections.append(f"  Type: {sc['type']}")

        # Evidence summary
        evidence = evidence_map.get(sc_id, [])
        if evidence:
            sections.append(f"  Evidence ({len(evidence)} passages):")
            for ev in evidence[:2]:
                sections.append(f"    [{ev['source']}]: {ev['text'][:200]}…")

        # Verifier verdicts
        if delib:
            sections.append(f"  Initial verdicts: {delib['initial_verdicts']}")
            sections.append(f"  Final verdicts: {delib['final_verdicts']}")
            sections.append(f"  Had disagreement: {delib['had_disagreement']}")

    return "\n".join(sections)


class ClaimSynthesizer:
    def __init__(self):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.SYNTHESIZER_MODEL

    def synthesize(
        self,
        claim_id: str,
        original_claim: str,
        sub_claims: List[Dict[str, Any]],
        evidence_map: Dict[str, List[Dict[str, Any]]],
        deliberation_results: List[Dict[str, Any]],
    ) -> SynthesisResult:
        prompt = _build_synthesis_prompt(
            claim_id, original_claim, sub_claims, evidence_map, deliberation_results
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYNTHESIZER_SYSTEM_PROMPT,
                temperature=config.SYNTHESIZER_TEMPERATURE,
                response_mime_type="application/json",
            ),
        )

        raw = re.sub(r"^```(?:json)?\s*", "", response.text.strip())
        raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = {
                    "final_verdict": "FALSE",
                    "confidence": 0.5,
                    "synthesis_reasoning": f"Parse error: {raw[:300]}",
                    "sub_claim_summary": [],
                    "limitations": "Synthesis parse error",
                }

        return SynthesisResult(
            claim_id=claim_id,
            original_claim=original_claim,
            final_verdict=data.get("final_verdict", "FALSE").upper(),
            confidence=float(data.get("confidence", 0.5)),
            synthesis_reasoning=data.get("synthesis_reasoning", ""),
            sub_claim_summary=data.get("sub_claim_summary", []),
            limitations=data.get("limitations", ""),
        )