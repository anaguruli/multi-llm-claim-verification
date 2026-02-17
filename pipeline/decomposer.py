"""
Claim Decomposer — Stage 1.
Uses Gemini to break a complex claim into atomic, independently verifiable sub-claims.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List

from google import genai
from google.genai import types

import config

DECOMPOSER_SYSTEM_PROMPT = """You are an expert claim decomposer for a fact-checking system.

Your task is to break down a complex claim into atomic sub-claims — the smallest independently verifiable assertions.

Rules:
1. Each sub-claim must be a single, standalone factual assertion.
2. Sub-claims should be exhaustive — together they must cover all verifiable content of the original claim.
3. Do NOT add information not present in the original claim.
4. Classify each sub-claim by type: factual_date, factual_number, causal_relationship, existence_claim, comparative_claim, or general_factual.
5. Output ONLY valid JSON, no markdown fences.

Output format:
{
  "original_claim": "<the original claim>",
  "sub_claims": [
    {
      "id": "sc_1",
      "text": "<atomic sub-claim>",
      "type": "<claim_type>"
    }
  ]
}"""


@dataclass
class SubClaim:
    id: str
    text: str
    type: str

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "type": self.type}


@dataclass
class DecompositionResult:
    original_claim: str
    claim_id: str
    sub_claims: List[SubClaim] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "original_claim": self.original_claim,
            "sub_claims": [sc.to_dict() for sc in self.sub_claims],
        }


class ClaimDecomposer:
    def __init__(self):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.DECOMPOSER_MODEL

    def decompose(self, claim_id: str, claim_text: str) -> DecompositionResult:
        """Decompose a single claim into atomic sub-claims."""
        prompt = f'Decompose the following claim:\n\n"{claim_text}"'

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=DECOMPOSER_SYSTEM_PROMPT,
                temperature=config.DECOMPOSER_TEMPERATURE,
                response_mime_type="application/json",
            ),
        )

        raw = response.text.strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)

        sub_claims = [
            SubClaim(id=f"{claim_id}_{sc['id']}", text=sc["text"], type=sc["type"])
            for sc in data.get("sub_claims", [])
        ]

        return DecompositionResult(
            original_claim=claim_text,
            claim_id=claim_id,
            sub_claims=sub_claims,
        )