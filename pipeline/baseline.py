"""
Single-LLM baseline — no decomposition, no RAG, no multi-agent.
One prompt → one verdict.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from google import genai
from google.genai import types

import config

BASELINE_SYSTEM_PROMPT = """You are a fact-checker. Given a claim, assess its truthfulness based on your knowledge.

Output ONLY valid JSON (no markdown):
{
  "claim": "<the claim>",
  "verdict": "<TRUE|FALSE|PARTIALLY_TRUE|MISLEADING>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief reasoning>"
}"""


@dataclass
class BaselineResult:
    claim_id: str
    claim_text: str
    verdict: str
    confidence: float
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


class BaselineEvaluator:
    def __init__(self):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.BASELINE_MODEL

    def evaluate(self, claim_id: str, claim_text: str) -> BaselineResult:
        prompt = f'Fact-check this claim:\n\n"{claim_text}"'

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=BASELINE_SYSTEM_PROMPT,
                temperature=config.BASELINE_TEMPERATURE,
                response_mime_type="application/json",
            ),
        )

        raw = re.sub(r"^```(?:json)?\s*", "", response.text.strip())
        raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group()) if match else {
                "verdict": "FALSE", "confidence": 0.5, "reasoning": "Parse error"
            }

        return BaselineResult(
            claim_id=claim_id,
            claim_text=claim_text,
            verdict=data.get("verdict", "FALSE").upper(),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
        )