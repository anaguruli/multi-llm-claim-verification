"""
RAG-based evidence retriever.
Chunks documents, embeds them with SentenceTransformers, builds a FAISS index,
and retrieves top-k passages for each sub-claim.
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config


@dataclass
class EvidencePassage:
    text: str
    source: str
    chunk_id: int
    relevance_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    sub_claim_id: str
    sub_claim_text: str
    retrieved_evidence: List[EvidencePassage] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sub_claim_id": self.sub_claim_id,
            "sub_claim_text": self.sub_claim_text,
            "retrieved_evidence": [e.to_dict() for e in self.retrieved_evidence],
        }


class RAGRetriever:
    """Chunk → embed → FAISS index → retrieve."""

    def __init__(
        self,
        kb_dir: str = config.KB_DIR,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        top_k: int = config.TOP_K_RETRIEVAL,
        embedding_model: str = config.EMBEDDING_MODEL,
    ):
        self.kb_dir = kb_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        print("[Retriever] Loading embedding model …")
        self.encoder = SentenceTransformer(embedding_model)

        self.chunks: List[str] = []
        self.sources: List[str] = []
        self.chunk_ids: List[int] = []
        self.index: faiss.IndexFlatIP | None = None

        self._build_index()

    # ── Index Construction ────────────────────────────────────────────────────

    def _tokenise(self, text: str) -> List[str]:
        """Split text into word tokens."""
        return re.split(r"\s+", text.strip())

    def _chunk_text(self, text: str, source: str) -> List[tuple[str, str]]:
        """
        Sliding-window word-level chunking.
        Returns list of (chunk_text, source).
        """
        words = self._tokenise(text)
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append((chunk, source))
            if end == len(words):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _build_index(self) -> None:
        """Load all KB documents, chunk them, embed, and build FAISS index."""
        print("[Retriever] Building FAISS index …")
        raw_chunks: List[tuple[str, str]] = []

        for fname in os.listdir(self.kb_dir):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(self.kb_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            doc_chunks = self._chunk_text(text, fname)
            raw_chunks.extend(doc_chunks)

        if not raw_chunks:
            raise ValueError(f"No .txt files found in '{self.kb_dir}'.")

        self.chunks = [c[0] for c in raw_chunks]
        self.sources = [c[1] for c in raw_chunks]
        self.chunk_ids = list(range(len(self.chunks)))

        # Embed all chunks
        embeddings = self.encoder.encode(
            self.chunks,
            show_progress_bar=True,
            normalize_embeddings=True,  # enables cosine via inner product
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # FAISS inner-product index (== cosine similarity for normalised vecs)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print(f"[Retriever] Index built: {len(self.chunks)} chunks from {self.kb_dir}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, sub_claim_id: str, sub_claim_text: str) -> RetrievalResult:
        """Retrieve top-k evidence passages for a single sub-claim."""
        query_vec = self.encoder.encode(
            [sub_claim_text], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, self.top_k)

        passages: List[EvidencePassage] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            passages.append(
                EvidencePassage(
                    text=self.chunks[idx],
                    source=self.sources[idx],
                    chunk_id=int(idx),
                    relevance_score=float(round(score, 4)),
                )
            )

        return RetrievalResult(
            sub_claim_id=sub_claim_id,
            sub_claim_text=sub_claim_text,
            retrieved_evidence=passages,
        )

    def retrieve_batch(
        self, sub_claims: List[Dict[str, str]]
    ) -> List[RetrievalResult]:
        """Retrieve evidence for a list of sub-claim dicts with 'id' and 'text'."""
        return [self.retrieve(sc["id"], sc["text"]) for sc in sub_claims]