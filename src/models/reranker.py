
from __future__ import annotations

import logging
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from src.data.schema import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diversity filter
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def diversity_filter(
    ranked: List[Tuple[int, float]],
    corpus: List[Document],
    top_k: int = 5,
    sim_threshold: float = 0.55,
) -> List[Tuple[int, float]]:
    
    kept: List[Tuple[int, float]] = []

    for doc_id, score in ranked:
        body_i = corpus[doc_id].body
        if all(
            _jaccard(body_i, corpus[kept_id].body) < sim_threshold
            for kept_id, _ in kept
        ):
            kept.append((doc_id, score))
        if len(kept) >= top_k:
            break

    # Greedy fill if filter was too aggressive
    if len(kept) < top_k:
        seen = {d for d, _ in kept}
        for doc_id, score in ranked:
            if doc_id not in seen:
                kept.append((doc_id, score))
            if len(kept) >= top_k:
                break

    return kept[:top_k]


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        device: str = "cpu",
    ) -> None:
        logger.info("Loading cross-encoder: %s", model_name)
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        logger.info("Cross-encoder loaded.")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float]],
        corpus: List[Document],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        
        if not candidates:
            return []

        pairs = [(query, corpus[doc_id].body) for doc_id, _ in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip([doc_id for doc_id, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
