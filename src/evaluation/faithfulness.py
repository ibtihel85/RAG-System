"""
Faithfulness evaluation using NLI entailment scoring.

Model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
  - Trained on MNLI + FEVER (fact-checking) + ANLI (adversarial NLI).
  - FEVER training specifically improves calibration for short factual spans
    like SQuAD answers.
  - Expected P(entailment) for verbatim extracted spans: 0.55–0.85.

Scoring strategy
----------------
For each result:
  1. Concatenate top-N retrieved chunks as a single NLI premise.
  2. Also score each chunk individually.
  3. Final score = max(concat_score, individual_max).

This two-pronged approach handles both cases: answers supported by the
concatenated context, and answers supported by a single strong chunk.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.schema import Document, RAGResult

logger = logging.getLogger(__name__)


class FaithfulnessEvaluator:

    NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    def __init__(self, model_name: str | None = None, device: str = "cpu") -> None:
        model_name = model_name or self.NLI_MODEL
        self.device = device

        logger.info("Loading NLI model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval().to(device)

        id2label = self.model.config.id2label
        self.entailment_idx = next(
            k for k, v in id2label.items() if "entail" in v.lower()
        )
        logger.info(
            "NLI model loaded. Label map: %s | entailment_idx=%d",
            id2label,
            self.entailment_idx,
        )

    @torch.inference_mode()
    def _score_pairs(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 8,
    ) -> List[float]:
        
        if not pairs:
            return []

        all_scores: List[float] = []

        for start in range(0, len(pairs), batch_size):
            mini = pairs[start : start + batch_size]
            premises = [p for p, _ in mini]
            hypotheses = [h for _, h in mini]

            enc = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                logits = self.model(**enc).logits

            probs = F.softmax(logits.float(), dim=-1)
            batch_scores = probs[:, self.entailment_idx].cpu().tolist()
            all_scores.extend(batch_scores)

            del enc, logits, probs
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return all_scores

    def batch_score(
        self,
        results: List[RAGResult],
        corpus: List[Document],
        max_chars_per_doc: int = 600,
        top_n_concat: int = 3,
        nli_batch_size: int = 8,
    ) -> List[float]:
        
        concat_pairs: List[Tuple[str, str]] = []
        indiv_pairs: List[Tuple[str, str]] = []
        indiv_counts: List[int] = []
        valid_mask: List[bool] = []

        for r in results:
            ans = r.generated_answer.strip()
            if (
                not ans
                or ans.lower() in ("i don't know", "i don't know.")
                or not r.reranked_top_ids
            ):
                valid_mask.append(False)
                indiv_counts.append(0)
                continue

            valid_mask.append(True)

            top_texts = [
                corpus[d].text[:max_chars_per_doc]
                for d in r.reranked_top_ids[:top_n_concat]
            ]
            premise = " ".join(top_texts)
            concat_pairs.append((premise, ans))

            indiv_pairs.extend([(t, ans) for t in top_texts])
            indiv_counts.append(len(top_texts))

        concat_scores = self._score_pairs(concat_pairs, batch_size=nli_batch_size)
        indiv_scores = self._score_pairs(indiv_pairs, batch_size=nli_batch_size)

        final_scores: List[float] = []
        concat_iter = iter(concat_scores)
        indiv_offset = 0
        count_iter = iter(indiv_counts)

        for is_valid in valid_mask:
            if not is_valid:
                final_scores.append(0.0)
                next(count_iter, 0)
                continue

            c_score = next(concat_iter)
            n_chunks = next(count_iter)
            chunk_scores = indiv_scores[indiv_offset : indiv_offset + n_chunks]
            indiv_offset += n_chunks
            indiv_max = max(chunk_scores) if chunk_scores else 0.0
            final_scores.append(max(c_score, indiv_max))

        return final_scores
