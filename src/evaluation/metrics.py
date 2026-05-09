from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

import numpy as np
from rouge_score import rouge_scorer as rouge_lib

from src.data.schema import Document, RAGResult

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lowercase, strip articles, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Answer quality
# ---------------------------------------------------------------------------

def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 score between *pred* and *gold* after normalisation."""
    pred_tokens = normalise(pred).split()
    gold_tokens = normalise(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    """Return True if *pred* and *gold* are identical after normalisation."""
    return normalise(pred) == normalise(gold)


_rouge = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F-measure between *pred* and *gold*."""
    return _rouge.score(pred, gold)["rougeL"].fmeasure


def compute_qa_metrics(results: List[RAGResult]) -> Dict[str, float]:
    
    em_scores, f1_scores, rl_scores = [], [], []
    for r in results:
        pred = r.generated_answer
        best_em = max(int(exact_match(pred, g)) for g in r.gold_answers)
        best_f1 = max(token_f1(pred, g) for g in r.gold_answers)
        best_rl = max(rouge_l(pred, g) for g in r.gold_answers)
        em_scores.append(best_em)
        f1_scores.append(best_f1)
        rl_scores.append(best_rl)
    return {
        "Exact Match": float(np.mean(em_scores)),
        "Token F1": float(np.mean(f1_scores)),
        "ROUGE-L": float(np.mean(rl_scores)),
    }


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def _gold_set(r: RAGResult):
    return {r.gold_doc_id}


def recall_at_k(
    results: List[RAGResult],
    doc_ids_field: str,
    k: int = 5,
) -> float:
    hits = 0
    for r in results:
        top_ids = getattr(r, doc_ids_field)[:k]
        if set(top_ids) & _gold_set(r):
            hits += 1
    return hits / len(results) if results else 0.0


def mrr_at_k(
    results: List[RAGResult],
    doc_ids_field: str,
    k: int = 5,
) -> float:
    
    rr_sum = 0.0
    for r in results:
        top_ids = getattr(r, doc_ids_field)[:k]
        gold = _gold_set(r)
        for rank, doc_id in enumerate(top_ids, start=1):
            if doc_id in gold:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(results) if results else 0.0


def _body_contains_answer(doc_body: str, gold_answers: List[str]) -> bool:
    
    body_lower = doc_body.lower()
    for ans in gold_answers:
        ans_norm = ans.strip().lower()
        if not ans_norm:
            continue
        if ans_norm in body_lower:
            return True
        if len(ans_norm.split()) <= 3:
            pattern = r"\b" + re.escape(ans_norm) + r"\b"
            if re.search(pattern, body_lower):
                return True
        if token_f1(ans, doc_body) >= 0.45:
            return True
    return False


def context_precision(
    results: List[RAGResult],
    corpus: List[Document],
    doc_ids_field: str,
    k: int = 5,
) -> float:
    
    ap_scores = []
    for r in results:
        top_ids = getattr(r, doc_ids_field)[:k]
        if not top_ids:
            ap_scores.append(0.0)
            continue

        gold = _gold_set(r)
        rel_flags = [
            1
            if (
                doc_id in gold
                or _body_contains_answer(corpus[doc_id].body, r.gold_answers)
            )
            else 0
            for doc_id in top_ids
        ]

        n_relevant = sum(rel_flags)
        if n_relevant == 0:
            ap_scores.append(0.0)
            continue

        ap = 0.0
        running_hits = 0
        for j, rel in enumerate(rel_flags, start=1):
            if rel:
                running_hits += 1
                ap += running_hits / j
        ap_scores.append(ap / n_relevant)

    return float(np.mean(ap_scores))