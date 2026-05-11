from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from tqdm.auto import tqdm

from src.data.schema import Document, QASample, RAGResult
from src.evaluation.faithfulness import FaithfulnessEvaluator
from src.evaluation.metrics import (
    compute_qa_metrics,
    context_precision,
    mrr_at_k,
    recall_at_k,
)
from src.evaluation.relevance import AnswerRelevanceEvaluator
from src.models.generator import AnswerGenerator
from src.models.reranker import CrossEncoderReranker, diversity_filter
from src.models.retriever import BM25Retriever, DenseRetriever, HybridRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: AnswerGenerator,
        corpus: List[Document],
    ) -> None:
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.hybrid = hybrid_retriever
        self.reranker = reranker
        self.generator = generator
        self.corpus = corpus

    def run(
        self,
        qa_samples: List[QASample],
        n_samples: Optional[int] = None,
        retrieval_pool: int = 150,
        rerank_top_k: int = 50,
        final_top_k: int = 5,
        use_diversity: bool = True,
        div_threshold: float = 0.55,
    ) -> List[RAGResult]:
        
        samples = qa_samples[:n_samples] if n_samples else qa_samples
        results: List[RAGResult] = []

        for sample in tqdm(samples, desc="RAG inference"):
            q = sample.question

            bm25_hits = self.bm25.retrieve(q, top_k=retrieval_pool)
            dense_hits = self.dense.retrieve(q, top_k=retrieval_pool)
            hybrid_hits = self.hybrid.retrieve(
                q, top_k=rerank_top_k, pool_size=retrieval_pool
            )
            reranked = self.reranker.rerank(
                q, hybrid_hits, self.corpus, top_k=rerank_top_k
            )

            if use_diversity:
                reranked_final = diversity_filter(
                    reranked,
                    self.corpus,
                    top_k=final_top_k,
                    sim_threshold=div_threshold,
                )
            else:
                reranked_final = reranked[:final_top_k]

            context_docs = [self.corpus[doc_id] for doc_id, _ in reranked_final]
            answer = self.generator.generate(q, context_docs)

            results.append(
                RAGResult(
                    qid=sample.qid,
                    question=q,
                    gold_answers=sample.answers,
                    gold_doc_id=sample.gold_doc_id,
                    gold_doc_ids=sample.gold_doc_ids,
                    bm25_pool_ids=[d for d, _ in bm25_hits],
                    dense_pool_ids=[d for d, _ in dense_hits],
                    hybrid_top_ids=[d for d, _ in hybrid_hits],
                    reranked_top_ids=[d for d, _ in reranked_final],
                    generated_answer=answer,
                )
            )

        logger.info("Pipeline complete. %d results generated.", len(results))
        return results


def run_evaluation(
    results: List[RAGResult],
    corpus: List[Document],
    faith_evaluator: Optional[FaithfulnessEvaluator] = None,
    relevance_evaluator: Optional[AnswerRelevanceEvaluator] = None,
    k_values: List[int] = (1, 3, 5),
) -> Dict[str, float]:
    
    metrics: Dict[str, float] = {}

    # Retrieval metrics
    retrieval_stages = [
        ("BM25", "bm25_pool_ids"),
        ("Dense", "dense_pool_ids"),
        ("Hybrid", "hybrid_top_ids"),
        ("Reranked", "reranked_top_ids"),
    ]
    for name, field in retrieval_stages:
        metrics[f"{name} MRR@5"] = mrr_at_k(results, field, k=5)
        for k in k_values:
            metrics[f"{name} Recall@{k}"] = recall_at_k(results, field, k=k)

    metrics["Context Precision@5 (Hybrid)"] = context_precision(
        results, corpus, "hybrid_top_ids", k=5
    )
    metrics["Context Precision@5 (Reranked)"] = context_precision(
        results, corpus, "reranked_top_ids", k=5
    )

    # Answer quality
    qa = compute_qa_metrics(results)
    metrics.update(qa)

    # RAGAS-style scores (optional)
    if faith_evaluator is not None:
        faith_scores = faith_evaluator.batch_score(results, corpus)
        metrics["Faithfulness (NLI)"] = float(np.mean(faith_scores))

    if relevance_evaluator is not None:
        rel_scores = relevance_evaluator.score_batch(results)
        metrics["Answer Relevance"] = float(np.mean(rel_scores))

    return metrics
