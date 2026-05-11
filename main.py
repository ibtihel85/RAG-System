

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

from src.data.loader import build_corpus_and_qa, load_squad_splits
from src.evaluation.faithfulness import FaithfulnessEvaluator
from src.evaluation.relevance import AnswerRelevanceEvaluator
from src.models.reranker import CrossEncoderReranker
from src.models.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from src.pipeline import RAGPipeline, run_evaluation
from src.utils import get_device, set_seed, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the RAG-NLI pipeline on SQuAD v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override pipeline.n_eval_samples from the config.",
    )
    p.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip LLM generation; only run retrieval + reranking.",
    )
    p.add_argument(
        "--no-faithfulness",
        action="store_true",
        help="Skip NLI faithfulness evaluation (faster on CPU).",
    )
    p.add_argument(
        "--output",
        default="results.json",
        help="Path to write evaluation metrics JSON.",
    )
    return p


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)

    # Setup
    setup_logging(getattr(logging, cfg["misc"].get("log_level", "INFO")))
    set_seed(cfg["misc"].get("seed", 42))
    device = get_device()
    logger.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info("Loading dataset …")
    _, val_data = load_squad_splits()
    d = cfg["data"]
    corpus, qa_samples = build_corpus_and_qa(
        val_data if d["split"] == "validation" else None,
        max_corpus_docs=d["max_corpus_docs"],
        max_qa_samples=d["max_qa_samples"],
        answerable_only=d["answerable_only"],
        chunk_window=d["chunk_window"],
        chunk_stride=d["chunk_stride"],
    )
    logger.info("Corpus: %d chunks | QA: %d samples", len(corpus), len(qa_samples))

    # ── Retrieval ─────────────────────────────────────────────────────────
    logger.info("Building BM25 index …")
    bm25_ret = BM25Retriever(corpus)

    logger.info("Building dense index …")
    dense_cfg = cfg["retrieval"]["dense"]
    dense_ret = DenseRetriever(
        corpus,
        model_name=dense_cfg["model_name"],
        batch_size=dense_cfg["batch_size"],
        device=device,
    )

    hybrid_cfg = cfg["retrieval"]["hybrid"]
    hybrid_ret = HybridRetriever(
        bm25=bm25_ret,
        dense=dense_ret,
        k_intra=hybrid_cfg["k_intra"],
        k_inter=hybrid_cfg["k_inter"],
        bm25_weight=hybrid_cfg["bm25_weight"],
        dense_weight=hybrid_cfg["dense_weight"],
    )

    # ── Reranker ─────────────────────────────────────────────────────────
    reranker_cfg = cfg["reranker"]
    reranker = CrossEncoderReranker(
        model_name=reranker_cfg["model_name"],
        max_length=reranker_cfg["max_length"],
        device=device,
    )

    # ── Generator ────────────────────────────────────────────────────────
    generator = None
    if not args.no_generate:
        from src.models.generator import AnswerGenerator

        gen_cfg = cfg["generator"]
        hf_token = gen_cfg.get("hf_token") or os.environ.get("HF_TOKEN")
        generator = AnswerGenerator(
            model_name=gen_cfg["model_name"],
            hf_token=hf_token,
            device=device,
        )

    # ── Pipeline run ─────────────────────────────────────────────────────
    pipe_cfg = cfg["pipeline"]
    n_samples = args.n_samples or pipe_cfg["n_eval_samples"]

    if generator is not None:
        pipeline = RAGPipeline(
            bm25_retriever=bm25_ret,
            dense_retriever=dense_ret,
            hybrid_retriever=hybrid_ret,
            reranker=reranker,
            generator=generator,
            corpus=corpus,
        )
        results = pipeline.run(
            qa_samples,
            n_samples=n_samples,
            retrieval_pool=pipe_cfg["retrieval_pool"],
            rerank_top_k=pipe_cfg["rerank_top_k"],
            final_top_k=pipe_cfg["final_top_k"],
            use_diversity=pipe_cfg["use_diversity"],
            div_threshold=pipe_cfg["div_threshold"],
        )
    else:
        # Retrieval-only mode: build dummy RAGResult objects
        from src.data.schema import RAGResult
        from src.models.reranker import diversity_filter

        logger.info("Running retrieval-only mode (no generation) …")
        from tqdm.auto import tqdm

        results = []
        for sample in tqdm(qa_samples[:n_samples], desc="Retrieval"):
            q = sample.question
            pool = pipe_cfg["retrieval_pool"]
            bm25_hits = bm25_ret.retrieve(q, top_k=pool)
            dense_hits = dense_ret.retrieve(q, top_k=pool)
            hybrid_hits = hybrid_ret.retrieve(q, top_k=pipe_cfg["rerank_top_k"], pool_size=pool)
            reranked = reranker.rerank(q, hybrid_hits, corpus, top_k=pipe_cfg["rerank_top_k"])
            reranked_final = diversity_filter(reranked, corpus, top_k=pipe_cfg["final_top_k"])
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
                    generated_answer="[skipped]",
                )
            )

    # ── Evaluation ───────────────────────────────────────────────────────
    eval_cfg = cfg.get("evaluation", {})

    faith_eval = None
    if eval_cfg.get("faithfulness", True) and not args.no_faithfulness and generator is not None:
        faith_eval = FaithfulnessEvaluator(
            model_name=eval_cfg.get("nli_model"),
            device=device,
        )

    rel_eval = AnswerRelevanceEvaluator() if eval_cfg.get("relevance", True) else None

    metrics = run_evaluation(
        results,
        corpus,
        faith_evaluator=faith_eval,
        relevance_evaluator=rel_eval,
        k_values=eval_cfg.get("k_values", [1, 3, 5]),
    )

    # ── Print + save ─────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Evaluation Summary")
    print("=" * 62)
    for name, val in metrics.items():
        print(f"  {name:<44}: {val:.4f}")
    print("=" * 62 + "\n")

    output_path = Path(args.output)
    output_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved to %s", output_path)


if __name__ == "__main__":
    main()
