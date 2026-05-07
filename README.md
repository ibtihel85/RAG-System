# RAG-NLI Pipeline

A modular Retrieval-Augmented Generation (RAG) system evaluated with NLI-based faithfulness scoring, built on SQuAD v2. The project implements a full pipeline from document chunking through hybrid retrieval, cross-encoder reranking, LLM generation, and RAGAS-style evaluation — with careful attention to the failure modes that affect each stage.

---

## Problem Statement

Standard RAG pipelines often suffer from three interacting failure modes:

1. **Retrieval failures** — the relevant chunk is never retrieved (low recall).
2. **Reranking failures** — the right chunk is retrieved but demoted by the cross-encoder.
3. **Faithfulness failures** — the LLM generates an answer not grounded in the retrieved context.

This project addresses all three systematically, using SQuAD v2 as the evaluation benchmark because its answer spans are short and verifiable, making it a clean testbed for extractive RAG.

---

## Approach

### Chunking
Raw SQuAD paragraphs (150–400 words) are split into overlapping sentence-window chunks using `window=4, stride=2`. This reduces inter-chunk Jaccard similarity from ~0.67 (with `window=3, stride=1`) to ~0.50, which meaningfully reduces redundancy in the retrieval pool.

Each chunk stores two text fields:
- `body` — raw chunk text (used for BM25 scoring and evaluation)
- `text` — title-prefixed chunk (used for dense encoding and cross-encoder)

This separation prevents title tokens from inflating BM25 scores and polluting faithfulness evaluation.

### Retrieval (Hybrid BM25 + Dense)
Three diversified BM25 query variants are generated per question (original + type token, content keywords only, WH-stripped form) and fused via intra-BM25 Reciprocal Rank Fusion (RRF, k=60). This result is then fused with a BGE-base dense retrieval pass via inter-RRF (k=10). Dense and BM25 are weighted 1.6:1.0 to reflect the BEIR benchmark gap (~63 vs ~52 NDCG@10).

### Reranking
A `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder reranks the top-50 hybrid candidates. Crucially, the cross-encoder receives `doc.body` (not `doc.text`), since MS-MARCO training data does not include topic prefixes. A Jaccard-based diversity filter (threshold 0.55) then removes near-duplicate chunks before the final top-5 is selected.

### Generation
`meta-llama/Llama-3.1-8B-Instruct` is prompted to perform **extractive** QA — copy a verbatim 1–8 word span from the passages. This design maximises faithfulness because the answer is a substring of retrieved context. The model is loaded in 4-bit NF4 quantisation (bitsandbytes) for Kaggle/Colab compatibility.

### Evaluation (RAGAS-style)
| Metric | Method |
|---|---|
| Recall@K | Gold chunk in top-K retrieved |
| MRR@5 | Mean Reciprocal Rank of first gold hit |
| Context Precision@5 | Average Precision@K (rank-weighted) |
| Exact Match | Normalised string equality |
| Token F1 | Token-level overlap (SQuAD eval script) |
| ROUGE-L | Longest common subsequence F-measure |
| Faithfulness | P(entailment) via DeBERTa-v3-base-MNLI-FEVER-ANLI |
| Answer Relevance | Answer-type heuristics (WH-word + length) |

---

## Results (300 samples, SQuAD v2 validation)

| Metric | Score |
|---|---|
| BM25 Recall@5 | ~0.62 |
| Dense Recall@5 | ~0.74 |
| Hybrid Recall@5 | ~0.80 |
| Hybrid+Rerank Recall@5 | ~0.82 |
| Context Precision@5 (Reranked) | ~0.71 |
| Exact Match | ~0.38 |
| Token F1 | ~0.52 |
| ROUGE-L | ~0.49 |
| Faithfulness (NLI) | ~0.63 |
| Answer Relevance | ~0.67 |

> Results vary slightly across runs due to dataset sampling. Scores reflect the `validation` split with `max_corpus_docs=6000, max_qa_samples=400`.

---

## Tech Stack

- **Retrieval**: `rank-bm25`, `BAAI/bge-base-en-v1.5`, `faiss-cpu`
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers)
- **Generation**: `meta-llama/Llama-3.1-8B-Instruct` (4-bit via bitsandbytes)
- **NLI Evaluation**: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- **Dataset**: `rajpurkar/squad_v2` (HuggingFace Datasets)
- **Testing**: pytest

---

## Project Structure

```
rag-nli-pipeline/
├── src/
│   ├── data/
│   │   ├── schema.py          # Document, QASample, RAGResult dataclasses
│   │   └── loader.py          # SQuAD loading, chunking, corpus construction
│   ├── models/
│   │   ├── retriever.py       # BM25Retriever, DenseRetriever, HybridRetriever
│   │   ├── reranker.py        # CrossEncoderReranker, diversity_filter
│   │   └── generator.py       # AnswerGenerator (Llama 4-bit)
│   ├── evaluation/
│   │   ├── metrics.py         # Recall@K, MRR, context precision, EM, F1, ROUGE
│   │   ├── faithfulness.py    # NLI-based faithfulness scorer
│   │   └── relevance.py       # Answer relevance heuristics
│   ├── utils/
│   │   ├── seed.py            # set_seed, get_device
│   │   └── logging_config.py  # setup_logging
│   └── pipeline.py            # RAGPipeline, run_evaluation
├── tests/
│   ├── test_data.py
│   ├── test_metrics.py
│   ├── test_retriever.py
│   └── test_relevance.py
├── configs/
│   ├── default.yaml           # Full run config
│   └── fast_dev.yaml          # Quick dev config (no LLM, small corpus)
├── main.py                    # CLI entry point
├── requirements.txt
└── .gitignore
```

---

## How to Run

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download NLTK data (one-time):
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### 2. Set your HuggingFace token (required for Llama)

```bash
export HF_TOKEN=hf_your_token_here
```

You must have access to `meta-llama/Llama-3.1-8B-Instruct` on HuggingFace.

### 3. Run the pipeline

**Full run (retrieval + reranking + generation + evaluation):**
```bash
python main.py
```

**Retrieval only (no LLM — fast, CPU-friendly):**
```bash
python main.py --config configs/fast_dev.yaml --no-generate
```

**Custom sample count:**
```bash
python main.py --n-samples 100 --no-faithfulness
```

Results are printed to the terminal and saved as `results.json`.

### 4. Run tests

```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Design Decisions

**Why sentence-window chunking over fixed-token chunking?**  
Sentence boundaries are semantically cleaner split points. Fixed-token chunking can cut mid-sentence, fragmenting the very span the retriever needs to find. The window=4/stride=2 combination keeps chunks short enough for precision while ensuring spans near boundaries appear in at least one chunk.

**Why BGE-base over all-MiniLM?**  
BGE-base-en-v1.5 scores ~63 NDCG@10 on BEIR benchmarks versus ~52 for all-MiniLM-L6-v2. The gap is consistent across domains. BGE also requires the asymmetric query prefix, which is easy to miss but important for retrieval quality.

**Why DeBERTa-MNLI-FEVER over nli-deberta-v3-small?**  
The small model was trained only on SNLI/MultiNLI and is poorly calibrated for short extractive answers — P(entailment) for a verbatim span like "1945" was near 0.05. The FEVER training in the larger model explicitly teaches fact-grounded short-form entailment.

**Why heuristic answer relevance instead of embedding similarity?**  
Cosine similarity between a short answer ("1945") and a long question string is systematically low in dense embedding space regardless of correctness. The WH-word type-match heuristic is less elegant but more reliable for SQuAD-style QA.

---

## Limitations

- The generator requires a CUDA GPU with ≥12 GB VRAM for comfortable 4-bit inference. CPU inference is possible but very slow.
- Answer Relevance is a heuristic, not a trained model. It works well for SQuAD-style factoid questions but would need revision for open-domain or multi-hop QA.
- The pipeline evaluates on SQuAD v2 validation only. Generalisation to other domains (medical, legal, technical) has not been tested.
- Retrieval pool sizes (150 candidates) and diversity thresholds (0.55) were tuned on this specific dataset and may need adjustment for other corpora.
