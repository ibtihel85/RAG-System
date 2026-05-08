from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import faiss
import numpy as np
from nltk.corpus import stopwords as _nltk_stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from src.data.schema import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared stop-word set
# ---------------------------------------------------------------------------

_SW = set(_nltk_stopwords.words("english")) | {
    "many",
    "much",
    "old",
    "far",
    "long",
    "tall",
    "deep",
    "wide",
    "what",
    "when",
    "where",
    "who",
    "whom",
    "which",
    "how",
    "why",
    "did",
    "does",
    "do",
    "was",
    "were",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "by",
    "for",
}


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Retriever:
    """Sparse BM25 retriever over Document.body.

    Args:
        corpus: Flat list of Document objects forming the index.
    """

    _STOPWORDS = {
        "a", "an", "the", "is", "was", "are", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "in", "of", "to", "and", "for", "on", "at", "by", "with",
        "from", "this", "that", "these", "those", "it", "its",
    }

    def __init__(self, corpus: List[Document]) -> None:
        self.corpus = corpus
        logger.info("Tokenising %d docs for BM25 …", len(corpus))
        tokenised = [
            [
                t
                for t in word_tokenize(doc.body.lower())
                if t.isalpha() and t not in self._STOPWORDS
            ]
            for doc in tqdm(corpus, desc="BM25 tokenise")
        ]
        self.bm25 = BM25Okapi(tokenised)
        logger.info("BM25 index built over %d docs.", len(corpus))

    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Retrieve top-k documents for *query* using BM25 scores.

        Args:
            query: Raw query string.
            top_k: Number of results to return.

        Returns:
            List of (doc_id, score) tuples sorted by descending score.
        """
        tokens = [
            t
            for t in word_tokenize(query.lower())
            if t.isalpha() and t not in self._STOPWORDS
        ]
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Dense (BGE + FAISS)
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Dense retriever using BAAI/bge-base-en-v1.5 + FAISS IndexFlatIP.

    Documents are encoded without a prefix; queries use the BGE instruction
    prefix for asymmetric retrieval.

    Args:
        corpus:     Flat list of Document objects.
        model_name: HuggingFace model identifier.
        batch_size: Encoding batch size.
        device:     'cuda' or 'cpu'.
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        corpus: List[Document],
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self.corpus = corpus
        self.model_name = model_name
        self.device = device

        logger.info("Loading bi-encoder: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

        logger.info("Encoding %d chunks (dim=%d) …", len(corpus), self.dim)
        texts = [doc.text for doc in corpus]
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(self.dim)
        if hasattr(faiss, "StandardGpuResources") and device == "cuda":
            logger.info("Using FAISS GPU index.")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.index.add(self.embeddings)
        logger.info("FAISS index built: %d vectors.", self.index.ntotal)

    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Return top-k docs by cosine similarity (via inner product on L2-normed vecs).

        Args:
            query: Raw query string (prefix added internally).
            top_k: Number of results to return.

        Returns:
            List of (doc_id, score) tuples sorted by descending score.
        """
        prefixed = self.QUERY_PREFIX + query
        q_emb = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        scores, indices = self.index.search(q_emb, top_k)
        return [
            (int(indices[0][i]), float(scores[0][i]))
            for i in range(len(indices[0]))
            if indices[0][i] >= 0
        ]


# ---------------------------------------------------------------------------
# Query expansion helpers
# ---------------------------------------------------------------------------

def _content_tokens(text: str, max_tokens: int = 6) -> str:
    tokens = [
        t
        for t in word_tokenize(text.lower())
        if t.isalpha() and t not in _SW and len(t) > 2
    ]
    tokens = sorted(set(tokens), key=len, reverse=True)
    return " ".join(tokens[:max_tokens])


def _get_type_token(q_lower: str) -> str:
    if re.search(r"^why ", q_lower):
        return "cause"
    if re.search(r"^(what|which) (type|kind|form|sort|category)", q_lower):
        return "type"
    return ""


def multi_expand_query(question: str) -> Tuple[List[str], str]:
    """Generate three diversified BM25 query variants and one dense query.

    Variants:
      v1 – original question + minimal type token (broadest recall)
      v2 – content keywords only (no WH-words, no type token)
      v3 – WH-word-stripped question (alternative surface form)

    Returns:
        (bm25_queries, dense_query) where bm25_queries is a list of 3 strings.
    """
    q = question.strip()
    q_lower = q.lower()

    content_kw = _content_tokens(q, max_tokens=6)
    type_token = _get_type_token(q_lower)

    v1_suffix = " ".join(filter(None, [type_token, content_kw])).strip()
    v1 = f"{q} {v1_suffix}".strip() if v1_suffix else q

    v2 = content_kw if content_kw else q

    stripped = re.sub(
        r"^(what|which|who|whom|when|where|why|how\s+\w+)\s+", "", q_lower
    ).strip()
    v3 = stripped if stripped and stripped != q_lower else q

    return [v1, v2, v3], q  # dense query is always the raw question


# ---------------------------------------------------------------------------
# Hybrid (RRF)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Hybrid BM25 + Dense retriever using Reciprocal Rank Fusion (RRF).

    Three diversified BM25 variants are fused first (intra-BM25 RRF), then
    the result is fused with the dense ranked list (inter RRF).

    Args:
        bm25:         Initialised BM25Retriever.
        dense:        Initialised DenseRetriever.
        k_intra:      RRF constant for intra-BM25 fusion.
        k_inter:      RRF constant for BM25-dense fusion.
        bm25_weight:  Scale applied to BM25 RRF scores during inter-fusion.
        dense_weight: Scale applied to dense RRF scores during inter-fusion.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        k_intra: int = 60,
        k_inter: int = 10,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.6,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.k_intra = k_intra
        self.k_inter = k_inter
        self.w_bm25 = bm25_weight
        self.w_dense = dense_weight

    def _fuse_bm25_variants(
        self,
        bm25_queries: List[str],
        pool_size: int,
    ) -> List[Tuple[int, float]]:
        fused: Dict[int, float] = defaultdict(float)
        for q_variant in bm25_queries:
            hits = self.bm25.retrieve(q_variant, top_k=pool_size)
            for rank, (doc_id, _) in enumerate(hits, start=1):
                fused[doc_id] += 1.0 / (self.k_intra + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
        pool_size: int = 150,
    ) -> List[Tuple[int, float]]:
        """Retrieve and fuse BM25 + dense results via RRF.

        Args:
            query:     Raw query string.
            top_k:     Final number of results to return.
            pool_size: Number of candidates to pull from each retriever.

        Returns:
            List of (doc_id, rrf_score) tuples.
        """
        bm25_queries, dense_q = multi_expand_query(query)

        bm25_fused = self._fuse_bm25_variants(bm25_queries, pool_size=pool_size)
        dense_results = self.dense.retrieve(dense_q, top_k=pool_size)

        rrf_scores: Dict[int, float] = defaultdict(float)
        for rank, (doc_id, _) in enumerate(bm25_fused, start=1):
            rrf_scores[doc_id] += self.w_bm25 / (self.k_inter + rank)
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            rrf_scores[doc_id] += self.w_dense / (self.k_inter + rank)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]