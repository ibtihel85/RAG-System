from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

from src.data.schema import Document

logger = logging.getLogger(__name__)


class BM25Retriever:
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
        
        tokens = [
            t
            for t in word_tokenize(query.lower())
            if t.isalpha() and t not in self._STOPWORDS
        ]
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]
