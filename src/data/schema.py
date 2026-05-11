"""
Core data structures for the RAG pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class Document:
    doc_id: int
    title: str
    text: str
    body: str
    source_ctx: str = ""


@dataclass
class QASample:
    qid: str
    question: str
    answers: List[str]
    gold_context: str
    gold_doc_id: Optional[int]
    gold_doc_ids: Set[int] = field(default_factory=set)




@dataclass
class RAGResult:

    qid: str
    question: str
    gold_answers: List[str]
    gold_doc_id: Optional[int]
    gold_doc_ids: Set[int]

    # Retrieved doc ids at each stage
    bm25_pool_ids: List[int]
    dense_pool_ids: List[int]
    hybrid_top_ids: List[int]
    reranked_top_ids: List[int]

    # Final generated answer
    generated_answer: str
