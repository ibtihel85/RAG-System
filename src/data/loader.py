from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from datasets import load_dataset
from nltk.tokenize import sent_tokenize

from src.data.schema import Document, QASample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_paragraph(
    text: str,
    title: str,
    base_id: int,
    window: int = 4,
    stride: int = 2,
) -> List[Document]:
    """Split a paragraph into overlapping sentence-window chunks.

    Args:
        text:    Raw paragraph string.
        title:   Article title prepended to doc.text (not doc.body).
        base_id: Starting doc_id for the first chunk of this paragraph.
        window:  Number of sentences per chunk.
        stride:  Sentences to advance between consecutive chunks.

    Returns:
        List of Document objects.  Short paragraphs (≤ window sentences) are
        returned as a single chunk.
    """
    sentences = sent_tokenize(text)

    if len(sentences) <= window:
        body = text
        display_text = f"{title}: {body}" if title else body
        return [
            Document(
                doc_id=base_id,
                title=title,
                text=display_text,
                body=body,
                source_ctx=text,
            )
        ]

    chunks: List[Document] = []
    local_id = 0
    i = 0

    while i < len(sentences):
        chunk_sents = sentences[i : i + window]
        body = " ".join(chunk_sents).strip()

        if len(body.split()) >= 10:
            display_text = f"{title}: {body}" if title else body
            chunks.append(
                Document(
                    doc_id=base_id + local_id,
                    title=title,
                    text=display_text,
                    body=body,
                    source_ctx=text,
                )
            )
            local_id += 1

        i += stride

    # Fallback: paragraph was all very short sentences
    if not chunks:
        return [
            Document(
                doc_id=base_id,
                title=title,
                text=f"{title}: {text}" if title else text,
                body=text,
                source_ctx=text,
            )
        ]

    return chunks


# ---------------------------------------------------------------------------
# Corpus + QA construction
# ---------------------------------------------------------------------------

def build_corpus_and_qa(
    dataset,
    max_corpus_docs: int = 6000,
    max_qa_samples: int = 400,
    answerable_only: bool = True,
    chunk_window: int = 4,
    chunk_stride: int = 2,
) -> Tuple[List[Document], List[QASample]]:
    """Build a chunked corpus and aligned QA pairs from a SQuAD split.

    Args:
        dataset:         A HuggingFace Dataset object (train or validation split).
        max_corpus_docs: Hard cap on total corpus chunks.
        max_qa_samples:  Hard cap on QA samples collected.
        answerable_only: Skip unanswerable SQuAD v2 questions if True.
        chunk_window:    Sentence window size passed to ``chunk_paragraph``.
        chunk_stride:    Stride passed to ``chunk_paragraph``.

    Returns:
        corpus:     Flat list of Document chunks.
        qa_pairs:   List of QASample objects whose gold_doc_ids index into corpus.
    """
    context_to_chunks: Dict[str, List[Document]] = {}
    corpus: List[Document] = []
    qa_pairs: List[QASample] = []

    for item in dataset:
        ctx = item["context"].strip()
        title = item["title"].strip()
        answers_list: List[str] = item["answers"]["text"]

        if answerable_only and not answers_list:
            continue

        # ── Build corpus chunks for this context (deduplicated) ──────────────
        if ctx not in context_to_chunks:
            if len(corpus) >= max_corpus_docs:
                continue

            new_chunks = chunk_paragraph(
                ctx,
                title,
                base_id=len(corpus),
                window=chunk_window,
                stride=chunk_stride,
            )
            available = max_corpus_docs - len(corpus)
            new_chunks = new_chunks[:available]
            if not new_chunks:
                continue

            # Assign globally unique doc_ids
            for j, chunk in enumerate(new_chunks):
                chunk.doc_id = len(corpus) + j

            context_to_chunks[ctx] = new_chunks
            corpus.extend(new_chunks)

        if len(qa_pairs) >= max_qa_samples:
            continue

        chunks_for_ctx = context_to_chunks[ctx]

        # ── Find which chunks contain each gold answer (in doc.body) ─────────
        gold_doc_ids: Set[int] = set()
        gold_doc_id: Optional[int] = None

        for ans in answers_list:
            ans_lower = ans.strip().lower()
            if not ans_lower:
                continue
            for chunk in chunks_for_ctx:
                if ans_lower in chunk.body.lower():
                    gold_doc_ids.add(chunk.doc_id)
                    if gold_doc_id is None:
                        gold_doc_id = chunk.doc_id

        # Fallback: pin to first chunk if no answer span matched
        if gold_doc_id is None:
            gold_doc_id = chunks_for_ctx[0].doc_id
            gold_doc_ids.add(gold_doc_id)

        qa_pairs.append(
            QASample(
                qid=item["id"],
                question=item["question"].strip(),
                answers=list(set(answers_list)),
                gold_context=ctx,
                gold_doc_id=gold_doc_id,
                gold_doc_ids=gold_doc_ids,
            )
        )

    logger.info(
        "Built corpus of %d chunks and %d QA pairs.", len(corpus), len(qa_pairs)
    )
    return corpus, qa_pairs


def load_squad_splits(trust_remote_code: bool = True):
    """Load SQuAD v2 train/validation splits from HuggingFace Hub.

    Returns:
        Tuple of (train_data, val_data) HuggingFace Dataset objects.
    """
    logger.info("Loading SQuAD v2 from HuggingFace Datasets …")
    squad = load_dataset("rajpurkar/squad_v2", trust_remote_code=trust_remote_code)
    return squad["train"], squad["validation"]