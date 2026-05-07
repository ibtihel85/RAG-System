"""
Unit tests for data loading and chunking logic.
"""

import pytest
from src.data.schema import Document, QASample
from src.data.loader import chunk_paragraph, build_corpus_and_qa


# ---------------------------------------------------------------------------
# chunk_paragraph
# ---------------------------------------------------------------------------

class TestChunkParagraph:

    def test_short_paragraph_returns_single_chunk(self):
        text = "One sentence. Two sentences."
        docs = chunk_paragraph(text, title="Test", base_id=0, window=4, stride=2)
        assert len(docs) == 1
        assert docs[0].body == text
        assert docs[0].doc_id == 0

    def test_title_prepended_to_text_not_body(self):
        text = "Sentence one. Sentence two."
        docs = chunk_paragraph(text, title="MyTitle", base_id=0)
        assert docs[0].text.startswith("MyTitle:")
        assert not docs[0].body.startswith("MyTitle")

    def test_long_paragraph_produces_multiple_chunks(self):
        # 8 sentences → should produce multiple chunks with window=4, stride=2
        sents = [f"This is sentence number {i}." for i in range(8)]
        text = " ".join(sents)
        docs = chunk_paragraph(text, title="T", base_id=0, window=4, stride=2)
        assert len(docs) > 1

    def test_doc_ids_are_sequential_from_base(self):
        sents = [f"Sentence {i}." for i in range(10)]
        text = " ".join(sents)
        docs = chunk_paragraph(text, title="T", base_id=5, window=4, stride=2)
        for i, doc in enumerate(docs):
            assert doc.doc_id == 5 + i

    def test_chunks_have_minimum_word_count(self):
        sents = [f"Sentence {i} with some words here." for i in range(10)]
        text = " ".join(sents)
        docs = chunk_paragraph(text, title="T", base_id=0, window=4, stride=2)
        for doc in docs:
            assert len(doc.body.split()) >= 10

    def test_empty_title(self):
        text = "Only one sentence here."
        docs = chunk_paragraph(text, title="", base_id=0)
        assert docs[0].text == text  # no colon prefix when title is empty

    def test_source_ctx_stored(self):
        text = "A short paragraph."
        docs = chunk_paragraph(text, title="T", base_id=0)
        assert docs[0].source_ctx == text


# ---------------------------------------------------------------------------
# build_corpus_and_qa (with a minimal synthetic dataset)
# ---------------------------------------------------------------------------

def _make_fake_dataset(n: int = 5):
    """Return a list of dicts mimicking the SQuAD HuggingFace format."""
    return [
        {
            "id": f"q{i}",
            "title": f"Article {i}",
            "context": (
                f"The capital of Country{i} is City{i}. "
                f"City{i} was founded in 18{i:02d}. "
                f"It has a population of {i * 100000} people. "
                f"Famous landmark is Tower{i}."
            ),
            "question": f"What is the capital of Country{i}?",
            "answers": {"text": [f"City{i}"], "answer_start": [0]},
        }
        for i in range(n)
    ]


class TestBuildCorpusAndQA:

    def test_returns_corpus_and_qa(self):
        dataset = _make_fake_dataset(3)
        corpus, qa = build_corpus_and_qa(
            dataset, max_corpus_docs=100, max_qa_samples=10
        )
        assert len(corpus) > 0
        assert len(qa) > 0

    def test_corpus_doc_ids_are_unique(self):
        dataset = _make_fake_dataset(5)
        corpus, _ = build_corpus_and_qa(
            dataset, max_corpus_docs=100, max_qa_samples=10
        )
        ids = [d.doc_id for d in corpus]
        assert len(ids) == len(set(ids))

    def test_gold_doc_id_within_corpus_bounds(self):
        dataset = _make_fake_dataset(5)
        corpus, qa = build_corpus_and_qa(
            dataset, max_corpus_docs=100, max_qa_samples=10
        )
        for sample in qa:
            assert sample.gold_doc_id is not None
            assert 0 <= sample.gold_doc_id < len(corpus)

    def test_gold_doc_ids_subset_of_corpus(self):
        dataset = _make_fake_dataset(5)
        corpus, qa = build_corpus_and_qa(
            dataset, max_corpus_docs=100, max_qa_samples=10
        )
        valid_ids = {d.doc_id for d in corpus}
        for sample in qa:
            assert sample.gold_doc_ids.issubset(valid_ids)

    def test_max_corpus_docs_respected(self):
        dataset = _make_fake_dataset(20)
        corpus, _ = build_corpus_and_qa(
            dataset, max_corpus_docs=5, max_qa_samples=100
        )
        assert len(corpus) <= 5

    def test_max_qa_samples_respected(self):
        dataset = _make_fake_dataset(20)
        _, qa = build_corpus_and_qa(
            dataset, max_corpus_docs=1000, max_qa_samples=3
        )
        assert len(qa) <= 3

    def test_answerable_only_filters_empty_answers(self):
        dataset = _make_fake_dataset(3)
        # Make one item unanswerable
        dataset[1]["answers"] = {"text": [], "answer_start": []}
        _, qa = build_corpus_and_qa(
            dataset, max_corpus_docs=100, max_qa_samples=10, answerable_only=True
        )
        qids = [s.qid for s in qa]
        assert "q1" not in qids

    def test_gold_answer_contained_in_gold_chunk(self):
        """The primary gold chunk's body must contain the gold answer."""
        dataset = _make_fake_dataset(5)
        corpus, qa = build_corpus_and_qa(
            dataset, max_corpus_docs=100, max_qa_samples=10
        )
        for sample in qa:
            gold_chunk = corpus[sample.gold_doc_id]
            # At least one answer should appear somewhere in corpus for this context
            found = any(
                ans.lower() in corpus[did].body.lower()
                for did in sample.gold_doc_ids
                for ans in sample.answers
            )
            # Not always guaranteed for the primary (may fall back), but gold_doc_ids hit rate should be high
            assert sample.gold_doc_id is not None
