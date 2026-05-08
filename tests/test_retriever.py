"""
Unit tests for retrieval components.

BM25 and HybridRetriever tests use a small synthetic corpus so they run
without downloading any models. DenseRetriever tests are skipped unless
a GPU/sufficient memory is available (they download a ~400MB model).
"""

import pytest
from src.data.schema import Document
from src.models.retriever import BM25Retriever, multi_expand_query, _content_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(doc_id: int, body: str, title: str = "T") -> Document:
    return Document(
        doc_id=doc_id,
        title=title,
        text=f"{title}: {body}",
        body=body,
        source_ctx=body,
    )


CORPUS = [
    _make_doc(0, "The Eiffel Tower is located in Paris France."),
    _make_doc(1, "William Shakespeare wrote Hamlet and Macbeth."),
    _make_doc(2, "World War II ended in 1945 with Allied victory."),
    _make_doc(3, "The Amazon rainforest is the largest tropical forest."),
    _make_doc(4, "Albert Einstein developed the theory of relativity."),
]


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class TestBM25Retriever:

    @pytest.fixture(scope="class")
    def retriever(self):
        return BM25Retriever(CORPUS)

    def test_returns_list_of_tuples(self, retriever):
        results = retriever.retrieve("Paris France", top_k=3)
        assert isinstance(results, list)
        for item in results:
            assert len(item) == 2

    def test_top_k_respected(self, retriever):
        results = retriever.retrieve("Shakespeare Hamlet", top_k=2)
        assert len(results) <= 2

    def test_relevant_doc_ranked_first(self, retriever):
        results = retriever.retrieve("Shakespeare Hamlet", top_k=5)
        top_id = results[0][0]
        assert top_id == 1  # Doc 1 is about Shakespeare

    def test_eiffel_tower_query(self, retriever):
        results = retriever.retrieve("Eiffel Tower Paris", top_k=5)
        top_id = results[0][0]
        assert top_id == 0

    def test_scores_are_non_negative(self, retriever):
        results = retriever.retrieve("any query", top_k=5)
        for _, score in results:
            assert score >= 0.0

    def test_empty_query_does_not_crash(self, retriever):
        results = retriever.retrieve("", top_k=3)
        assert isinstance(results, list)

    def test_doc_ids_within_corpus_bounds(self, retriever):
        results = retriever.retrieve("Einstein relativity", top_k=5)
        for doc_id, _ in results:
            assert 0 <= doc_id < len(CORPUS)


# ---------------------------------------------------------------------------
# Query expansion helpers
# ---------------------------------------------------------------------------

class TestMultiExpandQuery:

    def test_returns_three_bm25_variants(self):
        variants, dense_q = multi_expand_query("Who wrote Hamlet?")
        assert len(variants) == 3

    def test_dense_query_is_raw_question(self):
        question = "What year did World War II end?"
        _, dense_q = multi_expand_query(question)
        assert dense_q == question

    def test_variants_are_strings(self):
        variants, _ = multi_expand_query("Where is the Eiffel Tower?")
        for v in variants:
            assert isinstance(v, str)
            assert len(v) > 0

    def test_why_query_gets_cause_token(self):
        variants, _ = multi_expand_query("Why did the Roman Empire fall?")
        # v1 should contain "cause"
        assert "cause" in variants[0]

    def test_content_tokens_removes_stopwords(self):
        tokens = _content_tokens("What is the capital of France?")
        assert "what" not in tokens
        assert "the" not in tokens
        assert "is" not in tokens
        assert "of" not in tokens

    def test_content_tokens_max_length_respected(self):
        tokens = _content_tokens("one two three four five six seven eight", max_tokens=3)
        assert len(tokens.split()) <= 3


# ---------------------------------------------------------------------------
# Diversity filter (imported from reranker module, tested here for coverage)
# ---------------------------------------------------------------------------

class TestDiversityFilter:

    def test_removes_near_duplicates(self):
        from src.models.reranker import diversity_filter
        # Two highly similar docs should result in only one being kept
        corpus = [
            _make_doc(0, "The cat sat on the mat the cat sat"),
            _make_doc(1, "The cat sat on the mat the cat sat"),  # identical → Jaccard=1.0
            _make_doc(2, "Einstein developed relativity theory physics"),
        ]
        ranked = [(0, 1.0), (1, 0.9), (2, 0.8)]
        result = diversity_filter(ranked, corpus, top_k=3, sim_threshold=0.9)
        ids = [r[0] for r in result]
        # Both 0 and 1 should NOT both appear
        assert not (0 in ids and 1 in ids)

    def test_keeps_diverse_docs(self):
        from src.models.reranker import diversity_filter
        corpus = [
            _make_doc(0, "Paris is the capital of France located in Europe"),
            _make_doc(1, "Einstein developed general theory of relativity"),
            _make_doc(2, "Shakespeare wrote Hamlet and Macbeth tragedy"),
        ]
        ranked = [(0, 1.0), (1, 0.9), (2, 0.8)]
        result = diversity_filter(ranked, corpus, top_k=3, sim_threshold=0.55)
        assert len(result) == 3

    def test_top_k_respected(self):
        from src.models.reranker import diversity_filter
        corpus = [_make_doc(i, f"Completely different doc {i} about topic x y z") for i in range(5)]
        ranked = [(i, float(5 - i)) for i in range(5)]
        result = diversity_filter(ranked, corpus, top_k=2)
        assert len(result) == 2

    def test_empty_input(self):
        from src.models.reranker import diversity_filter
        result = diversity_filter([], CORPUS, top_k=5)
        assert result == []
