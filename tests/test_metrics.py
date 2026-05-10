
import pytest
from src.data.schema import Document, RAGResult
from src.evaluation.metrics import (
    normalise,
    token_f1,
    exact_match,
    rouge_l,
    recall_at_k,
    mrr_at_k,
    context_precision,
    compute_qa_metrics,
    _body_contains_answer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    gold_answers,
    reranked_ids,
    gold_doc_ids=None,
    gold_doc_id=0,
    generated_answer="",
    bm25_pool_ids=None,
    dense_pool_ids=None,
    hybrid_top_ids=None,
):
    return RAGResult(
        qid="q0",
        question="What year?",
        gold_answers=gold_answers,
        gold_doc_id=gold_doc_id,
        gold_doc_ids=gold_doc_ids or {gold_doc_id},
        bm25_pool_ids=bm25_pool_ids or [],
        dense_pool_ids=dense_pool_ids or [],
        hybrid_top_ids=hybrid_top_ids or [],
        reranked_top_ids=reranked_ids,
        generated_answer=generated_answer,
    )


def _make_corpus(n=5):
    return [
        Document(
            doc_id=i,
            title="Title",
            text=f"Title: Chunk {i} content about topic {i}.",
            body=f"Chunk {i} content about topic {i}.",
            source_ctx="",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_lowercases(self):
        assert normalise("Hello World") == "hello world"

    def test_strips_articles(self):
        assert "the" not in normalise("the cat sat on the mat")
        assert "a" not in normalise("a dog")
        assert "an" not in normalise("an apple")

    def test_removes_punctuation(self):
        assert "," not in normalise("Hello, World!")
        assert "." not in normalise("End.")

    def test_collapses_whitespace(self):
        result = normalise("  multiple   spaces  ")
        assert "  " not in result


# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------

class TestTokenF1:
    def test_identical_strings(self):
        assert token_f1("1945", "1945") == pytest.approx(1.0)

    def test_disjoint_strings(self):
        assert token_f1("apple", "orange") == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = token_f1("William Shakespeare", "Shakespeare")
        assert 0.0 < score < 1.0

    def test_empty_prediction(self):
        assert token_f1("", "gold") == pytest.approx(0.0)

    def test_empty_gold(self):
        assert token_f1("prediction", "") == pytest.approx(0.0)

    def test_both_empty(self):
        assert token_f1("", "") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_exact_hit(self):
        assert exact_match("Paris", "Paris") is True

    def test_article_stripped(self):
        assert exact_match("The Beatles", "Beatles") is True

    def test_case_insensitive(self):
        assert exact_match("paris", "Paris") is True

    def test_mismatch(self):
        assert exact_match("London", "Paris") is False


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

class TestRougeL:
    def test_identical(self):
        assert rouge_l("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint(self):
        assert rouge_l("foo", "bar") == pytest.approx(0.0)

    def test_partial(self):
        score = rouge_l("the cat sat on the mat", "the cat")
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_hit_at_1(self):
        r = _make_result(["ans"], reranked_ids=[0, 1, 2], gold_doc_id=0)
        assert recall_at_k([r], "reranked_top_ids", k=1) == pytest.approx(1.0)

    def test_miss_at_1_hit_at_3(self):
        r = _make_result(["ans"], reranked_ids=[1, 2, 0], gold_doc_id=0)
        assert recall_at_k([r], "reranked_top_ids", k=1) == pytest.approx(0.0)
        assert recall_at_k([r], "reranked_top_ids", k=3) == pytest.approx(1.0)

    def test_empty_results(self):
        assert recall_at_k([], "reranked_top_ids", k=5) == pytest.approx(0.0)

    def test_multi_gold_ids(self):
        r = _make_result(
            ["ans"], reranked_ids=[3, 4, 5], gold_doc_id=0, gold_doc_ids={3, 0}
        )
        assert recall_at_k([r], "reranked_top_ids", k=1) == pytest.approx(1.0)

    def test_partial_recall_over_batch(self):
        r1 = _make_result(["ans"], reranked_ids=[0], gold_doc_id=0)
        r2 = _make_result(["ans"], reranked_ids=[9], gold_doc_id=0)
        assert recall_at_k([r1, r2], "reranked_top_ids", k=1) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# MRR@K
# ---------------------------------------------------------------------------

class TestMRRAtK:
    def test_gold_at_rank_1(self):
        r = _make_result(["ans"], reranked_ids=[0, 1, 2], gold_doc_id=0)
        assert mrr_at_k([r], "reranked_top_ids", k=3) == pytest.approx(1.0)

    def test_gold_at_rank_2(self):
        r = _make_result(["ans"], reranked_ids=[1, 0, 2], gold_doc_id=0)
        assert mrr_at_k([r], "reranked_top_ids", k=3) == pytest.approx(0.5)

    def test_gold_not_in_top_k(self):
        r = _make_result(["ans"], reranked_ids=[1, 2, 3], gold_doc_id=0)
        assert mrr_at_k([r], "reranked_top_ids", k=3) == pytest.approx(0.0)

    def test_empty_results(self):
        assert mrr_at_k([], "reranked_top_ids", k=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Context Precision
# ---------------------------------------------------------------------------

class TestContextPrecision:
    def test_perfect_precision(self):
        corpus = _make_corpus(5)
        # Body of chunk 0 contains the gold answer
        corpus[0] = Document(0, "T", "T: gold answer here", "gold answer here", "")
        r = _make_result(
            gold_answers=["gold answer"],
            reranked_ids=[0],
            gold_doc_id=0,
            gold_doc_ids={0},
        )
        score = context_precision([r], corpus, "reranked_top_ids", k=1)
        assert score == pytest.approx(1.0)

    def test_zero_precision_no_relevant(self):
        corpus = _make_corpus(5)
        r = _make_result(
            gold_answers=["zzz"],
            reranked_ids=[1, 2, 3],
            gold_doc_id=0,
            gold_doc_ids={0},
        )
        score = context_precision([r], corpus, "reranked_top_ids", k=3)
        assert score == pytest.approx(0.0)

    def test_empty_top_ids(self):
        corpus = _make_corpus(5)
        r = _make_result(gold_answers=["ans"], reranked_ids=[], gold_doc_id=0)
        score = context_precision([r], corpus, "reranked_top_ids", k=5)
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Compute QA metrics
# ---------------------------------------------------------------------------

class TestComputeQAMetrics:
    def test_perfect_answer(self):
        r = _make_result(
            gold_answers=["1945"],
            reranked_ids=[],
            generated_answer="1945",
        )
        m = compute_qa_metrics([r])
        assert m["Exact Match"] == pytest.approx(1.0)
        assert m["Token F1"] == pytest.approx(1.0)

    def test_wrong_answer(self):
        r = _make_result(
            gold_answers=["Paris"],
            reranked_ids=[],
            generated_answer="London",
        )
        m = compute_qa_metrics([r])
        assert m["Exact Match"] == pytest.approx(0.0)
        assert m["Token F1"] == pytest.approx(0.0)

    def test_multi_gold_best_of_n(self):
        r = _make_result(
            gold_answers=["William Shakespeare", "Shakespeare"],
            reranked_ids=[],
            generated_answer="Shakespeare",
        )
        m = compute_qa_metrics([r])
        assert m["Exact Match"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Body-contains-answer helper
# ---------------------------------------------------------------------------

class TestBodyContainsAnswer:
    def test_exact_substring(self):
        assert _body_contains_answer("The answer is 1945.", ["1945"])

    def test_case_insensitive(self):
        assert _body_contains_answer("Paris is the capital.", ["paris"])

    def test_no_match(self):
        assert not _body_contains_answer("Nothing relevant here.", ["1945"])

    def test_empty_answer(self):
        assert not _body_contains_answer("Some text.", [""])

    def test_short_word_boundary(self):
        # "in" should not match "interesting"
        assert not _body_contains_answer("interesting case", ["in"])
