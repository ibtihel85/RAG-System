import pytest
from src.data.schema import RAGResult
from src.evaluation.relevance import AnswerRelevanceEvaluator


def _make_result(question: str, answer: str) -> RAGResult:
    return RAGResult(
        qid="q0",
        question=question,
        gold_answers=["placeholder"],
        gold_doc_id=0,
        gold_doc_ids={0},
        bm25_pool_ids=[],
        dense_pool_ids=[],
        hybrid_top_ids=[],
        reranked_top_ids=[],
        generated_answer=answer,
    )


class TestAnswerRelevanceEvaluator:

    @pytest.fixture(scope="class")
    def evaluator(self):
        return AnswerRelevanceEvaluator()

    def test_number_answer_scores_high_for_how_many(self, evaluator):
        r = _make_result("How many bones are in the human body?", "206")
        score = evaluator.score_batch([r])[0]
        assert score > 0.5

    def test_non_number_scores_low_for_how_many(self, evaluator):
        r = _make_result("How many people live there?", "some people")
        score = evaluator.score_batch([r])[0]
        assert score < 0.5

    def test_capitalised_answer_scores_high_for_who(self, evaluator):
        r = _make_result("Who wrote Hamlet?", "William Shakespeare")
        score = evaluator.score_batch([r])[0]
        assert score > 0.5

    def test_year_answer_scores_high_for_when(self, evaluator):
        r = _make_result("When did World War II end?", "1945")
        score = evaluator.score_batch([r])[0]
        assert score > 0.5

    def test_idk_scores_zero(self, evaluator):
        r = _make_result("Who built the pyramids?", "I don't know.")
        score = evaluator.score_batch([r])[0]
        assert score == pytest.approx(0.0)

    def test_long_answer_penalised(self, evaluator):
        short_r = _make_result("Where is the Eiffel Tower?", "Paris")
        long_r = _make_result(
            "Where is the Eiffel Tower?",
            "The Eiffel Tower is located in the city of Paris in France in Europe",
        )
        scores = evaluator.score_batch([short_r, long_r])
        assert scores[0] > scores[1]

    def test_batch_length_matches_input(self, evaluator):
        results = [
            _make_result("Who?", "Someone"),
            _make_result("When?", "1990"),
            _make_result("Where?", "London"),
        ]
        scores = evaluator.score_batch(results)
        assert len(scores) == 3

    def test_all_scores_in_range(self, evaluator):
        results = [
            _make_result("How many?", "42"),
            _make_result("Who?", "Einstein"),
            _make_result("When?", "1945"),
        ]
        for score in evaluator.score_batch(results):
            assert 0.0 <= score <= 1.0
