

from __future__ import annotations

import re
from typing import List

from src.data.schema import RAGResult


class AnswerRelevanceEvaluator:
    
    _WH_NUMBER = re.compile(
        r"^how (many|much|old|far|long|often|tall|deep|wide)"
        r"|what (percentage|number|amount|proportion|fraction"
        r"|speed|temperature|distance|height|weight)",
        re.IGNORECASE,
    )
    _WH_DATE = re.compile(
        r"^when |^what (year|date|century|decade|time|month|day)",
        re.IGNORECASE,
    )
    _WH_PERSON = re.compile(r"^who |^whom ", re.IGNORECASE)
    _WH_PLACE = re.compile(r"^where ", re.IGNORECASE)
    _NUM_PATTERN = re.compile(
        r"\b\d[\d,.]*\b|^\s*(one|two|three|four|five|six|seven|eight|nine|ten)\s*$",
        re.IGNORECASE,
    )

    def _answer_type_match(self, question: str, answer: str) -> float:
        if self._WH_NUMBER.search(question):
            return 0.85 if self._NUM_PATTERN.search(answer) else 0.15
        if self._WH_DATE.search(question):
            return 0.85 if self._NUM_PATTERN.search(answer) else 0.25
        if self._WH_PERSON.search(question):
            words = answer.split()
            has_cap = any(w[0].isupper() for w in words if w)
            return 0.80 if has_cap else 0.20
        if self._WH_PLACE.search(question):
            words = answer.split()
            has_cap = any(w[0].isupper() for w in words if w)
            return 0.80 if has_cap else 0.25
        return 0.55  # neutral for open "what/which" definitional questions

    @staticmethod
    def _length_score(answer: str) -> float:
        word_count = len(answer.split())
        if word_count == 0:
            return 0.0
        if word_count <= 8:
            return 1.0
        if word_count <= 15:
            return 0.7
        return max(0.1, 1.0 - (word_count - 15) * 0.05)

    def score_batch(self, results: List[RAGResult]) -> List[float]:
        
        scores = []
        for r in results:
            ans = r.generated_answer
            if not ans or ans.strip().lower() in ("i don't know", "i don't know."):
                scores.append(0.0)
                continue

            type_match = self._answer_type_match(r.question, ans)
            length_score = self._length_score(ans)
            scores.append(float(0.65 * type_match + 0.35 * length_score))

        return scores
