"""
Shared pytest fixtures.
"""

import pytest
from src.data.schema import Document


@pytest.fixture(scope="session")
def small_corpus():
    """A tiny 5-document corpus for tests that need one."""
    return [
        Document(
            doc_id=i,
            title=f"Article {i}",
            text=f"Article {i}: This is the content of document {i} about topic {i}.",
            body=f"This is the content of document {i} about topic {i}.",
            source_ctx=f"Full paragraph for article {i}.",
        )
        for i in range(5)
    ]
