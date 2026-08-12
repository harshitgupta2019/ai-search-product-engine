from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.search_engine import ProductSearchEngine


DATA_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "products.json"
)


class FakeEmbeddingModel:
    """
    Deterministic fake embedding model.

    Tests should not depend on downloading a transformer
    model or having network access.
    """

    def __init__(self):
        self.vectors = {
            "keyboard": np.array(
                [1.0, 0.0, 0.0]
            ),
            "gaming": np.array(
                [0.9, 0.1, 0.0]
            ),
            "gift": np.array(
                [0.0, 1.0, 0.0]
            ),
            "playstation": np.array(
                [0.0, 0.9, 0.1]
            ),
            "mouse": np.array(
                [0.0, 0.0, 1.0]
            ),
        }

    def encode(
        self,
        texts,
        *,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        def one(text: str) -> np.ndarray:
            lower = text.lower()

            vector = np.zeros(3)

            for token, weight in self.vectors.items():
                if token in lower:
                    vector += weight

            if not np.any(vector):
                vector[0] = 1.0

            return vector / np.linalg.norm(vector)

        if isinstance(texts, str):
            return one(texts)

        return np.vstack(
            [one(text) for text in texts]
        )


def fake_engine() -> ProductSearchEngine:
    return ProductSearchEngine(
        DATA_PATH,
        model=FakeEmbeddingModel(),
    )


@pytest.fixture(autouse=True)
def use_fake_engine(monkeypatch):
    monkeypatch.setattr(
        "app.main.get_engine",
        fake_engine,
    )


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_under_price_constraint():
    response = client.get(
        "/search",
        params={
            "q": "gaming keyboard under ₹5000"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["parsed_query"]["max_price"]
        == 5000.0
    )

    assert (
        body["parsed_query"]["category_hint"]
        == "keyboard"
    )

    assert body["results"]

    assert all(
        item["price"] <= 5000
        for item in body["results"]
    )

    assert (
        body["results"][0]["category"]
        == "Gaming Keyboards"
    )


def test_between_price_constraint():
    parsed = fake_engine().parse_query(
        "keyboard between ₹2,000 and ₹5,000"
    )

    assert parsed.min_price == 2000
    assert parsed.max_price == 5000


def test_playstation_query_returns_gift_cards():
    response = client.get(
        "/search",
        params={
            "q": "Gift cards for PlayStation"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["results"]

    assert body["results"][0]["id"] in {
        "p4",
        "p5",
    }

    assert (
        body["results"][0]["category"]
        == "Gift Cards"
    )


def test_budget_is_soft_preference():
    response = client.get(
        "/search",
        params={
            "q": "Budget mechanical keyboard"
        },
    )

    assert response.status_code == 200

    results = response.json()["results"]

    assert results

    # Budget should favor a cheaper relevant product,
    # but it should not behave like a hard price filter.
    assert results[0]["price"] <= 3499

    assert (
        results[0]["score_breakdown"]["budget"]
        >= 0
    )


def test_limit():
    response = client.get(
        "/search",
        params={
            "q": "keyboard",
            "limit": 2,
        },
    )

    assert response.status_code == 200

    assert len(
        response.json()["results"]
    ) <= 2


def test_no_results_when_hard_constraint_excludes_catalog():
    response = client.get(
        "/search",
        params={
            "q": "keyboard under ₹100"
        },
    )

    assert response.status_code == 200

    assert (
        response.json()["results"]
        == []
    )


def test_empty_query_rejected():
    response = client.get(
        "/search",
        params={
            "q": ""
        },
    )

    assert response.status_code == 422


def test_result_scores_are_explainable():
    response = client.get(
        "/search",
        params={
            "q": "gaming keyboard under ₹5000"
        },
    )

    result = response.json()["results"][0]

    breakdown = result[
        "score_breakdown"
    ]

    expected = (
        0.65 * breakdown["semantic"]
        + 0.15 * breakdown["lexical"]
        + 0.10 * breakdown["category"]
        + 0.10 * breakdown["budget"]
    )

    assert abs(
        result["score"] - expected
    ) < 0.001