from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


PRICE = r"(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)"

STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "the",
    "with",
    "under",
    "below",
    "less",
    "than",
    "above",
    "over",
    "more",
    "up",
    "to",
    "in",
    "on",
    "of",
    "is",
    "are",
    "me",
    "show",
    "find",
    "looking",
    "need",
    "want",
    "please",
    "give",
    "card",
    "cards",
    "store",
    "get",
}

CATEGORY_ALIASES = {
    "gaming keyboard": "keyboard",
    "mechanical keyboard": "keyboard",
    "keyboard": "keyboard",
    "gaming headset": "headset",
    "headphones": "headset",
    "earbuds": "earbuds",
    "mouse": "mouse",
    "monitor": "monitor",
    "laptop": "laptop",
    "playstation": "playstation",
    "xbox": "xbox",
    "gift card": "gift card",
    "gift cards": "gift card",
    "smartphone": "smartphone",
    "phone": "smartphone",
}


class EmbeddingModel(Protocol):
    def encode(
        self,
        texts: str | list[str],
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        ...


class SentenceTransformerModel:
    """
    Adapter around sentence-transformers.

    Keeping the model behind this interface makes the search engine
    easy to test without loading a real transformer model.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        texts,
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ):
        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )


@dataclass(frozen=True)
class ParsedQuery:
    min_price: float | None = None
    max_price: float | None = None
    category_hint: str | None = None
    budget_hint: bool = False


class ProductSearchEngine:
    """
    In-memory hybrid semantic search engine.

    Ranking:
        semantic = 65%
        lexical  = 15%
        category = 10%
        budget   = 10%

    Explicit price constraints are hard filters.
    Budget/cheap/affordable are soft ranking preferences.
    """

    def __init__(
        self,
        data_path: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model: EmbeddingModel | None = None,
    ):
        self.data_path = Path(data_path)
        self.model = model or SentenceTransformerModel(model_name)

        self.products = self._load_products()

        self.product_texts = [
            self._product_text(product)
            for product in self.products
        ]

        # Product embeddings are generated once at startup.
        self.embeddings = np.asarray(
            self.model.encode(
                self.product_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

    def _load_products(self) -> list[dict[str, Any]]:
        suffix = self.data_path.suffix.lower()

        if suffix == ".json":
            raw = json.loads(
                self.data_path.read_text(encoding="utf-8")
            )

        elif suffix == ".csv":
            import csv

            with self.data_path.open(
                newline="",
                encoding="utf-8",
            ) as file:
                raw = list(csv.DictReader(file))

        else:
            raise ValueError("Dataset must be JSON or CSV")

        if not isinstance(raw, list):
            raise ValueError(
                "Dataset must contain a list of products"
            )

        required = {
            "id",
            "title",
            "description",
            "category",
            "price",
        }

        products = []
        seen_ids: set[str] = set()

        for product in raw:
            missing = required - set(product)

            if missing:
                raise ValueError(
                    f"Product missing fields: {sorted(missing)}"
                )

            item = dict(product)

            item["id"] = str(item["id"])
            item["price"] = float(item["price"])

            if item["id"] in seen_ids:
                raise ValueError(
                    f"Duplicate product id: {item['id']}"
                )

            if item["price"] < 0:
                raise ValueError(
                    f"Product price cannot be negative: {item['id']}"
                )

            seen_ids.add(item["id"])
            products.append(item)

        return products

    @staticmethod
    def _product_text(product: dict[str, Any]) -> str:
        """
        Build the text representation used for embeddings.

        Repeating the title gives the product name slightly more
        importance in semantic matching.
        """

        return " ".join(
            str(product[key])
            for key in (
                "title",
                "title",
                "description",
                "category",
            )
        )

    @staticmethod
    def parse_query(query: str) -> ParsedQuery:
        q = query.lower().replace(",", "")

        min_price = None
        max_price = None

        # Example:
        # "between ₹2000 and ₹5000"
        between = re.search(
            rf"\bbetween\s+{PRICE}\s+and\s+{PRICE}\b",
            q,
        )

        if between:
            min_price = float(between.group(1))
            max_price = float(between.group(2))

        else:
            # Examples:
            # under ₹5000
            # below 5000
            # less than 5000
            # up to 5000
            upper = re.search(
                rf"\b(?:under|below|less than|up to)\s+{PRICE}\b",
                q,
            )

            # Examples:
            # above ₹5000
            # over 5000
            # more than 5000
            lower = re.search(
                rf"\b(?:above|over|more than)\s+{PRICE}\b",
                q,
            )

            if upper:
                max_price = float(upper.group(1))

            if lower:
                min_price = float(lower.group(1))

        # Longest-first ensures that "mechanical keyboard"
        # is checked before the generic "keyboard".
        category_hint = None

        for phrase in sorted(
            CATEGORY_ALIASES,
            key=len,
            reverse=True,
        ):
            if phrase in q:
                category_hint = CATEGORY_ALIASES[phrase]
                break

        budget_hint = any(
            term in q
            for term in (
                "budget",
                "cheap",
                "affordable",
                "low cost",
            )
        )

        return ParsedQuery(
            min_price=min_price,
            max_price=max_price,
            category_hint=category_hint,
            budget_hint=budget_hint,
        )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(
                r"[a-z0-9]+",
                text.lower(),
            )
            if token not in STOP_WORDS
            and len(token) > 1
        }

    def _lexical_score(
        self,
        query: str,
        product: dict[str, Any],
    ) -> float:
        query_tokens = self._tokens(query)

        if not query_tokens:
            return 0.0

        title_tokens = self._tokens(
            product["title"]
        )

        body_tokens = self._tokens(
            self._product_text(product)
        )

        title_recall = (
            len(query_tokens & title_tokens)
            / len(query_tokens)
        )

        body_recall = (
            len(query_tokens & body_tokens)
            / len(query_tokens)
        )

        return min(
            1.0,
            0.7 * title_recall
            + 0.3 * body_recall,
        )

    @staticmethod
    def _category_score(
        parsed: ParsedQuery,
        product: dict[str, Any],
    ) -> float:
        if not parsed.category_hint:
            return 0.0

        haystack = " ".join(
            str(product[field]).lower()
            for field in (
                "title",
                "description",
                "category",
            )
        )

        aliases = {
            "keyboard": ("keyboard",),
            "headset": ("headset", "headphone"),
            "earbuds": ("earbud",),
            "mouse": ("mouse",),
            "monitor": ("monitor",),
            "laptop": ("laptop",),
            "playstation": ("playstation",),
            "xbox": ("xbox",),
            "gift card": ("gift card",),
            "smartphone": (
                "smartphone",
                "phone",
            ),
        }

        return float(
            any(
                term in haystack
                for term in aliases.get(
                    parsed.category_hint,
                    (),
                )
            )
        )

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> tuple[
        ParsedQuery,
        list[dict[str, Any]],
    ]:
        query = query.strip()

        if not query:
            raise ValueError(
                "Search query cannot be empty"
            )

        if not 1 <= limit <= 50:
            raise ValueError(
                "limit must be between 1 and 50"
            )

        parsed = self.parse_query(query)

        query_embedding = np.asarray(
            self.model.encode(
                query,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

        candidates = []

        for index, product in enumerate(
            self.products
        ):
            # Explicit prices are hard constraints.
            if (
                parsed.max_price is not None
                and product["price"] > parsed.max_price
            ):
                continue

            if (
                parsed.min_price is not None
                and product["price"] < parsed.min_price
            ):
                continue

            cosine = float(
                self.embeddings[index]
                @ query_embedding
            )

            # Cosine similarity is [-1, 1].
            # Normalize to [0, 1].
            semantic = (cosine + 1.0) / 2.0

            lexical = self._lexical_score(
                query,
                product,
            )

            category = self._category_score(
                parsed,
                product,
            )

            candidates.append(
                {
                    "product": product,
                    "semantic": semantic,
                    "lexical": lexical,
                    "category": category,
                }
            )

        # Budget is intentionally a soft preference.
        if parsed.budget_hint and candidates:
            prices = [
                float(item["product"]["price"])
                for item in candidates
            ]

            low = min(prices)
            high = max(prices)

            span = max(high - low, 1.0)

            for item in candidates:
                item["budget"] = (
                    1.0
                    - (
                        item["product"]["price"]
                        - low
                    )
                    / span
                )
        else:
            for item in candidates:
                item["budget"] = 0.0

        # Hybrid relevance score.
        for item in candidates:
            item["score"] = (
                0.65 * item["semantic"]
                + 0.15 * item["lexical"]
                + 0.10 * item["category"]
                + 0.10 * item["budget"]
            )

        # Deterministic tie-breakers make tests and production
        # behavior reproducible.
        candidates.sort(
            key=lambda item: (
                -item["score"],
                item["product"]["price"],
                item["product"]["id"],
            )
        )

        results = []

        for item in candidates[:limit]:
            product = item["product"]

            results.append(
                {
                    **product,
                    "score": round(
                        float(item["score"]),
                        4,
                    ),
                    "score_breakdown": {
                        "semantic": round(
                            float(item["semantic"]),
                            4,
                        ),
                        "lexical": round(
                            float(item["lexical"]),
                            4,
                        ),
                        "category": round(
                            float(item["category"]),
                            4,
                        ),
                        "budget": round(
                            float(item["budget"]),
                            4,
                        ),
                    },
                }
            )

        return parsed, results