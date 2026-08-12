from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query

from .schemas import ParsedQuery, SearchResponse, SearchResult
from .search_engine import ProductSearchEngine


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "products.json"

app = FastAPI(
    title="AI Product Search Engine",
    version="1.1.0",
    description=(
        "Hybrid semantic product search using local embeddings, deterministic "
        "query constraints, lexical relevance and category intent."
    ),
)


@lru_cache(maxsize=1)
def get_engine() -> ProductSearchEngine:
    """Lazy-load the ML model so importing the API does not download a model."""
    return ProductSearchEngine(DATA_PATH)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse, tags=["search"])
def search(
    q: str = Query(
        ...,
        min_length=1,
        max_length=300,
        description="Natural-language product query, e.g. 'gaming keyboard under ₹5000'.",
    ),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results."),
):
    try:
        parsed, results = get_engine().search(q, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SearchResponse(
        query=q.strip(),
        parsed_query=ParsedQuery(
            min_price=parsed.min_price,
            max_price=parsed.max_price,
            category_hint=parsed.category_hint,
            budget_hint=parsed.budget_hint,
        ),
        results=[SearchResult(**result) for result in results],
    )
