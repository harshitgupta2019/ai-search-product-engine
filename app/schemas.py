from typing import Optional

from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    semantic: float = Field(ge=0, le=1)
    lexical: float = Field(ge=0, le=1)
    category: float = Field(ge=0, le=1)
    budget: float = Field(ge=0, le=1)


class SearchResult(BaseModel):
    id: str
    title: str
    description: str
    category: str
    price: float
    score: float = Field(ge=0, le=1)
    score_breakdown: ScoreBreakdown


class ParsedQuery(BaseModel):
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    category_hint: Optional[str] = None
    budget_hint: bool = False


class SearchResponse(BaseModel):
    query: str
    parsed_query: ParsedQuery
    results: list[SearchResult]