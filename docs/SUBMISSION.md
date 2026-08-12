# Lysto — AI Product Search Assignment Submission

## 1. Solution summary

This submission implements a production-minded MVP for natural-language product search using FastAPI and a hybrid retrieval/ranking pipeline.

The system accepts a query such as `Gaming keyboard under ₹5000`, extracts explicit constraints, removes products that violate hard constraints, and ranks the remaining catalog using semantic similarity plus lexical, category, and budget signals.

## 2. Deliverables

| Deliverable | Location |
|---|---|
| Working API | `app/main.py` |s
| Search/ranking engine | `app/search_engine.py` |
| Request/response schemas | `app/schemas.py` |
| Product dataset | `data/products.json` |
| Automated tests | `tests/test_search.py` |
| Test cases + expected behavior | `docs/test_cases.md` |
| Architecture | `docs/architecture.md` |
| Optional LLM prompt | `docs/llm_prompt.md` |
| Setup + design decisions | `README.md` |

## 3. API contract

### Health

`GET /health`

Returns:

```json
{"status":"ok"}
```

### Search

`GET /search?q=<natural-language-query>&limit=<1-50>`

The response contains:

- original query
- parsed price/category/budget intent
- ranked products
- an explainable score breakdown

## 4. Ranking strategy

The final score is:

```text
0.65 × semantic similarity
+ 0.15 × lexical relevance
+ 0.10 × category relevance
+ 0.10 × budget preference
```

Hard price constraints are applied before scoring.

This separation is intentional: an explicit user constraint such as `under ₹5000` must never be overridden by semantic relevance.

## 5. Why hybrid search?

Semantic retrieval handles meaning and synonyms. Lexical matching preserves strong exact-term signals such as product names. Category intent improves precision when a query clearly targets a product family. The budget signal handles words such as `budget` without pretending that they represent an exact price ceiling.

## 6. AI/LLM decision

The runtime does not require an external LLM. This keeps the assignment deterministic, inexpensive, and runnable without an API key.

An optional LLM query-understanding prompt is provided in `docs/llm_prompt.md`. If introduced in production, the LLM would produce structured intent; retrieval and final ranking would remain deterministic.

## 7. Testing

The suite contains 9 tests covering:

- health endpoint
- upper price constraints
- inclusive price ranges
- PlayStation gift-card intent
- budget as a soft preference
- result limits
- zero-result hard constraints
- empty-query validation
- score explainability

The tests use a deterministic fake embedding model, so they do not depend on downloading a transformer model and remain fast in CI.

Latest local run:

```text
9 passed in 0.59s
```

## 8. Production evolution

For a larger catalog, the in-memory matrix should be replaced with an ANN/vector index such as pgvector or OpenSearch. A production system would also add BM25/full-text retrieval, rank fusion or a learned ranker, relevance judgments, query analytics, caching, asynchronous indexing, model/index versioning, and observability.

## 9. Assumptions and scope

The assignment-sized catalog is loaded from JSON and prices are numeric INR values. Inventory, seller, availability, location, and personalization are outside the current scope. Supported natural-language price expressions are documented in the README and tests.
