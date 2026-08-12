# AI Product Search Engine

A production-minded MVP for the Lysto **AI Product Search Engine** assignment.

The service accepts natural-language queries, understands simple structured constraints such as price ranges, and ranks products using a hybrid semantic + lexical relevance strategy rather than keyword matching alone.

## Assignment coverage

| Requirement | Implementation |
|---|---|
| Load CSV/JSON dataset | JSON and CSV loaders with schema validation |
| Product fields | `id`, `title`, `description`, `category`, `price` |
| Natural-language search | FastAPI `GET /search` |
| Semantic understanding | `all-MiniLM-L6-v2` sentence embeddings |
| Ranked results | Hybrid weighted scoring |
| Example queries | Gaming keyboard, PlayStation gift cards, budget mechanical keyboard |
| Test input/output | `docs/test_cases.md` + automated pytest suite |
| Ranking explanation | `README.md` + `docs/architecture.md` |
| AI prompt | `docs/llm_prompt.md` |
| Assumptions | README section |

## 1. Design overview

```text
Request
  -> API validation
  -> query parsing
       - hard price constraints
       - category intent
       - budget preference
  -> hard constraint filtering
  -> hybrid relevance scoring
       - semantic similarity: 65%
       - lexical relevance: 15%
       - category relevance: 10%
       - budget preference: 10% when applicable
  -> deterministic tie-breakers
  -> ranked JSON response
```

Product embeddings are generated once when the search engine is initialized. The query embedding is generated at request time.

For the assignment-sized catalog, a linear in-memory scan is intentional: it avoids introducing infrastructure that does not add value at this scale while keeping the retrieval pipeline easy to reason about.

## 2. Why hybrid search?

Pure keyword matching is brittle. For example, a user searching for `budget mechanical keyboard` may expect products described as `gaming keyboard`, `tactile switches`, or `RGB backlight` even when the exact query words do not all occur in the title.

Semantic similarity handles conceptual relevance. Lexical matching protects exact high-signal terms. Category matching adds a small deterministic intent boost. Structured constraints such as `under ₹5000` are treated as **hard filters**, because violating an explicit user constraint is worse than losing a small amount of semantic relevance.

## 3. Ranking formula

For each candidate:

```text
score =
    0.65 * semantic_similarity
  + 0.15 * lexical_score
  + 0.10 * category_score
  + 0.10 * budget_score
```

`budget_score` is zero unless the query contains terms such as `budget`, `cheap`, or `affordable`.

### Semantic score

The product's title, description and category are embedded at indexing time. The query is embedded at request time. Because embeddings are normalized, their dot product is cosine similarity. The value is mapped from `[-1, 1]` to `[0, 1]` before weighting.

### Lexical score

Token overlap is calculated against the product text. Title matches receive more weight than description/category matches. This provides a useful exact-match signal without turning the system into a keyword-only search engine.

### Category score

A small controlled vocabulary maps phrases such as `mechanical keyboard`, `PlayStation`, and `gift cards` to normalized category intents. Matching products receive a deterministic boost.

### Budget score

`budget` is treated as a soft preference rather than a hard price limit. Among otherwise eligible candidates, lower-priced products receive a larger budget score. This allows a relevant product to remain searchable even when the user did not specify an exact maximum price.

## 4. Query constraints

Supported examples:

- `under ₹5000`
- `below 5000`
- `less than ₹5000`
- `up to ₹5000`
- `above ₹2000`
- `over ₹2000`
- `between ₹2000 and ₹5000`

Price constraints are inclusive for ranges and hard filters for search.

## 5. API

### `GET /health`

```json
{"status": "ok"}
```

### `GET /search`

Parameters:

- `q`: required natural-language query, max 300 characters
- `limit`: optional result count, 1–50, default 10

Example:

```bash
curl "http://localhost:8000/search?q=Gaming%20keyboard%20under%20%E2%82%B95000&limit=5"
```

Response includes both the parsed query and a transparent score breakdown for every result.

## 6. Project structure

```text
.
├── app/
│   ├── main.py              # FastAPI routes and API validation
│   ├── schemas.py           # Pydantic response contracts
│   └── search_engine.py     # Parsing, embeddings, filtering, ranking
├── data/
│   └── products.json        # Assignment catalog
├── docs/
│   ├── architecture.md     # Architecture diagram and decisions
│   ├── llm_prompt.md        # Optional LLM query-understanding prompt
│   └── test_cases.md       # Test input/output examples
├── tests/
│   └── test_search.py       # Unit + API tests
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

## 7. Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

The embedding model is downloaded on first use. No API key is required.

Interactive API documentation is available at `/docs`.

## 8. Run tests

```bash
pytest -q
```

Tests use a tiny deterministic fake embedding model, so the suite does not download a transformer model or depend on network access. This keeps CI fast while the production path uses the real sentence-transformer model.

The suite covers:

- health endpoint
- price constraints
- range parsing
- PlayStation semantic/category intent
- budget preference
- result limits
- zero-result behavior
- request validation
- score-breakdown consistency

## 9. AI component / prompt

The runtime does **not** require an external LLM. Semantic understanding is provided by a local embedding model plus deterministic query parsing.

An optional LLM-based query-understanding prompt is provided in `docs/llm_prompt.md`. If introduced in production, the LLM should return structured intent/constraints rather than generate the final product ranking. Retrieval and ranking should remain deterministic and testable.

## 10. Assumptions

1. Product IDs are unique.
2. Price is numeric and expressed in INR.
3. The assignment catalog is small enough for an in-memory embedding matrix.
4. Inventory, availability, seller, brand and location are outside scope.
5. Explicit price constraints are hard constraints.
6. `budget`/`cheap`/`affordable` are soft preferences because they do not define an exact threshold.
7. The controlled category vocabulary is intentionally small and can be expanded as the catalog grows.
8. Search relevance should be evaluated with labeled query/product judgments before tuning weights for production.

## 11. Production evolution

If the catalog grows significantly, I would evolve the same logical pipeline rather than rewrite the API:

1. **Retrieval:** move embeddings to pgvector, OpenSearch, Pinecone, or another ANN-capable vector store.
2. **Hybrid retrieval:** combine vector search with BM25/full-text retrieval.
3. **Ranking:** use reciprocal-rank fusion or a learned ranker trained on click/relevance data.
4. **Indexing:** move embedding generation to an asynchronous indexing worker and version the embedding model.
5. **Observability:** log query latency, zero-result rate, top-result clicks, constraint extraction failures and model/index versions.
6. **Evaluation:** maintain a small human-labeled relevance set and track Recall@K / NDCG@K during model or weight changes.
7. **Caching:** cache frequent query embeddings and/or final result sets where freshness requirements allow it.

## 12. Trade-offs

### Why not use an LLM for every search?

For this problem, an LLM would add latency, cost, network dependency and another failure mode. The required semantic retrieval can be achieved locally with embeddings. An LLM is more useful for extracting richer structured intent when the query language becomes significantly more complex.

### Why not use a vector database?

The supplied dataset is tiny. A vector database would increase operational complexity without materially improving the assignment solution. The interface is deliberately structured so the retrieval layer can be replaced later.

### Why expose score breakdown?

Search ranking is easier to debug and review when the result contains its component scores. This also makes weight tuning more explainable during an interview discussion.
