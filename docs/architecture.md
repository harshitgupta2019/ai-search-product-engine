# Architecture

```text
                         GET /search?q=...
                                  |
                                  v
                       +----------------------+
                       | FastAPI API layer    |
                       | validation + schema  |
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       | Query understanding   |
                       | price + category +    |
                       | budget hints           |
                       +----------+-----------+
                                  |
                     hard price filtering
                                  |
                                  v
                       +----------------------+
                       | Candidate scoring     |
                       |                      |
                       | semantic  65%         |
                       | lexical   15%         |
                       | category  10%         |
                       | budget    10%          |
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       | Ranked JSON response  |
                       | + score breakdown     |
                       +----------------------+

Startup/indexing path:

products.json
     |
     v
Dataset validation
     |
     v
Product text construction
     |
     v
Sentence-transformer embeddings
     |
     v
In-memory embedding matrix
```

## Why this architecture?

The assignment catalog is small, so an in-memory matrix keeps the implementation simple and easy to review. Product embeddings are computed once at startup; each request only embeds the query and scores candidates.

Hard constraints are applied before ranking. This prevents a highly semantically similar product that violates a user-specified budget from appearing in the results.
