# AI Prompts Used

I used AI assistance during the development of this assignment for
implementation ideas, debugging, testing, and reviewing the overall
approach.

Below are some of the prompts I used during development.

---

## 1. Getting started

> I have to build an AI product search engine for an assignment.
> It should take queries like "Gaming keyboard under ₹5000" and return
> relevant products from a JSON/CSV dataset. I want to use Python and
> FastAPI. How would you structure the project?s

---

## 2. Semantic search

> I don't want this to be just keyword search. What's a simple way to
> add semantic search using embeddings without making the project too
> complicated?

---

## 3. Ranking results

> If I have semantic similarity, keyword matching and category matching,
> how should I combine them into one score? I also want to be able to
> explain why one product ranked above another.

---

## 4. Price handling

> How should I handle queries like "under ₹5000", "above ₹2000" and
> "between ₹2000 and ₹5000"? I think explicit price limits should be
> treated as filters rather than just ranking signals.

---

## 5. Budget queries

> What should I do with something like "budget mechanical keyboard"?
> There isn't an exact price mentioned, so I don't want to randomly
> decide that budget means under ₹3000. How can I handle this?

---

## 6. Testing

> Can you help me write tests for the three example queries from the
> assignment and also cover things like no results, price filters and
> result limits?

---

## 7. Making tests independent of the model

> My tests shouldn't have to download a sentence transformer model every
> time. What's a clean way to mock the embedding model so the tests stay
> deterministic?

---

## 8. Reviewing the implementation

> Can you review this like a backend interviewer would? I'm looking for
> anything that feels over-engineered, unclear or missing from an
> assignment submission.

---

## 9. README

> Help me write a README for this project that explains how to run it,
> how the search works, the ranking logic and the assumptions I made.
> Keep it practical rather than sounding like a marketing document.

---

## 10. Final review

> Go through the assignment requirements and my implementation and tell
> me if I have missed any deliverable or important edge case.