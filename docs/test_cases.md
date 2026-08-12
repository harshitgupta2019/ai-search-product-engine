# Test Inputs and Expected Outputs

The following cases demonstrate both the required examples and edge cases.

| Query | Expected behavior |
|---|---|
| `Gaming keyboard under ₹5000` | Only products priced at or below ₹5,000; gaming keyboards should rank first. |
| `Gift cards for PlayStation` | PlayStation gift cards should rank above unrelated gift cards. |
| `Budget mechanical keyboard` | Mechanical keyboards should rank highly; cheaper matching products receive a soft budget boost. |
| `keyboard between ₹2000 and ₹5000` | Only products in the inclusive price range are returned. |
| `keyboard under ₹100` | Valid request with zero results; no fallback outside the hard price constraint. |
| empty query | HTTP 422 from request validation. |
| `keyboard`, `limit=2` | At most two ranked results. |

## Example output

```json
{
  "query": "Gaming keyboard under ₹5000",
  "parsed_query": {
    "min_price": null,
    "max_price": 5000.0,
    "category_hint": "keyboard",
    "budget_hint": false
  },
  "results": [
    {
      "id": "p1",
      "title": "Redragon K552 Mechanical Gaming Keyboard",
      "category": "Gaming Keyboards",
      "price": 3499,
      "score": 0.91,
      "score_breakdown": {
        "semantic": 0.94,
        "lexical": 0.82,
        "category": 1.0,
        "budget": 0.0
      }
    }
  ]
}
```

The exact numeric score can change if the embedding model is upgraded. The important contract is that hard constraints are respected and results are ordered by the documented hybrid relevance score.
