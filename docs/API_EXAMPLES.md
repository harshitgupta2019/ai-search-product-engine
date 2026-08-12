# API Examples

## Example 1 — Gaming keyboard under ₹5000

```bash
curl "http://localhost:8000/search?q=Gaming%20keyboard%20under%20%E2%82%B95000&limit=5"
```

Expected behavior:

- `max_price = 5000`
- category hint = `keyboard`
- no product above ₹5000 is returned
- gaming keyboards should rank highly

## Example 2 — PlayStation gift cards

```bash
curl "http://localhost:8000/search?q=Gift%20cards%20for%20PlayStation&limit=5"
```

Expected behavior: PlayStation gift cards rank above unrelated gift cards.

## Example 3 — Budget mechanical keyboard

```bash
curl "http://localhost:8000/search?q=Budget%20mechanical%20keyboard&limit=5"
```

Expected behavior: mechanical keyboards rank highly and lower-priced matching products receive a soft budget preference.

## Example 4 — No result under hard constraint

```bash
curl "http://localhost:8000/search?q=keyboard%20under%20%E2%82%B9100"
```

Expected behavior: an empty result list. The service must not fall back to products that violate the explicit budget.
