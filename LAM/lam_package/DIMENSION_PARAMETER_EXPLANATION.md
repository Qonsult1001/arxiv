# Dimension Parameter Explanation

## Question: Why `dimensions=` vs `dimension=`?

**Answer: They do the SAME thing!** It's just different naming conventions.

## Both Work Identically

```python
from lam import LAM

model = LAM("LAM-base-v1")

# Both of these do EXACTLY the same thing:
emb1 = model.encode(sentences, dimensions=64)  # plural
emb2 = model.encode(sentences, dimension=64)   # singular

# Result: emb1 and emb2 are identical
```

## Why Two Names?

1. **`dimensions=` (plural)** - LAM's original API
   - More grammatically correct (you're specifying multiple dimensions: 64, 128, 256, 384)
   - Matches Python naming conventions

2. **`dimension=` (singular)** - Added for `lam_embed` compatibility
   - `lam_embed` uses `dimension=` (singular)
   - Allows drop-in compatibility with `lam_embed` code

## Technical Details

Internally, LAM handles both:
```python
# In lam/__init__.py
def encode(self, ..., dimensions=None, dimension=None, ...):
    # Both parameters do the same thing
    target_dim = dimensions if dimensions is not None else (dimension if dimension is not None else None)
    # ... rest of code uses target_dim
```

## Recommendation

- **Use `dimensions=`** if writing new LAM code (matches LAM's API)
- **Use `dimension=`** if porting from `lam_embed` (drop-in compatibility)

**Both work identically - choose whichever you prefer!**

