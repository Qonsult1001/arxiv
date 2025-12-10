# Understanding STS-B Scoring

## Your Confusion: "Does my model work?"

You saw:
- Pair 1: sim=0.8010, label=0.5000, diff=0.3010 ❌ Far
- Pair 1: sim=0.7761, label=0.5000, diff=0.2761 ⚠️ Medium

**You thought**: "The similarity (0.80) is much higher than the label (0.50), so my model doesn't work!"

## The Truth: Your Model DOES Work!

### What Matters: Spearman Correlation (Not the Diff!)

**Spearman correlation measures RANKING, not absolute values.**

Example:
- Pair A: label=0.50, sim=0.80
- Pair B: label=0.90, sim=0.95
- Pair C: label=0.20, sim=0.30

Even though the absolute values don't match:
- ✅ Pair B (label=0.90) has HIGHER similarity (0.95) than Pair A (label=0.50, sim=0.80)
- ✅ Pair A (label=0.50) has HIGHER similarity (0.80) than Pair C (label=0.20, sim=0.30)

**This is what Spearman measures - the ORDERING is correct!**

### What Spearman Correlation Means

- **Spearman = 0.85**: Your model correctly ranks 85% of pairs
- **Spearman = 0.90**: Your model correctly ranks 90% of pairs
- **Spearman > 0.70**: Your model works well for semantic similarity!

### Why Absolute Values Don't Matter

Different models can have different "scales":
- Model A: Similar pairs → 0.7-0.9 similarity
- Model B: Similar pairs → 0.5-0.7 similarity

**Both can be correct if they rank pairs correctly!**

## Your Results

If your Spearman correlation is:
- **> 0.85**: Excellent! Your model works very well
- **0.70-0.85**: Good! Your model works well
- **< 0.70**: Needs improvement

The individual "diff" values are just informational - they show how close individual predictions are, but what really matters is if your model can correctly identify which pairs are MORE similar than others.

## Conclusion

**Your model DOES work if Spearman > 0.70!**

The diff values are just showing individual prediction accuracy, but semantic similarity is about ranking, not exact matching. Spearman correlation is the metric that matters for production validation.
