# memU Benchmark Results: Independent Verification

## Summary

We independently ran memU's LoCoMo benchmark on conv-26 to verify their claimed 98.1% accuracy.

**Result: 87.7%** — a 10.4 percentage point discrepancy from their published numbers.

---

## Our Results vs Their Claims

| Category | Our Run | Their Claim | Difference |
|----------|---------|-------------|------------|
| Single-hop (Cat 1) | 25/32 (78.1%) | 30/32 (93.8%) | **-15.7%** |
| Multi-hop (Cat 2) | 33/37 (89.2%) | 37/37 (100%) | **-10.8%** |
| Temporal (Cat 3) | 13/13 (100%) | 13/13 (100%) | 0% |
| Open-domain (Cat 4) | 62/70 (88.6%) | 69/70 (98.6%) | **-10.0%** |
| Adversarial (Cat 5) | 2/2 (100%) | 2/2 (100%) | 0% |
| **TOTAL** | **135/154 (87.7%)** | **151/154 (98.1%)** | **-10.4%** |

---

## Benchmark Configuration

### Our Run
- **Date**: 2026-02-21
- **Model**: openai/gpt-4o-mini (via OpenRouter)
- **Memory Extraction**: Force regenerated (--force-resum)
- **Workers**: 5 parallel
- **Total Time**: 2063.86 seconds (~34 minutes)
- **Questions Evaluated**: 154 of 199

### Their Claimed Configuration (from result.json)
- **Model**: gpt-4.1-mini-2
- **Profile Usage**: prompt (include in prompt)
- **Workers**: 20 parallel
- **Questions Evaluated**: 154 of 199

---

## Key Observations

### 1. 45 Questions Were Skipped

Both runs evaluated only 154 of the 199 questions. The 45 skipped questions are primarily from Category 5 (Adversarial), which tests:
- Unanswerable questions
- Questions with false premises
- Questions requiring "I don't know" answers

This means **only 2 of 47 adversarial questions were evaluated**.

### 2. Temporal Reasoning is Strong

Category 3 (Temporal) achieved 100% on both runs. This suggests:
- The explicit timestamp format ("Mentioned at 1:56 pm on 8 May, 2023") is effective
- Temporal questions may be easier for LLM-based systems

### 3. Single-Hop Shows Largest Gap

Our 78.1% vs their 93.8% on single-hop questions (15.7% gap) is concerning. Possible explanations:
- Model differences (gpt-4.1-mini-2 vs gpt-4o-mini)
- Prompt variations
- Cherry-picked runs in published results

### 4. Multi-Hop Also Underperforms

Our 89.2% vs their claimed 100% suggests their multi-hop results may have been unusually good.

---

## Errors by Category

### Category 1: Single-hop (7 errors)
- Basic factual questions about events
- May be retrieval failures (relevant event not in top-K)

### Category 2: Multi-hop (4 errors)
- Questions requiring connecting multiple facts
- Reasoning chain failures

### Category 3: Temporal (0 errors)
- Perfect performance on time-based questions

### Category 4: Open-domain (8 errors)
- Broader questions about characters
- May require more comprehensive retrieval

### Category 5: Adversarial (0 errors)
- Only 2 questions evaluated
- Not a representative sample

---

## Cost Analysis

### Actual Cost for This Run
- **Embedding calls**: ~398K tokens
- **LLM calls (extraction)**: ~228 calls
- **LLM calls (QA)**: ~462 calls
- **Total**: ~$0.65 (estimated)

### Time
- Memory extraction: ~10 minutes
- QA answering: ~24 minutes
- **Total**: ~34 minutes

---

## Files Generated

```
external/memU-experiment/
├── enhanced_memory_test_results_20260221_164543.json  # Full results
├── qa_error_log_20260221_164543.txt                   # Error details

benchmarks/memu/memory/
├── Caroline_events.txt                                # Extracted events
├── Caroline_profile.txt                               # Character profile
├── Melanie_events.txt
└── Melanie_profile.txt
```

---

## Reproducibility

To reproduce this benchmark:

```bash
cd /Users/julian/Tresors/Privat/Code/rage-benchmarks
source .venv/bin/activate
python3 benchmarks/memu/run_memu_conv26.py \
    --model="openai/gpt-4o-mini" \
    --force-resum
```

Note: Requires OpenRouter API key configured in the script.

---

## Conclusion

memU's published accuracy claims appear inflated by approximately 10 percentage points. This could be due to:

1. **Model differences**: They may have used a different/better model
2. **Cherry-picking**: They may have selected their best run
3. **Evaluation differences**: Different evaluation criteria
4. **Version differences**: Code may have changed since publication

Regardless, even the 87.7% we achieved is impressive for a retrieval-based system. The architecture has merit—the claims just need calibration.

---

*Benchmark run on 2026-02-21 using memU-experiment commit from NevaMind-AI/memU-experiment*
