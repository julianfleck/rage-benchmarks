# RAGE Benchmark Results

## 2026-02-17 Initial Run

### Setup
- **Database**: rage-substrate/substrate.db (586 frames)
- **Queries**: 5 direct factual queries
- **RAGE Effort**: medium (default changed from low)
- **Baseline**: OpenAI text-embedding-3-small via OpenRouter

### Key Findings

#### The Low Effort Problem
Initially, RAGE with `effort="low"` returned **0 frames for all queries**. The low effort mode uses fast sample-only retrieval which didn't find matches in this database. 

Switching to `effort="medium"` fixed this:
- Uses query expansion (4 variations)
- Applies phase resonance reranking
- Successfully retrieves relevant frames

#### Performance Comparison

| System   | Recall@1 | Recall@5 | Avg Latency | Avg Tokens |
|----------|----------|----------|-------------|------------|
| RAGE     | 0.10     | 0.10     | 12.2s       | 303        |
| Baseline | 0.00     | 0.20     | 0.5s        | 313        |

#### Analysis

**RAGE Strengths:**
- Better Recall@1 (found the exact answer 10% of the time vs 0%)
- Retrieved 10 frames per query (more context)
- Query expansion helps find variations

**RAGE Weaknesses:**
- Much slower (12s vs 0.5s) - 24x slower
- Query expansion sometimes fails (JSON parsing errors)
- Doesn't outperform baseline significantly on this dataset

**Baseline Strengths:**
- Much faster
- Better Recall@5 (20% vs 10%)
- Simpler, more predictable

### Sample Query Results

**Query 1: "What is the BNF grammar for RAGE addresses?"**
- RAGE: Found "Grammar (BNF)" at position 3 (R@1=0.5, R@5=0.5)
- Baseline: Found it at position 5 (R@1=0.0, R@5=1.0)

**Query 2-5: Other direct queries**
- Both systems struggled to find exact expected frames
- This may indicate:
  - Queries need better expected frame titles
  - Database doesn't contain the expected content
  - Retrieval needs tuning

### Recommendations

1. **Fix the default effort level**: Changed `simple_retrieval.py` default from "low" to "medium"
2. **Fix query expansion errors**: Some queries cause JSON parsing failures
3. **Test on larger datasets**: 586 frames is small
4. **Add more query categories**: Test temporal and structural queries
5. **Investigate low effort mode**: Why does it return 0 frames?

### Technical Issues Fixed

1. ✅ Import paths work correctly
2. ✅ Substrate tools.execute_sync API integration
3. ✅ BaselineRAG loads frames from SQLite
4. ✅ OpenRouter embeddings work (no OpenAI API key needed)
5. ✅ Metrics calculation working

### Next Steps

- Run full benchmark (56 queries)
- Investigate query expansion failures
- Profile RAGE latency (where's the 12s going?)
- Test with high effort mode
- Compare against different baseline chunk sizes
