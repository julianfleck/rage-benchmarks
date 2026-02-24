# memU Architecture Analysis vs RAGE

## Summary

After examining the memU codebase ([NevaMind-AI/memU](https://github.com/NevaMind-AI/memU) and [NevaMind-AI/memU-experiment](https://github.com/NevaMind-AI/memU-experiment)), here's what gives memU its claimed 92.09% accuracy on LoCoMo.

## Key Architectural Differences

### memU's "Secret Sauce"

1. **Extremely Thorough Memory Extraction (LLM-Heavy)**
   - Each conversation session goes through **3 iterations of LLM refinement**
   - Uses a "sufficiency checker" that evaluates if extracted memories are complete
   - Iterates until either sufficient or max iterations reached
   - This means **~6 LLM calls per session** (extract + check × 3) per character
   - For conv-26 with 19 sessions and 2 characters: **~228 LLM calls just for memory building**

2. **Hybrid Retrieval with Aggressive Deduplication**
   ```
   BM25 (keyword) → 10 results
   String Matching (substring + similarity) → 10 results
   Semantic Embeddings (cosine similarity) → 10 results
   ↓
   Combine & Deduplicate with weighted scoring
   ↓
   Top-K final results
   ```

3. **Character-Centric Memory Organization**
   - Separate files: `{Character}_events.txt`, `{Character}_profile.txt`
   - Profiles are cleaned and consolidated after session processing
   - Events are timestamped with session date/time

4. **Multi-iteration Question Answering**
   - ResponseAgent can do up to 3 retrieval iterations per question
   - Each iteration refines search keywords based on previous results
   - Uses LLM to judge if retrieved content is "sufficient" to answer

### RAGE's Approach (Different Philosophy)

1. **Single-Pass Attention-Based Retrieval**
   - No iterative LLM refinement during ingestion
   - Phase resonance for relevance propagation
   - Temporal decay built into the substrate

2. **Graph-Structured Storage**
   - Frames connected by typed edges
   - Attention fields propagate through graph
   - No separate "extraction" step - content is stored directly

3. **Divergence-Oriented**
   - Designed to resist convergence
   - Surfaces structurally salient memories even without semantic overlap

## Why memU Gets High Scores

### 1. Brute-Force Completeness
memU's iterative extraction ensures **nothing is missed**. Example from our test:

```
Iteration 1: Extract events → "sufficient: False, missing_info: Caroline mentioned..."
Iteration 2: Re-extract with context → "sufficient: False, missing_info: ..."
Iteration 3: Final extraction with all context → stored
```

This is essentially using GPT-4 as a "did I miss anything?" checker 3 times per session.

### 2. Question-Specific Retrieval Refinement
When answering questions, memU can:
- Search with initial keywords
- Check if results are sufficient
- Refine keywords and search again
- Repeat up to 3 times

This is essentially **using the LLM to do retrieval-augmented retrieval**.

### 3. No Information Loss
- Every conversation utterance is analyzed
- Profile information is extracted separately from events
- Temporal information preserved in timestamps

## Cost/Speed Implications

For conv-26 alone:
- **19 sessions × 2 characters × ~6 LLM calls = ~228 calls** for memory building
- **199 questions × ~3 retrieval iterations = ~597 calls** for QA
- **Plus evaluation calls**

Total: **800+ LLM calls** for one LoCoMo conversation

Our partial benchmark run showed **~10 seconds per session** for extraction, suggesting:
- Full conv-26 memory building: ~3-4 minutes
- Full QA evaluation: ~30-40 minutes
- **Total: ~45 minutes per conversation**

## Implications for RAGE

### What RAGE Could Adopt
1. **Multi-iteration extraction checking** (expensive but thorough)
2. **Hybrid retrieval** (BM25 + embeddings + string match)
3. **Explicit profile vs events separation**

### What RAGE Does Differently (Potential Advantages)
1. **Graph structure** enables structural queries memU can't do
2. **Phase resonance** may surface non-obvious connections
3. **Temporal decay** is built-in vs memU's static timestamps
4. **Single-pass ingestion** is much faster and cheaper

### The Real Question
Is memU's 92% accuracy worth:
- 800+ LLM calls per conversation?
- 45+ minutes processing time?
- No structural/graph query capability?

Or can RAGE achieve competitive accuracy with:
- Single-pass ingestion
- Graph-based retrieval
- Attention propagation

## Recommendations

1. **Run full benchmark** (needs ~1 hour):
   ```bash
   python3 benchmarks/memu/run_memu_conv26.py --model="openai/gpt-4o-mini"
   ```

2. **Compare on same questions**: Create RAGE benchmark runner that answers the same 199 conv-26 questions

3. **Analyze error patterns**: What questions does memU get right that RAGE gets wrong? Are they:
   - Multi-hop reasoning?
   - Temporal reasoning?
   - Entity disambiguation?

4. **Consider hybrid approach**: Could RAGE use iterative extraction for critical sessions while maintaining graph structure?

## Actual memU Results (Pre-computed)

The memU-experiment repo includes pre-computed results (`result.json`) from their official benchmark:

### Overall (All 10 LoCoMo Conversations)
- **Total Questions**: 1,986
- **Correct**: 1,420
- **Overall Accuracy**: 92.09%

### Conv-26 Specific Results
| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| Single-hop | 30 | 32 | 93.8% |
| Multi-hop | 37 | 37 | 100% |
| Temporal | 13 | 13 | 100% |
| Open-domain | 69 | 70 | 98.6% |
| Adversarial | 2 | 2 | 100% |
| **TOTAL** | **151** | **154** | **98.1%** |

**Key Observation**: Conv-26 has 199 questions total, but memU only evaluated 154 (they excluded most category 5 adversarial questions).

### Benchmark Configuration Used
- Model: `gpt-4.1-mini-2`
- Profile usage: `prompt` (include profile in prompt)
- Max workers: 20 (parallel processing)
- Image captions: Included

## Conclusion

memU's high LoCoMo scores come from **brute-force LLM iteration** rather than architectural innovation. It's essentially using GPT-4 as a comprehensive note-taker that double-checks its own work multiple times.

This is expensive and slow, but thorough. The question is whether RAGE's attention-based approach can achieve similar completeness more efficiently, or if the use case requires this level of LLM-heavy processing.

**The 98.1% accuracy on conv-26 is impressive, but:**
1. It required ~800+ LLM calls per conversation
2. Processing takes ~45 minutes per conversation
3. Most adversarial questions (45/47) were excluded
4. RAGE may be able to achieve similar results faster with graph-based retrieval

---

*Analysis based on:*
- memU source code examination
- Partial benchmark run on conv-26
- Pre-computed result.json analysis
- LoCoMo benchmark structure analysis
