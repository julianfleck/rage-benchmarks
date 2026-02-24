# LoCoMo Benchmark Retrieval Analysis
**Date:** 2026-02-20  
**Current Score:** 31.2% (62/199 correct)  
**Target:** 92%

## Executive Summary

After detailed analysis of 137 failures (69% failure rate), the breakdown is:

| Failure Type | Count | % of Failures | Description |
|--------------|-------|---------------|-------------|
| **Comprehension** | 63 | 46% | Answer IS in context, model answered wrong |
| **Partial Retrieval** | 37 | 27% | Some GT words in context, incomplete info |
| **Pure Retrieval** | 37 | 27% | Answer completely absent from context |

**Key Finding:** The "80% retrieval failures" estimate was too high. Only ~27-54% of failures are retrieval issues (depending on how you count partial). The remaining ~46% are comprehension failures where the LLM failed to extract the correct answer from valid context.

---

## Failure Categories (137 total)

| Category | Failures | Key Issue |
|----------|----------|-----------|
| multi-hop | 44 | Requires combining info from multiple frames |
| adversarial | 29 | Model fooled by wrong-person attribution |
| single-hop | 28 | Mixed retrieval + comprehension |
| temporal | 25 | Model echoes "last Saturday" instead of computing date |
| commonsense | 11 | Requires inference the model won't make |

---

## Pure Retrieval Failures Analysis (37 cases)

### Pattern 1: Semantic Gap (10 cases)
**Example:** "Where did Caroline move from 4 years ago?" → GT: "Sweden"

**Problem:** The word "Sweden" appears in a frame about a necklace gift:
> "This necklace is super special to me - a gift from my grandma in my home country, Sweden."

The query "move from" has zero semantic overlap with "necklace gift from grandma".

**What would help:**
- **Slot search**: The frame has `origin: "Sweden"` in slots
- **Entity-focused multi-query**: "Caroline origin", "Caroline home country"

### Pattern 2: Multi-Source Aggregation (17 cases)
**Example:** "Where has Melanie camped?" → GT: "beach, mountains, forest"

**Problem:** Evidence is spread across 3 separate frames in different sessions (D6:16, D4:6, D8:32). Retrieval found "Melanie went camping with her kids" but NOT the specific locations.

**What would help:**
- **Slot aggregation**: Frames have `location: "beach"`, `location: "mountains"`, etc.
- **Entity-constrained retrieval**: Filter to Melanie frames, then aggregate `location` slots

### Pattern 3: Specific Terms Not Matched (10 cases)
**Example:** "What kind of art does Caroline make?" → GT: "abstract art"  
**Retrieved context mentions:** "trans-themed paintings", "self-portrait"

The specific term "abstract" appears in a different frame that wasn't retrieved.

**What would help:**
- **Multi-query expansion**: "Caroline art style", "Caroline paintings type"

---

## Comprehension Failures Analysis (63 cases)

### Pattern 1: Temporal Computation (25 cases)
**Example:** "When did Melanie run a charity race?" → GT: "The Sunday before 25 May 2023"

Context has:
```
created_at: 2023-05-25T13:14:00
Melanie ran a charity race for mental health last Saturday.
```

Model answered: "Last Saturday" (echoing content) instead of computing "May 20/21, 2023"

**Problem:** Model doesn't do date math from `created_at` timestamps.

**Fix:** Better system prompt for temporal reasoning, or preprocessing that converts relative dates.

### Pattern 2: Wrong Answer Extraction (20 cases)
**Example:** "What is Caroline's identity?" → GT: "Transgender woman"

Context contains: "Caroline's Transgender Transition", "undergoing a personal transition"

Model answered: "Explored through art."

**Problem:** Model picked a tangential detail instead of the direct answer.

**Fix:** Better answer extraction prompting, or structured output format.

### Pattern 3: Partial Answers (18 cases)
**Example:** "What career path has Caroline decided to pursue?"  
GT: "counseling or mental health for Transgender people"  
Pred: "Counseling or mental health"

Context has all the words, model just didn't include the specificity.

**Fix:** Prompt for complete answers.

---

## Adversarial Failures (29/47 = 62% failure rate)

Adversarial questions are designed to confuse speakers:
- "What did Caroline realize after her charity race?" (Caroline didn't run a race, Melanie did)

**Problem:** Retrieval finds semantically similar content from the WRONG person.

**What would help:**
- **Entity filtering**: If question asks about "Caroline's charity race" but no such frame exists, return empty context
- **Confidence thresholding**: Low-confidence results should be excluded

---

## Specific Improvement Recommendations

### Priority 1: Multi-Query with Entity Focus
**Impact:** Would address ~15 pure retrieval failures

Current NLP expansion extracts phrases but doesn't generate entity-focused variations.

**Implementation:**
```python
def expand_query_with_entities(query: str) -> List[str]:
    queries = [query]  # Original
    
    # Extract named entities
    entities = extract_entities(query)  # e.g., ["Caroline", "Melanie"]
    
    # Generate entity-focused variations
    for entity in entities:
        queries.append(f"{entity}")  # Just the entity
        queries.append(f"{entity} {extract_topic(query)}")  # Entity + topic
    
    return queries
```

For "Where has Melanie camped?":
- Original: "Where has Melanie camped?"
- Entity-focused: "Melanie", "Melanie camping", "Melanie camping location"

### Priority 2: Slot-Aware Retrieval
**Impact:** Would address ~20 retrieval failures where slot data exists

Many frames have structured slots like:
- `location: "mountains"`, `location: "beach"`
- `origin: "Sweden"`
- `activity: "camping"`

**Implementation:**
1. Index slots as searchable text in FTS5 (already done but not fully utilized)
2. For questions about locations/origins/activities, weight slot matches higher
3. Add slot-specific query variations: "location:mountains", "activity:camping"

### Priority 3: Entity Filtering for Adversarial Robustness
**Impact:** Would address ~20 adversarial failures

If question mentions "Caroline's charity race" but no Caroline+charity frames exist, the model should abstain rather than finding Melanie's charity race.

**Implementation:**
```python
def filter_by_mentioned_entities(query: str, frames: List[Frame]) -> List[Frame]:
    entities = extract_entities(query)  # ["Caroline"]
    
    if not entities:
        return frames
    
    # Keep only frames that mention at least one entity
    filtered = []
    for frame in frames:
        frame_entities = extract_entities(frame.content)
        if any(e in frame_entities for e in entities):
            filtered.append(frame)
    
    return filtered if filtered else frames  # Fall back to all if none match
```

### Priority 4: Temporal Preprocessing
**Impact:** Would address ~20 temporal comprehension failures

Convert relative dates to absolute in retrieved context before sending to LLM:

```python
def preprocess_temporal_context(context: str, frame_dates: dict) -> str:
    # For each frame with created_at, resolve relative terms
    # "last Saturday" + created_at=2023-05-25 → "Saturday May 20, 2023"
    ...
```

### Priority 5: Better Answer Extraction Prompting
**Impact:** Would address ~15 comprehension failures

Current prompt:
> "Answer questions concisely using only the provided context. Give SHORT PHRASE answers."

Improved prompt:
> "Extract the EXACT answer from the context. Include all relevant details. For temporal questions, COMPUTE the actual date using the frame's created_at timestamp."

---

## Metrics Comparison: What Each Improvement Would Yield

| Improvement | Est. Failures Fixed | New Score |
|-------------|---------------------|-----------|
| Baseline (current) | 0 | 31.2% |
| + Multi-query entity focus | +15 | 38.7% |
| + Slot-aware retrieval | +12 | 44.7% |
| + Entity filtering | +10 | 49.7% |
| + Temporal preprocessing | +15 | 57.3% |
| + Better prompting | +10 | 62.3% |

**Estimated ceiling with all improvements:** ~65-70%

The remaining gap to 92% would require:
- Multi-hop reasoning (combining info from multiple frames)
- Advanced commonsense inference
- More sophisticated adversarial detection

---

## Implementation Order

1. **Week 1:** Multi-query with entity extraction (highest ROI)
2. **Week 2:** Slot-aware retrieval weighting
3. **Week 3:** Entity filtering + confidence thresholding
4. **Week 4:** Temporal preprocessing + prompt improvements

Each improvement can be tested independently with the benchmark.

---

## Code Locations

- **Query expansion:** `rage_substrate/retrieval/context_retrieval.py:_expand_query_nlp()`
- **Keyword search:** `rage_substrate/retrieval/keyword_retrieval.py:fts_search()`
- **Benchmark runner:** `rage-benchmarks/benchmarks/locomo_benchmark.py:RAGEQARunner`

## Appendix: Sample Failures with Full Context

See `results/locomo_rejudged.json` for complete data including:
- Full retrieved context for each question
- Ground truth and predictions
- Tool calls made
- Latency metrics
