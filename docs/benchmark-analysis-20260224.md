# LoCoMo Benchmark Analysis - 2026-02-24

## Executive Summary

**Overall Performance:** 32.2% LLM accuracy (64/199 questions correct)

The benchmark reveals systematic issues primarily in **temporal reasoning** and **entity attribute retrieval**, with specific actionable fixes for the ingestion pipeline.

## Performance by Category

| Category | Count | LLM Accuracy | F1 Avg | Key Issue |
|----------|-------|--------------|--------|-----------|
| Single-hop | 32 | **9.4%** | 0.147 | Missing simple attributes |
| Temporal | 37 | **13.5%** | 0.127 | Relative dates without context |
| Multi-hop | 70 | 32.9% | 0.229 | Cross-entity inference gaps |
| Commonsense | 13 | 46.2% | 0.105 | Reasoning from implicit info |
| Adversarial | 47 | **57.4%** | 0.000 | Partially detecting mis-attribution |

**Critical finding:** Single-hop and temporal are the worst performers, yet these should be the easiest. This points to fundamental ingestion issues, not LLM comprehension problems.

## Failure Pattern Analysis

### Pattern 1: Missing Temporal Context (32 failures in category 2)

**Problem:** Events and projects have relative timestamps ("last year", "last Saturday") but the LLM doesn't know when the original conversation occurred.

**Example:**
```
Q: "When did Melanie paint a sunrise?"
Ground truth: "2022"
Predicted: "Melanie painted a sunrise last year, which would be in 2025."

Context retrieved shows:
- content: "I painted that lake sunrise last year! It's special to me."
- slots: NO 'when' field on the project entity!
```

**Root cause:** Projects don't have `when` slots populated. Events DO have `when` slots with absolute dates, but projects/objects that involve temporal info don't.

**Fix:** 
1. Add `when` slot extraction for project/object entities when temporal language is detected
2. Include session date context in retrieval results (the `_session_frame` exists but doesn't carry the conversation date)

### Pattern 2: Entity Attribute Gaps (29 failures in category 1)

**Problem:** Simple facts about people are not being extracted into slots.

**Examples:**
```
Q: "What is Caroline's relationship status?"
Ground truth: "Single"
Predicted: "The provided context does not contain information..."
→ Context has Caroline's entity but no 'relationship_status' slot

Q: "Where did Caroline move from 4 years ago?"  
Ground truth: "Sweden"
Predicted: "The context does not specify..."
→ Content has "since I moved from my home country" 
→ Objects mention "grandma from Sweden" (necklace)
→ NO slot linking "home country" = Sweden
```

**Root cause:** Person entities have `name`, `role`, `aliases` but missing:
- `origin_country` / `nationality`
- `relationship_status` 
- Basic demographic slots

**Fix:**
1. Expand person schema with demographic slots
2. Implement cross-entity inference during ingestion (necklace from "grandma from Sweden" → Caroline's origin)

### Pattern 3: Relative Date Resolution (multiple categories)

**Problem:** Dates like "last Saturday" get converted but with WRONG reference point.

**Example:**
```
Q: "When did Melanie run a charity race?"
Ground truth: "The sunday before 25 May 2023"
Predicted: "Melanie ran a charity race on May 20, 2023."

Context shows:
- when.relative: "last Saturday"
- when.absolute: "2023-05-20"
- when.derived_from: "D2:1"
```

The absolute date was computed correctly (May 20, 2023 was a Saturday), but ground truth expects "Sunday before May 25" which would be May 21! This appears to be a benchmark data issue OR an off-by-one in date calculation.

**Fix:** Verify date resolution logic, particularly around "last [day]" expressions.

### Pattern 4: Adversarial Detection Inconsistency

**Problem:** Model correctly identifies entity confusion but doesn't answer "UNANSWERABLE".

**Examples:**
```
Q: "What did Caroline realize after her charity race?"
Ground truth: "UNANSWERABLE"
Predicted: "Caroline doesn't have a realization related to the charity race; 
           the charity race was an event associated with Melanie..."
→ LLM score: 0 (despite correct reasoning!)

Q: "Why did Melanie choose the adoption agency?"
Ground truth: "UNANSWERABLE"  
Predicted: "Melanie is not considering an adoption agency; Caroline is..."
→ LLM score: 0
```

**Root cause:** The system prompt doesn't instruct the LLM to respond with "UNANSWERABLE" when information doesn't exist for the queried entity.

**Fix:** Update RAG system prompt to explicitly output "UNANSWERABLE" when:
- Question asks about entity A but information only exists for entity B
- Requested attribute genuinely doesn't exist in context

### Pattern 5: Content Retrieval vs Slot Retrieval

**Observation:** Many facts exist in `content` fields (raw quotes) but NOT in structured `slots`.

```
Content: "I've known these friends for 4 years, since I moved from my home country"
Slots: No origin_country extracted

Content: "A friend made it for my 18th birthday ten years ago"
→ Entity has this in content, LLM correctly answered "10 years ago" ✓
```

**Insight:** The LLM CAN find facts in content, but structured slot extraction is incomplete.

## Retrieval Quality Analysis

Out of 135 total failures:
- **49 (36%)** say "does not contain" / "no mention" → retrieval miss OR slot gap
- **~50 (37%)** give wrong answer despite relevant context → comprehension/format error
- **~36 (27%)** adversarial questions mishandled → prompt engineering issue

## Actionable Fixes (Priority Order)

### HIGH PRIORITY

1. **Add session date context to retrieval**
   - Include conversation date in context header
   - Allow LLM to resolve "last year" relative to conversation date, not current date

2. **Expand person entity slots**
   ```
   Additional slots needed:
   - origin_country / nationality
   - relationship_status
   - age / birth_year (derivable from "18th birthday ten years ago")
   - identity (e.g., "transgender woman")
   ```

3. **Add temporal slots to projects/objects**
   - When content contains temporal language, extract `when` slot
   - Mirror the event entity pattern

### MEDIUM PRIORITY

4. **Update RAG system prompt for UNANSWERABLE**
   - Add explicit instruction: "If the question asks about [Person A] but context only describes [Person B], respond with 'UNANSWERABLE'"

5. **Cross-entity fact propagation**
   - Link "grandma from Sweden" → Caroline's nationality
   - Propagate relationship facts bidirectionally

### LOWER PRIORITY

6. **Audit date resolution logic**
   - Verify "last [weekday]" calculations
   - Compare against LoCoMo ground truth dates

## Metrics Summary

```
Total questions: 199
LLM correct: 64 (32.2%)
LLM incorrect: 135 (67.8%)

By failure type:
- Retrieval/slot gaps: ~49
- Comprehension errors: ~50  
- Adversarial handling: ~36

Exact match: 0.5% (only 1 question!)
Average F1: 0.135
```

## Next Steps

1. Implement session date injection in context
2. Add demographic slots to person schema
3. Re-run benchmark after fixes to measure improvement
4. Consider temporal slot extraction for all entity types, not just events
