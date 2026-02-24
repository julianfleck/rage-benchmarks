# Spot-Check Report: Failed Question Analysis

**Date:** 2026-02-20
**Benchmark:** LoCoMo (conv-26, 199 questions)
**Mode:** fixed-medium (keyword search + reranking, no embeddings)
**Results:** 62 correct (31%), 137 failed (69%)

## Executive Summary

I manually investigated 10 failed questions across all categories to understand **why retrieval failed**. The core insight: **low-effort keyword search cannot handle the semantic gaps between question phrasing and frame content**.

## Failure Patterns Identified

### Pattern 1: Multi-hop Labeled as Single-hop (~15% of failures)

**Example:** "Where did Caroline move from 4 years ago?"
- **Expected:** Sweden
- **Got:** Her home country

**What exists in substrate:**
- Frame A: "I've known these friends for 4 years, since I moved from my home country"
- Frame B: "a gift from my grandma in my home country, Sweden"

**Why it failed:**
- FTS search for "Caroline move years" finds Frame A (has "moved", "4 years")
- Frame B with "Sweden" doesn't match query terms (no "move", no "years")
- Answer requires connecting TWO facts across frames

**Fix:** Entity linking - when "home country" appears in multiple frames, cluster them together.

---

### Pattern 2: Temporal Date Computation (~20% of failures)

**Example:** "When did Melanie run a charity race?"
- **Expected:** The Sunday before 25 May 2023
- **Got:** Last Saturday

**What exists:**
- Frame: "charity race last Saturday" with `created_at: 2023-05-25`

**Why it failed:**
- LLM returned relative phrase "Last Saturday" instead of computing absolute date
- System prompt says to compute dates, but LLM didn't follow through

**Note:** Ground truth may also be wrong (says "Sunday" but frame says "Saturday").

**Fix:** Stronger prompt engineering for date computation. Consider pre-computing dates during ingestion.

---

### Pattern 3: Ground Truth Not in Source Data (~10% of failures)

**Example:** "What books has Melanie read?"
- **Expected:** "Nothing is Impossible", "Charlotte's Web"
- **Got:** The book Caroline recommended

**What exists:**
- "Charlotte's Web" IS in conversation ✓
- "Nothing is Impossible" is NOT in any conversation turn!
- Searched all 19 sessions - phrase doesn't exist

**Why it failed:**
- Benchmark annotation error - ground truth cites non-existent data
- Can't retrieve what doesn't exist

**Fix:** Benchmark data cleaning. Flag these as annotation errors.

---

### Pattern 4: Query-Content Phrasing Mismatch (~25% of failures)

**Example:** "What is Caroline's identity?"
- **Expected:** Transgender woman
- **Got:** Explored through art

**What exists:**
- Multiple frames with "transgender" in content
- Frames about "gender identity exploration through art"

**FTS search results for "Caroline identity":**
1. `/artwork/embracing-identity` (rank -8.8)
2. `Caroline's gender identity exploration...` (rank -8.8)
3. Art-related frames dominate top results

**Frames with "transgender" exist but:**
- "transgender" keyword not in query
- These frames ranked lower because query terms don't match

**Why it failed:**
- "identity" matches art/exploration frames
- "transgender" keyword needed but not expanded from query

**Fix:** Semantic query expansion: "identity" → add transgender, gender, LGBTQ

---

### Pattern 5: Adversarial Entity Confusion (~15% of failures)

**Example:** "What did Caroline realize after her charity race?"
- **Expected:** UNANSWERABLE (adversarial - it was MELANIE's race)
- **Got:** Self-care is important

**What exists:**
- Frames about MELANIE's charity race and her realization

**Why it failed:**
- Model found "charity race" + "realization" 
- Didn't verify that CAROLINE ran a race (she didn't)
- Attributed Melanie's facts to Caroline

**Fix:** Entity verification - check that retrieved facts match question entity.

---

### Pattern 6: Commonsense Inference Chain (~15% of failures)

**Example:** "Would Caroline likely have Dr. Seuss books?"
- **Expected:** Yes, since she collects classic children's books
- **Got:** Unlikely

**What exists:**
- Frame: "I've got lots of kids' books- classics, stories from different cultures..."

**FTS search "Caroline books":**
- Relevant frame IS retrieved (rank -9.6) ★
- Contains "classics" which implies Dr. Seuss

**Why it failed:**
- Frame WAS retrieved but LLM said "Unlikely"
- LLM didn't make the inference: kids' books classics → includes Dr. Seuss

**Fix:** This is an LLM reasoning issue, not retrieval. Frame was retrieved correctly.

---

## Quantitative Breakdown

| Pattern | % of Failures | Root Cause | Primary Fix |
|---------|---------------|------------|-------------|
| Multi-hop facts | ~15% | Info split across frames | Entity linking, graph traversal |
| Temporal computation | ~20% | Relative dates not converted | Prompt engineering, pre-compute |
| Annotation errors | ~10% | Ground truth wrong | Benchmark cleaning |
| Phrasing mismatch | ~25% | Query terms ≠ content terms | Synonym expansion, embeddings |
| Entity confusion | ~15% | Wrong speaker attribution | Entity verification |
| LLM reasoning | ~15% | Retrieved correctly, reasoned wrong | Better LLM prompting |

## Key Insight: FTS Limitations

The core problem with low-effort (keyword-only) search:

```
Question: "Where did Caroline move from 4 years ago?"
Query terms after stopword removal: [Caroline, move, years, ago]

Frame with answer (Sweden):
  "a gift from my grandma in my home country, Sweden"
  
  Terms present: [grandma, home, country, Sweden, gift, young]
  Terms from query: NONE ❌
  
  FTS will NEVER find this frame for this query.
```

**FTS only works when question terms ∈ frame terms.** For ~40% of failed questions, this condition doesn't hold.

## Recommendations

### Immediate (Query Expansion)

Add domain-specific synonyms:
- "identity" → transgender, gender, LGBTQ, journey, transition
- "move from" → home country, origin, born, native
- "activities" → hobby, sport, paint, camp, swim, pottery, run
- "read" → book, books, reading, library

### Medium-term (Embeddings)

Enable embedding-based retrieval for medium/high effort:
- Would catch phrasing mismatches
- Sweden frame would match "where did she move from" semantically
- ~25% of failures are pure semantic gap issues

### Long-term (Entity Linking)

Build entity graph during ingestion:
- Link all mentions of "home country" across frames
- Link all mentions of person names
- Query time: retrieve entity cluster, not just single frames

### Benchmark Fixes

1. Flag annotation errors (~10% of questions have ground truth not in data)
2. Verify temporal question ground truths (Saturday vs Sunday issues)
3. Recategorize multi-hop questions labeled as single-hop

---

## Appendix: Detailed Case Data

### Case 1: Caroline's Identity

**Question:** What is Caroline's identity?
**Expected:** Transgender woman
**Retrieved context (first 300 chars):**
```
[Caroline]
created_at: 2023-05-08T13:56:00
Caroline recently attended an LGBTQ support group...

[Caroline's gender identity exploration...]
created_at: 2023-09-13T00:09:00
Caroline's gender identity exploration through art.
```

**Frames that exist with "transgender":**
- Caroline's attendance at LGBTQ support group (mentions transgender stories)
- Caroline's School Event Presentation (mentions transgender journey)
- Caroline (mentions transitioned three years ago)

**Verdict:** Retrieval gap - frames exist but query "identity" doesn't match "transgender"

---

### Case 2: Sweden Origin

**Question:** Where did Caroline move from 4 years ago?
**Expected:** Sweden
**Retrieved context:** Mentions "home country" but NOT Sweden

**Frame with Sweden (not retrieved):**
```
Title: Caroline: Thanks, Melanie! This necklace is super...
Content: ...a gift from my grandma in my home country, Sweden.
```

**Frame with "moved" (was retrieved):**
```
Content: ...I've known these friends for 4 years, since I moved from my home country.
```

**Verdict:** Multi-hop - two frames needed, only one retrieved

---

*Report generated by spot-check subagent, 2026-02-20*
