# Context Gap Analysis: LoCoMo Benchmark (2026-02-21)

## Executive Summary

**The #1 retrieval problem: No embeddings are being used.** 

The LoCoMo substrate has **0% embedding coverage** (0/839 frames have embeddings). All retrieval falls back to FTS5 keyword search, which has fundamental limitations for semantic matching.

### Headline Numbers
- **Total questions:** 199
- **Wrong answers:** 120 (60.3%)
- **Breakdown of failures:**
  - Multi-hop: 39 failures
  - Single-hop: 27 failures  
  - Temporal: 25 failures
  - Adversarial: 20 failures (all 20 adversarial Qs are wrong)
  - Commonsense: 9 failures

## Root Cause Analysis

### 1. No Semantic Search (Critical)

```sql
-- Query run on locomo_substrate.db
SELECT COUNT(*) FROM frames WHERE embedding IS NOT NULL;
-- Result: 0

SELECT COUNT(*) FROM frames WHERE embedding_summary IS NOT NULL;
-- Result: 0
```

**Impact:** All retrieval relies on keyword matching via FTS5. This means:
- "Sweden" only matches frames containing "Sweden" - not "home country"
- "What does Melanie think about adoption" won't find "You'll be an awesome mom!"
- Synonyms, paraphrases, and semantic similarity are completely missed

### 2. Missing Graph Traversal

Related frames are not connected via edges:

```
Caroline's adoption decision frame: frm_3a8c4a1eb24b890f
Edges FROM this frame: (none)
Edges TO this frame: (none)

Melanie's response "You're doing something amazing! You'll be an awesome mom!": frm_892cfafda3926496
Parent: /locomo/conv-26 (generic conversation territory)
```

**Impact:** Multi-hop questions fail because:
- "What does Melanie think about Caroline's decision?" needs to traverse from decision → response
- No explicit edges connect these semantically related frames
- FTS search for "Melanie think adoption" doesn't match "awesome mom" frame

### 3. Temporal Slot Extraction Errors

The extraction pipeline incorrectly computes dates from relative references:

| Question | Ground Truth | Predicted | Error |
|----------|-------------|-----------|-------|
| When did Melanie run a charity race? | Sunday before May 25 (= May 21) | Saturday May 20 | Off by 1 day |
| When did Caroline give speech at school? | Week before June 9 | June 2 | Frame has `startDate: 2023-06-02` (correct) but GT expects "week before June 9" format |
| When did Caroline meet friends/family/mentors? | Week before June 9 | June 29 | Retrieved WRONG frame (picnic frame from July context) |

**Analysis:** 
- Original message: "I ran a charity race for mental health last Saturday" (sent May 25)
- Frame slot: `startDate: 2023-05-20` (Saturday)
- Ground truth expects: "Sunday before May 25" = May 21
- This suggests either benchmark GT error OR extraction miscalculated day-of-week

### 4. Entity Confusion in Adversarial Questions

Adversarial questions ask about the WRONG person (e.g., "What did Melanie's necklace symbolize?" when only Caroline has a necklace). The model should say "UNANSWERABLE" but instead gives the adversarial answer:

| Question | Should Answer | Model Said | Result |
|----------|---------------|------------|--------|
| What type of adoption agency is Melanie considering? | UNANSWERABLE | LGBTQ+ individuals | ❌ Gave adversarial answer |
| What does Melanie's necklace symbolize? | UNANSWERABLE | love, faith, and strength | ❌ Gave Caroline's answer |
| What country is Melanie's grandma from? | UNANSWERABLE | Sweden (also mentioned Caroline's) | ❌ Confused entities |
| What was grandpa's gift to Caroline? | UNANSWERABLE | necklace | ❌ Wrong family member |

**Why this happens:**
- FTS search for "Melanie necklace" still returns frames mentioning both Melanie AND necklace (even if unrelated)
- No entity linking to filter "necklace" facts TO the correct person
- The context contains frames about Caroline's necklace, so model assumes it applies to Melanie

---

## Detailed Case Studies

### Case 1: Where did Caroline move from 4 years ago?

**Ground Truth:** Sweden  
**Predicted:** "Caroline moved from her home country 4 years ago."  
**LLM Score:** 0 (wrong - didn't name Sweden)

**What's in the substrate:**
```
frm_3e684a1389bfb451: "Thanks, Melanie! This necklace is super special to me - 
a gift from my grandma in my home country, Sweden."
```

**What was retrieved:**
1. Caroline's Support System - mentions "home country" but NOT "Sweden"
2. General messages about Caroline

**Why retrieval failed:**
- FTS query "Where did Caroline move from 4 years ago" matches "move" and "4 years"
- The Sweden frame mentions "home country" but not "move" or "4 years" explicitly
- Semantic search would have linked "move from" ↔ "home country" ↔ "Sweden"

---

### Case 2: What does Melanie think about Caroline's decision to adopt?

**Ground Truth:** "she thinks Caroline is doing something amazing and will be an awesome mom"  
**Predicted:** "The context does not provide any specific statement from Melanie..."  
**LLM Score:** 0

**What's in the substrate:**
```
frm_892cfafda3926496: "You're doing something amazing! Creating a family 
for those kids is so lovely. You'll be an awesome mom! Good luck!"
```

**What was retrieved:**
1. Caroline's Decision to Adopt
2. General Melanie messages
3. Various unrelated messages

**Why retrieval failed:**
- Query "What does Melanie think about Caroline's decision to adopt" 
- FTS matches "Melanie", "Caroline", "adopt"
- The answer frame has "amazing", "family", "mom" - no overlap with query terms!
- No graph traversal: decision frame → response frame
- Would need: follow conversation thread from adoption mention to Melanie's reply

---

### Case 3: Where has Melanie camped?

**Ground Truth:** beach, mountains, forest  
**Predicted:** "The specific location where Melanie camped is not mentioned in the context."  
**LLM Score:** 0

**What's in the substrate:**
```
frm_50f6c30fb947da11: "Melanie and her family went on a camping trip in the forest."
frm_c5980a7d2266fc92: "Melanie enjoys hiking in the mountains and exploring forests"  
/events/melanies-beach-trip: (territory exists but content not retrieved)
```

**Why retrieval failed:**
- Information is SCATTERED across multiple frames
- FTS "Where has Melanie camped" matches "Melanie" and "camp" in some frames
- But "beach" is in a different frame than "forest"
- No aggregation across related frames
- Multi-hop: needs to collect facts from 3+ frames about same topic

---

### Case 4: Adversarial - What does Melanie's necklace symbolize?

**Ground Truth:** UNANSWERABLE (Melanie has no necklace - it's Caroline's)  
**Adversarial Answer:** love, faith, and strength  
**Predicted:** "Melanie's necklace symbolizes love, faith, and strength."  
**LLM Score:** 0

**What's in the substrate:**
```
frm_3e684a1389bfb451: "This necklace is super special to me - a gift from 
my grandma in my home country, Sweden. She gave it to me when I was young, 
and it stands for love, faith and strength."
-- Author: Caroline (in slots)
```

**Why model failed:**
- FTS query "Melanie necklace" returns frames with "necklace"
- The necklace frame mentions Caroline, not Melanie
- But FTS doesn't understand "this frame is about Caroline's necklace, not Melanie's"
- No entity-bound slot filtering: `necklace.owner = Caroline`
- Model sees necklace facts in context, assumes they apply to query subject

---

## Pattern Summary

| Pattern | Count | Examples |
|---------|-------|----------|
| Missing semantic match (no embeddings) | ~50% | Sweden/home country, awesome mom/think about adoption |
| Multi-hop traversal needed | ~25% | Melanie's opinion on adoption, camping locations |
| Entity confusion (adversarial) | 100% of adversarial | All 20 adversarial questions wrong |
| Temporal slot errors | ~15% | Charity race date, speech date |
| Information scattered across frames | ~10% | Camping locations in 3 frames |

---

## Recommendations (Priority Order)

### 1. Generate Embeddings (Critical - blocks everything)

```bash
# Run embedding generation for all frames
cd rage-substrate
python -m rage_substrate.tasks.generate_embeddings \
  --db ~/.openclaw/workspace/repos/rage-benchmarks/data/locomo/locomo_substrate.db \
  --batch-size 50
```

**Expected impact:** 30-40% accuracy improvement from semantic search alone.

### 2. Entity-Scoped Retrieval

When query mentions "Melanie", filter frames where:
- `slots.author = "Melanie"` OR
- `slots.attendee` contains Melanie OR  
- `slots.subject` is Melanie

For adversarial questions: if question asks about Person X, and retrieved frame is about Person Y, mark as low confidence or exclude.

### 3. Conversation Thread Traversal  

For multi-hop questions:
1. Find initial match (Caroline's adoption decision)
2. Traverse to parent message
3. Find sibling/child messages from other speakers
4. Include responses in context

```python
# Pseudocode for thread-aware retrieval
decision_frame = find("Caroline adoption decision")
parent = get_parent(decision_frame)  # /locomo/conv-26 or specific session
related = find_siblings(parent, speaker="Melanie")  # Melanie's responses in same thread
```

### 4. Temporal Query Enhancement

When query contains temporal words ("when", "what date"):
1. Extract temporal references from query
2. Boost frames with `startDate` or `timestamp` slots
3. Return slot values directly, not just frame content

### 5. Frame Aggregation for Scattered Facts

For aggregate queries ("Where has X done Y?"):
1. Detect aggregate intent (list all places, events, etc.)
2. Retrieve multiple frames per entity
3. Merge related facts before answering

---

## Quick Wins vs Strategic Fixes

### Quick Wins (Do Now)
- [ ] Generate embeddings for all 839 frames (~10 min job)
- [ ] Re-run benchmark with embeddings → measure improvement
- [ ] Add entity filter when query names specific person

### Strategic Fixes (Phase 2)
- [ ] Build conversation thread graph during ingestion
- [ ] Add entity linking to bind facts to people
- [ ] Implement slot-aware retrieval for structured queries
- [ ] Create adversarial detection (question asks about X, answer is about Y)

---

## Verification Queries

To verify the embedding issue is fixed:

```sql
-- After embedding generation
SELECT COUNT(*) FROM frames WHERE embedding IS NOT NULL;
-- Should be: 839 (100%)

-- Check coverage
SELECT frame_type, COUNT(*), 
       SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embedding
FROM frames 
GROUP BY frame_type;
```

---

## Appendix: Database Schema (Relevant Tables)

```sql
frames (839 rows):
  - id, title, content, summary, slots (JSON)
  - embedding (NULL - this is the problem!)
  - embedding_summary (NULL)
  - frame_type, parent_id, territory

edges (105 rows):
  - source_id, target_id, edge_type
  -- Very sparse! Most frames not connected

couplings (10,484 rows):
  - frame_a, frame_b, coupling_type, strength
  -- Semantic relationships exist but avg strength = 0.105 (weak)
```
