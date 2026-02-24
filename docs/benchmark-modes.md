# LoCoMo Benchmark Modes Overview

## Benchmark Modes

The benchmark supports 5 modes that control how context is retrieved for answering questions.

### 1. `autonomous`
**Full LLM autonomy** — Model can call any tools freely.

- LLM decides which tools to use: `context()`, `traverse()`, `get()`, etc.
- Can make multiple queries with different phrasings
- Can traverse to related frames
- Up to 10 turns of tool use
- Most expensive but most flexible

### 2. `fixed-low`
**Single context call, keyword search only**

```
context(query=question, effort="low")
```

- Fast (~200ms)
- Keyword/FTS search only — no embeddings
- NLP query expansion (extracts key phrases from question)
- Good baseline for speed

### 3. `fixed-medium`
**Single context call, hybrid search**

```
context(query=question, effort="medium")
```

- Balanced (~500ms)
- Hybrid retrieval: semantic embeddings + keyword search
- NLP query expansion
- RRF fusion across query variations
- Coupling + parent expansion

### 4. `fixed-high`
**Single context call, full SPIRAL**

```
context(query=question, effort="high")
```

- Thorough (~2s)
- Full SPIRAL pipeline with LOOP phase
- Hierarchy expansion (traverses up/down)
- Phase resonance reranking
- Same query expansion as medium, but more traversal

### 5. `fixed-low-traverse`
**Low context + deterministic traversal**

```
context(query=question, effort="low")
→ traverse(frame_id, direction="up")
→ traverse(frame_id, direction="down")
```

- Adds fixed traversal after initial retrieval
- Cheaper than high but explores hierarchy
- Deterministic (no LLM decisions)

---

## Effort Levels (in SPIRAL)

The `context()` tool's effort parameter controls retrieval depth:

| Effort | Search Type | Query Expansion | Traversal | Target Time |
|--------|-------------|-----------------|-----------|-------------|
| low    | Keyword/FTS only | NLP phrases | None | ~200ms |
| medium | Hybrid (semantic + keyword) | NLP phrases | Coupling + parents | ~500ms |
| high   | Hybrid + LOOP | NLP phrases | Full hierarchy | ~2s |

### Common Pipeline (all levels)
1. **Preprocess query** — remove metadata, markdown
2. **NLP query expansion** — extract key phrases (fast, ~50ms)
3. **Session intent** — build context from query history
4. **Attention weighting** — boost based on recency/relevance
5. **Unified reranking** — type boosts, fuzzy matching, phase resonance
6. **Deduplication + token budget**

### What Each Level Adds
- **Medium over Low**: Embedding-based semantic search, RRF fusion
- **High over Medium**: Hierarchy expansion (LOOP), deeper traversal, phase resonance reranking (0.4 weight)

---

## Current Problem: Retrieval Failures

Analysis shows **80% of benchmark failures are retrieval failures** — the right frames aren't found at any effort level.

### What's NOT the problem
- Effort level (high doesn't help much over medium)
- Traversal (fixed-low-traverse doesn't fix it)
- Comprehension (only 20% of failures)

### Hypotheses for retrieval failures
1. **Query-embedding mismatch** — Question phrasing doesn't match how frame content was embedded
2. **NLP expansion insufficient** — Phrase extraction misses key terms
3. **Frame content gaps** — Information exists but in unexpected form
4. **Slot data not searchable** — Structured slot values (names, dates) not in embedding or keyword index

---

## Ideas to Test

### Multi-query at benchmark level
Instead of single `context(question)`, generate 3-4 rephrasings:
- Original question
- Entity-focused ("Caroline")  
- Verb-focused ("job", "work")
- Synonym variation

### Slot-aware search
Add slot values to searchable content:
```
Content: "Caroline works as a nurse"
Slots: {person: "Caroline", occupation: "nurse"}
Searchable: "Caroline works as a nurse [person:Caroline occupation:nurse]"
```

### Filter by entity
If question mentions "Caroline", add filter:
```
context(query=question, filter="/persons/caroline")
```

### Temporal-aware retrieval
For temporal questions, ensure `created_at` propagates correctly and is searchable.
