# Learnings from memU: A Deep Dive into Iterative Memory Architecture

## Executive Summary

After running memU's benchmark independently (87.7% vs their claimed 98.1%), we've analyzed their architecture in detail. While their headline numbers are inflated, the system contains valuable ideas worth adapting for RAGE.

---

## Part 1: How memU Actually Works

### The Three-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 1: MEMORY INGESTION                        │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Conversation │───▶│   Extract    │───▶│  Sufficiency │          │
│  │    Session    │    │   Events     │    │    Check     │          │
│  └──────────────┘    └──────────────┘    └──────┬───────┘          │
│                                                  │                   │
│                           ┌──────────────────────┼──────────────┐   │
│                           │                      ▼              │   │
│                           │  Sufficient? ──NO──▶ Refine        │   │
│                           │       │              Extraction     │   │
│                           │      YES             (up to 3x)     │   │
│                           │       │              │              │   │
│                           │       ▼              └──────────────┘   │
│                           │  Save to:                               │
│                           │  - {Character}_events.txt               │
│                           │  - {Character}_profile.txt              │
│                           └─────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 2: HYBRID RETRIEVAL                        │
│                                                                      │
│  Question ──┬──▶ BM25 Search ──────────┐                            │
│             │                          │                             │
│             ├──▶ String Matching ──────┼──▶ Combine & Dedupe ──▶ Top-K
│             │                          │                             │
│             └──▶ Embedding Search ─────┘                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 3: ITERATIVE QA                            │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Retrieve   │───▶│  Sufficiency │───▶│   Generate   │          │
│  │   Events     │    │    Check     │    │   Answer     │          │
│  └──────────────┘    └──────┬───────┘    └──────────────┘          │
│                             │                                        │
│                             ▼                                        │
│                    Sufficient? ──NO──▶ Generate New Query            │
│                         │              Retrieve More Events          │
│                        YES             (up to 3 iterations)          │
│                         │                                            │
│                         ▼                                            │
│                    Final Answer                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Memory Ingestion (The "Secret Sauce")

memU's ingestion is **aggressively thorough**. For each conversation session:

#### Step 1: Event Extraction
```
Prompt: analyze_session_for_events.txt

Key instructions:
- "Ensure that the information from every message is included"
- "Be comprehensive - capture all events, no matter how minor"
- "Include events related to people who are close to {character}"
- "Preserve exact quotes when they reveal important information"
```

#### Step 2: Sufficiency Check (The Iteration Loop)
```
Prompt: check_events_completeness.txt

The LLM checks:
- "Are all events involving {character_name}, no matter how minor, captured?"
- "Are any important details (who, what, when, where, why, how) missing?"
- "Are the events about people who are close to {character_name} captured?"

Returns: { sufficient: bool, missing_info: string, confidence: float }
```

#### Step 3: Refinement (If Not Sufficient)
```
Prompt: refine_events_extraction.txt

Re-extracts with explicit focus on:
- Previous extraction (to avoid regression)
- Missing information identified by sufficiency check
- Specific instruction: "IMPORTANT: Especially focus on the missing information"
```

**This loops up to 3 times per character per session.**

For conv-26: 19 sessions × 2 characters × ~6 LLM calls = **~228 LLM calls** just for ingestion.

#### Important: Ingestion is Question-Agnostic

The ingestion sufficiency check compares **conversation vs extracted events**—it does NOT have access to benchmark questions. The prompt asks:

> "Please compare the conversation and the extracted events, and determine whether all the events about the speaker included in the extracted events without omission."

This means memU isn't "cheating" by targeting what's needed for specific questions. It's simply being exhaustive about capturing everything from conversations. The benchmark advantage comes from this thoroughness, not from question-aware optimization.

### Phase 2: Hybrid Retrieval

memU uses three retrieval methods in parallel:

```python
# From response_agent.py:_multi_modal_search()

combined_score = (
    0.3 * string_score +      # String matching (exact overlap, substring, fuzzy)
    0.3 * bm25_score +        # BM25 keyword scoring
    2.0 * semantic_score      # Embedding cosine similarity (primary)
)
```

#### BM25 Search
- Standard BM25Okapi implementation
- Tokenizes on whitespace
- Normalized by query length

#### String Matching
- Exact word overlap
- Substring matching
- Fuzzy matching via `difflib.SequenceMatcher`
- Partial word matching

#### Semantic Search
- Uses `text-embedding-ada-002`
- Caches embeddings to avoid repeated API calls
- Cosine similarity scoring

### Phase 3: Iterative Question Answering

This is where it gets interesting. memU doesn't just retrieve once—it **iterates until it has enough information**.

```python
# From response_agent.py:answer_question()

for iteration in range(max_iterations):  # default: 3
    # Search with current query
    search_result = self.search_character_events_profile(current_search_query, ...)

    # Check if we have enough
    sufficiency_result = self._check_content_sufficiency(question, current_content)

    if sufficiency_result.get("sufficient", False):
        break

    # Generate new query based on what's missing
    new_query_result = self._generate_new_query(question, missing_info, current_content)
    current_search_query = new_query_result.get("new_query")
```

#### What Happens When Retrieval is Insufficient?

When `_check_content_sufficiency` returns `sufficient: false`, memU:
1. Extracts the `missing_info` from the response
2. Calls `_generate_new_query(question, missing_info, current_content)`
3. Searches again with the refined query
4. Combines and deduplicates all results
5. Repeats until sufficient or max iterations reached

This is the QA-phase sufficiency check—it compares **question vs retrieved content**, unlike the ingestion check which compares **conversation vs extracted events**.

The `generate_answer` prompt includes explicit reasoning:

```
1. First, use <thinking>...</thinking> tags to analyze:
   - Break down what the question is asking
   - Identify relevant information from the context
   - **IMPORTANT**: Look for multiple answers scattered across different events
   - **IMPORTANT**: Think of the time carefully, find the real time of the event

2. Then, provide your final answer using <result>...</result> tags
```

---

## Part 2: What memU Does Well

### 1. Exhaustive Information Capture

The iterative extraction with sufficiency checking ensures **nothing is lost**. The prompt explicitly says:

> "Ensure that the information from every message is included in at least one output paragraph"

This is brute-force but effective for benchmarks.

### 2. Clear Separation of Concerns

```
memory/
├── Caroline_events.txt      # Timestamped event records
├── Caroline_profile.txt     # Static profile information
├── Melanie_events.txt
└── Melanie_profile.txt
```

Each character has:
- **Events**: Temporal, contextual records ("Mentioned at 1:56 pm on 8 May, 2023: ...")
- **Profile**: Consolidated static facts (cleaned and deduplicated after ingestion)

### 3. Query Refinement During QA

If initial retrieval isn't sufficient, memU generates a new, more targeted query:

```python
# From generate_new_query prompt
"Generate a new search query that will help find the missing information"
```

This is effectively **retrieval-augmented retrieval**.

### 4. Multi-Modal Retrieval

Combining BM25, string matching, and embeddings catches different types of matches:
- BM25: Keyword relevance
- String matching: Exact phrases, entity names
- Embeddings: Semantic similarity

---

## Part 3: What memU Does Poorly

### 1. Extreme Computational Cost

- **~228 LLM calls** for ingestion (one conversation)
- **~597 LLM calls** for QA (199 questions × 3 iterations)
- **~45 minutes** per conversation

This doesn't scale.

### 2. No Structural Intelligence

memU stores flat text files. It cannot:
- Traverse relationships ("Who does Caroline know through Melanie?")
- Find patterns ("What topics come up repeatedly?")
- Surface structural salience (connections, not just content)

### 3. Inflated Published Results

They excluded 45 of 47 adversarial questions from evaluation. More importantly, their published "98.1%" becomes **87.7%** when we run the same code independently—a 10+ percentage point discrepancy that's likely due to cherry-picked runs or model differences.

### 4. Static Storage

No temporal decay, no attention propagation. All memories have equal weight regardless of recency or relevance history.

---

## Part 4: Learnings for RAGE

### Insight 1: Hybrid Retrieval is Better

**Current RAGE**: Single retrieval mode (attention-based)

**memU's approach**: BM25 + String + Embeddings combined

**Recommendation**:
```
Implement hybrid retrieval in RAGE:
- Keep attention/phase resonance as the structural component
- Add BM25 for keyword precision
- Add embedding similarity for semantic matching
- Weighted combination with learned weights
```

### Insight 2: Sufficiency Checking is Valuable

**Current RAGE**: Retrieve once, answer

**memU's approach**: Retrieve → Check sufficiency → Refine query → Retrieve again

**Recommendation**:
```
Add a "sufficiency check" step to RAGE's QA pipeline:
- If initial retrieval looks incomplete, iterate
- Use the LLM to identify what's missing
- Generate refined queries
- This could be the "autonomous mode" vs "fixed mode" distinction
```

### Insight 3: Clearer Memory Organization

**Your thought**:
> "if we could just query /persons/caroline/events and /persons/caroline/profile, it would be much easier for a model to navigate"

**memU's structure**: `{Character}_events.txt`, `{Character}_profile.txt`

**Proposed RAGE structure**:
```
/persons/
├── caroline/
│   ├── events      # Temporal frame references
│   ├── profile     # Static entity facts
│   └── relations   # Connections to other entities
├── melanie/
│   └── ...
/agents/
├── spot/
│   ├── events      # Agent activity log
│   ├── profile     # Agent configuration/persona
│   └── capabilities
└── researcher-xyz/
    └── ...
/topics/
├── lgbtq-support/
│   └── frames      # Related frames
└── painting/
    └── ...
```

This gives the model **predictable paths** to navigate, while maintaining RAGE's graph structure underneath.

### Insight 4: Two-Mode Operation

**Your thought**:
> "we probably need a mixture between low effort mode and autonomous mode: give context on first call already, then let model decide if it needs to dig deeper"

**Proposed implementation**:

```
Mode 1: Quick Mode (Current "fixed" approach)
- Single retrieval pass
- Direct answer
- Use when: Simple queries, time-sensitive responses

Mode 2: Thorough Mode (memU-inspired)
- Retrieve → Check sufficiency → Iterate
- Query refinement if needed
- Use when: Complex questions, accuracy critical

The model should CHOOSE which mode to use:
- Simple factual question → Quick Mode
- Multi-hop reasoning → Thorough Mode
- Temporal reasoning → Thorough Mode with temporal focus
```

### Insight 5: Prompt Engineering Matters

memU's prompts are detailed and specific. Key patterns worth adopting:

1. **Explicit reasoning structure**:
```
<thinking>
[Your step-by-step reasoning process here]
</thinking>

<result>
[Your final answer here]
</result>
```

2. **Comprehensive capture instructions**:
```
"Be comprehensive - capture all events, no matter how minor they seem"
"Ensure that the information from every message is included"
```

3. **Temporal awareness**:
```
"Think of the time carefully, the time of an event can be relative to
when it is mentioned by the speaker, you need to find out the real time"
```

---

## Part 5: Concrete RAGE Improvements

### Improvement 1: Entity-Centric Navigation Layer

Add a navigation layer on top of the hypergraph:

```python
class EntityNavigator:
    def get_person_events(self, person_id: str) -> List[Frame]:
        """Returns all event frames involving this person"""
        pass

    def get_person_profile(self, person_id: str) -> Dict:
        """Returns consolidated profile facts"""
        pass

    def get_agent_history(self, agent_id: str) -> List[Frame]:
        """Returns agent's activity history"""
        pass
```

This provides the **legibility** memU has while maintaining RAGE's **structural power**.

### Improvement 2: Adaptive Retrieval Pipeline

```python
class AdaptiveRetriever:
    def retrieve(self, query: str, mode: str = "auto") -> RetrievalResult:
        if mode == "auto":
            mode = self._estimate_complexity(query)

        if mode == "quick":
            return self._single_pass_retrieval(query)
        elif mode == "thorough":
            return self._iterative_retrieval(query, max_iterations=3)

    def _iterative_retrieval(self, query, max_iterations):
        results = []
        for i in range(max_iterations):
            new_results = self._single_pass_retrieval(query)
            results.extend(new_results)

            if self._is_sufficient(query, results):
                break

            query = self._refine_query(query, results)

        return self._deduplicate(results)
```

### Improvement 3: Hybrid Scoring

```python
def hybrid_score(query: str, frame: Frame) -> float:
    # RAGE's unique contributions
    attention_score = frame.attention_field.resonance_with(query)
    structural_score = frame.graph_centrality()
    temporal_score = frame.temporal_relevance()

    # Borrowed from memU
    bm25_score = bm25_search(query, frame.content)
    embedding_score = cosine_similarity(embed(query), embed(frame.content))

    return (
        0.3 * attention_score +      # RAGE structural
        0.2 * structural_score +     # Graph position
        0.1 * temporal_score +       # Recency
        0.2 * bm25_score +           # Keyword precision
        0.2 * embedding_score        # Semantic similarity
    )
```

### Improvement 4: Root Frame Updates

**Your insight**:
> "whenever we ingest new material, we just need to check if the root frame needs to be updated"

Implementation:

```python
class FrameManager:
    def ingest(self, content: str, source_context: Dict):
        # Extract entities
        entities = self._extract_entities(content)

        for entity in entities:
            # Get or create root frame for entity
            root = self._get_or_create_root(f"/persons/{entity.name}")

            # Check if profile needs updating
            if self._has_new_profile_info(content, entity, root):
                self._update_profile(root, content, entity)

            # Add event frame
            event_frame = self._create_event_frame(content, entity)
            root.connect(event_frame, edge_type="has_event")
```

---

## Part 6: Benchmark Strategy

### What We Learned

1. **memU's 98.1% is unreproducible** - we got 87.7% with identical code
2. **Temporal reasoning is memU's strength** - 100% on category 3
3. **Single-hop is memU's weakness** - 78.1% (vs claimed 93.8%)
4. **Most adversarial questions were excluded**

### RAGE Benchmark Approach

1. **Run RAGE on same 154 questions** - apples-to-apples comparison
2. **Analyze error patterns** - where does RAGE fail that memU succeeds?
3. **Test full 199 questions** - including adversarial (more honest benchmark)
4. **Measure efficiency** - LLM calls, time, cost per question

### Expected RAGE Advantages

- **Structural queries**: "Who knows who through whom"
- **Pattern detection**: Recurring topics, connections
- **Efficiency**: Single-pass should be faster than 3-iteration

### Expected RAGE Weaknesses

- **Exhaustive recall**: memU's brute-force extraction catches more
- **Keyword precision**: Pure attention might miss exact matches

---

## Conclusion

memU achieves high benchmark scores through computational brute force: iterative extraction, iterative retrieval, and iterative answering. It's expensive (~$0.78 per conversation, ~45 minutes) but thorough.

RAGE's graph-based approach offers different strengths: structural intelligence, temporal decay, attention propagation. The opportunity is to **adopt memU's best ideas** (hybrid retrieval, sufficiency checking, clear entity organization) while **maintaining RAGE's unique advantages** (graph structure, efficiency, divergence resistance).

The path forward:
1. Add entity-centric navigation layer (`/persons/`, `/agents/`, `/topics/`)
2. Implement hybrid retrieval (attention + BM25 + embeddings)
3. Add optional iterative mode for complex queries
4. Keep single-pass mode as the fast default
5. Benchmark both systems honestly on full LoCoMo

---

*Document generated from memU source analysis and independent benchmark run on 2026-02-21*
