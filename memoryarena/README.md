# MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks

**Paper:** [arXiv:2602.16313](https://arxiv.org/abs/2602.16313) (February 2026)  
**Website:** https://memoryarena.github.io/  
**Data:** [HuggingFace: ZexueHe/memoryarena](https://huggingface.co/datasets/ZexueHe/memoryarena)

## Overview

MemoryArena introduces a paradigm shift in evaluating agent memory by assessing it within **Memory-Agent-Environment loops** rather than isolated recall tasks. Unlike existing benchmarks that test either:
- **Memorization alone** (LoCoMo, LongMemEval) — static recall via QA without action consequences
- **Action alone** (WebArena, SWE-bench) — single-session tasks without long-term memory requirements

MemoryArena evaluates both **simultaneously** through multi-session agentic tasks where:
1. Agents acquire memory through environment interactions
2. Memory must be **distilled and stored** across sessions
3. Later actions depend on **correctly retrieving and applying** that memory

## Key Findings

### LoCoMo-Saturating Agents Fail Here

**Critical result:** Agents achieving near-perfect scores on existing long-context memory benchmarks like LoCoMo exhibit **low task completion rates** in MemoryArena:

| Benchmark Type | LoCoMo Performance | MemoryArena Success Rate |
|----------------|-------------------|-------------------------|
| Long-context agents | ~90%+ | 2-19% average SR |
| Memory agents (MemGPT, Mem0) | High recall | 0-15% average SR |
| RAG systems | Strong retrieval | 0-23% average SR |

This gap reveals that **current memory evaluations don't translate to effective memory use in agentic settings**.

### Why Agents Fail

1. **Belief drift**: Small errors accumulate across sessions (SR@k decays with depth)
2. **Representation mismatch**: External memory returns compressed/reordered info incompatible with in-context learning
3. **Training mismatch**: Memory systems not jointly optimized with task agents
4. **Latency overhead**: External memory adds 30-100% execution time without consistent performance gains

## Benchmark Composition

### Four Evaluation Environments

| Environment | Tasks | Avg Sessions | Avg Steps | Avg Trace Length |
|------------|-------|--------------|-----------|------------------|
| **Bundled Web Shopping** | 150 | 6 | ~42 | 41.5k tokens |
| **Group Travel Planning** | 270 | 5-9 | ~41 | 40.6k tokens |
| **Progressive Web Search** | 256 | 2-16 | ~122 | 122.4k tokens |
| **Formal Reasoning (Math)** | 40 | 2-16 | ~18 | 18.1k tokens |
| **Formal Reasoning (Physics)** | 20 | 2-12 | ~14 | 14.1k tokens |

**Total:** 766 tasks, 6.9 avg subtasks per task, 57 avg steps

### Task Characteristics

#### 1. Bundled Web Shopping
- **Setup**: Multi-session shopping where later purchases depend on attributes of earlier items
- **Dependencies**: Compatibility constraints (e.g., TV size → mount size), preference consistency
- **Data source**: Extended from WebShop with 5 domains (Electronics, Home Decor, Baking, Beauty, Grocery)
- **Creation**: Human-annotated compatibility chains with feature-level filtering

**Example dependency chain:**
```
Session 1: Buy 75-inch TV (specific model)
Session 2: Buy TV mount (must support 70-80" and that model's VESA pattern)
Session 3: Buy HDMI cable (must match TV's HDMI version)
...
```

#### 2. Group Travel Planning
- **Setup**: Base itinerary + sequential travelers with personalized constraints
- **Dependencies**: JOIN constraints (share activity with previous member) or RELATION constraints (e.g., "hotel 2 levels higher rated than Rebecca's")
- **Data source**: Extended from TravelPlanner (45 base trips → 270 group scenarios)
- **Complexity**: Up to 8 additional travelers, dependency chains of depth 4, 30 activity slots per itinerary

**Most challenging environment:** Near-zero success rate for all methods (PS and SR ~0%)

#### 3. Progressive Web Search
- **Setup**: Incremental constraint addition — each session adds a new search condition
- **Dependencies**: Final answer must satisfy ALL previously introduced constraints
- **Data source**: Filtered from BrowseComp-Plus (830 → 256 tasks)
- **Filtering**: Removed single-interaction-solvable queries, enforced strict causal ordering

**Example:**
```
Session 1: "Find a laptop under $1000"
Session 2: "...with at least 16GB RAM" (must remember price constraint)
Session 3: "...from a brand with >4.5 star rating" (must remember both)
```

#### 4. Sequential Formal Reasoning
- **Setup**: Research-level derivation chains from real academic papers
- **Dependencies**: Each lemma/proposition depends on previously established results
- **Creation**: PhD-level experts manually curate papers with long structured derivations
- **Domains**: Math (pure math, optimization, learning theory), Physics (HEP theory/phenom/lattice, condensed matter)

**Quality:** Expert-verified ground-truth answers, far beyond existing math benchmarks (e.g., AIME)

## How It Differs from LoCoMo

| Dimension | LoCoMo | MemoryArena |
|-----------|--------|-------------|
| **Evaluation focus** | Memorization (recall) | Memory-conditioned action |
| **Task structure** | Single QA query per conversation | Multi-session interdependent subtasks |
| **Environment** | Static text | Dynamic agent-environment loops |
| **Success metric** | Factual recall accuracy | Task completion via correct actions |
| **Memory use** | Passive retrieval | Active: store → retrieve → act |
| **Dependencies** | None (independent questions) | Explicit causal chains across sessions |
| **Trace length** | Long conversations (~tokens) | 14-122k tokens (median 40k) |
| **Agent actions** | Not required | Central (avg 57 steps/task) |

**Key difference:** LoCoMo tests "Did you remember X?" — MemoryArena tests "Can you use X to decide Y?"

## Relevance to RAGE

### 1. Multi-Session Task Structure
MemoryArena's session-based evaluation directly maps to RAGE's multi-phase architecture:
- **Subtasks = Phases**: Each subtask is a separate execution session
- **Memory-Agent-Environment loop = RAGE substrate**: Persistent state across phases
- **Interdependencies**: Later phases depend on outcomes/learnings from earlier ones

### 2. POMDP Framing
The paper explicitly frames MemoryArena as a **POMDP testbed** (Section 4.6):
- Agent never observes full task state
- External memory ≈ belief-state estimation mechanism
- Low SR suggests current memory ≠ sufficient statistics for belief tracking

**Connection to RAGE:** Our substrate's addressing system and phase dependencies are designed to maintain belief state — MemoryArena provides empirical evidence for why this matters.

### 3. Evaluation Insights for RAGE

**What we can learn:**
- **Latency matters**: External memory adds 30-100% overhead — RAGE's low-latency addressing design is validated
- **Retrieval > Consolidation**: In tasks requiring precise info reuse (like formal reasoning), RAG systems outperform heavy abstraction/consolidation (like MemGPT)
- **Joint optimization**: Memory mechanisms must be co-designed with task execution, not bolted on
- **Decay with depth**: SR@k analysis shows all current methods degrade with dependency depth — RAGE should track this metric

**Integration opportunity:** Use MemoryArena tasks as RAGE benchmark suite — particularly formal reasoning (aligns with our inductive learning goals).

### 4. Benchmark Integration Opportunities

#### Immediate Integration
- **Formal reasoning tasks** map cleanly to RAGE's inductive phase structure:
  - Each lemma/proposition = one phase
  - Background context = addressing targets from prior phases
  - Verification = phase completion check

#### Harder Integration
- **Bundled shopping / travel planning**: Require external environment simulators (WebShop API, TravelPlanner DB)
- **Progressive search**: Need web search tool integration

#### Hybrid Approach
1. Start with **formal reasoning** (40 math + 20 physics tasks) — pure reasoning, no external deps
2. Abstract the **dependency pattern** from other domains → create RAGE-native tasks with similar interdependence structure
3. Measure **SR@k** (success rate at depth k) as a standard metric

### 5. Missing Pieces in MemoryArena (Where RAGE Can Contribute)

1. **No undo/rollback**: Tasks are linear — RAGE supports non-linear phase exploration
2. **No collaborative memory**: Single agent only — RAGE enables multi-agent shared substrate
3. **Static task structure**: Dependencies are fixed — RAGE allows dynamic phase graphs
4. **Evaluation only**: No training signal — RAGE's substrate enables learning across tasks

## Data Availability

### HuggingFace Datasets

Load via:
```python
from datasets import load_dataset

# Individual domains
ds = load_dataset("ZexueHe/memoryarena", "bundled_shopping")
ds = load_dataset("ZexueHe/memoryarena", "progressive_search")
ds = load_dataset("ZexueHe/memoryarena", "group_travel_planner")
ds = load_dataset("ZexueHe/memoryarena", "formal_reasoning_math")
ds = load_dataset("ZexueHe/memoryarena", "formal_reasoning_phys")
```

### Dataset Schema

Each task is a dict with:
- `id` (int): Unique task identifier
- `questions` (list[str]): Ordered subtask queries
- `answers` (list): Ground-truth answers (format varies by domain)
- `backgrounds` (str | list[str]): Necessary context (domain-specific)

#### Domain-Specific Schemas

**Bundled Shopping & Progressive Search:**
- No backgrounds (context from environment)
- Answers are product ASINs or search results

**Group Travel Planning:**
- Background = base person's full itinerary
- Answers = daily plan dicts (days, city, transportation, meals, attractions, accommodation)

**Formal Reasoning:**
- Backgrounds = list (one per subtask: definitions, lemmas, notation)
- Answers = mathematical results (proofs, derivations)

### License
**Creative Commons Attribution 4.0 International (CC-BY-4.0)**

## Evaluation Code

**Status:** No official code release found as of 2026-02-22.

The paper mentions evaluation harness but no GitHub repository is publicly available yet. The website (https://memoryarena.github.io/) provides dataset access but not implementation code.

**Next steps for RAGE integration:**
1. Monitor for code release (check paper authors' GitHub profiles)
2. Consider implementing RAGE-compatible evaluation harness for formal reasoning tasks
3. Start with HuggingFace data + manual verification against paper's metrics

## Citation

```bibtex
@misc{he2026memoryarenabenchmarkingagentmemory,
  title={MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks},
  author={Zexue He and Yu Wang and Churan Zhi and Yuanzhe Hu and Tzu-Ping Chen and Lang Yin and Ze Chen and Tong Arthur Wu and Siru Ouyang and Zihan Wang and Jiaxin Pei and Julian McAuley and Yejin Choi and Alex Pentland},
  year={2026},
  eprint={2602.16313},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2602.16313}
}
```

## Key Authors & Institutions

- **Zexue He** (lead author)
- **Yejin Choi** (Allen Institute for AI / University of Washington)
- **Alex Pentland** (MIT Media Lab)
- **Julian McAuley** (UC San Diego)
- Multiple authors from top AI research labs and universities

## Additional Resources

- **Paper PDF:** https://arxiv.org/pdf/2602.16313
- **HTML version:** https://arxiv.org/html/2602.16313v1
- **Official website:** https://memoryarena.github.io/
- **Dataset:** https://huggingface.co/datasets/ZexueHe/memoryarena

---

**Last updated:** 2026-02-22  
**Documented by:** Spot (OpenClaw agent)
