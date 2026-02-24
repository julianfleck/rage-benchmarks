# MemoryArena Dataset

## Quick Start

The MemoryArena benchmark data is hosted on HuggingFace Datasets and can be loaded directly without manual downloads.

### Installation

```bash
pip install datasets
```

### Loading Data

```python
from datasets import load_dataset

# Load all five domains
bundled_shopping = load_dataset("ZexueHe/memoryarena", "bundled_shopping")
progressive_search = load_dataset("ZexueHe/memoryarena", "progressive_search")
group_travel = load_dataset("ZexueHe/memoryarena", "group_travel_planner")
math_reasoning = load_dataset("ZexueHe/memoryarena", "formal_reasoning_math")
phys_reasoning = load_dataset("ZexueHe/memoryarena", "formal_reasoning_phys")

# Access test split (only split available)
print(f"Bundled shopping tasks: {len(bundled_shopping['test'])}")
print(f"Progressive search tasks: {len(progressive_search['test'])}")
print(f"Group travel tasks: {len(group_travel['test'])}")
print(f"Math reasoning tasks: {len(math_reasoning['test'])}")
print(f"Physics reasoning tasks: {len(phys_reasoning['test'])}")
```

## Dataset Statistics

| Domain | Tasks | Avg Subtasks | Avg Trace Length | HuggingFace Name |
|--------|-------|--------------|------------------|------------------|
| Bundled Web Shopping | 150 | 6 | 41.5k tokens | `bundled_shopping` |
| Progressive Web Search | 256 | 2-16 | 122.4k tokens | `progressive_search` |
| Group Travel Planning | 270 | 5-9 | 40.6k tokens | `group_travel_planner` |
| Formal Reasoning (Math) | 40 | 2-16 | 18.1k tokens | `formal_reasoning_math` |
| Formal Reasoning (Physics) | 20 | 2-12 | 14.1k tokens | `formal_reasoning_phys` |
| **Total** | **766** | 6.9 avg | ~40k median | — |

## Data Schema

### Common Fields

All domains share these core fields:

```python
{
    "id": int,              # Unique task identifier
    "questions": list[str], # Ordered list of subtask queries
    "answers": list,        # Ground-truth answers (format varies by domain)
    "backgrounds": varies   # Context/prerequisites (domain-specific)
}
```

### Domain-Specific Schemas

#### 1. Bundled Web Shopping

```python
{
    "id": 0,
    "questions": [
        "Buy item 1 with constraints...",
        "Buy item 2 compatible with item 1...",
        # ... 6 subtasks total
    ],
    "answers": [
        {"target_asin": "B00TUDFEW2", "attributes": ["Almond Flour", "Gluten-Free", ...]},
        {"target_asin": "B08957C9ZH", "attributes": [...]},
        # ... corresponding answers
    ]
    # No backgrounds field
}
```

**Answer format:**
- `target_asin`: Product identifier in WebShop database
- `attributes`: List of product features used for compatibility checking

**Note:** Requires WebShop environment for execution (product database not included in dataset).

#### 2. Progressive Web Search

```python
{
    "id": 0,
    "questions": [
        "Find X",
        "Find X with constraint Y",
        "Find X with constraints Y and Z",
        # ... progressively more constrained
    ],
    "answers": [
        "Result matching X",
        "Result matching X and Y",
        "Result matching X, Y, and Z",
        # ... each answer satisfies all prior constraints
    ]
    # No backgrounds field
}
```

**Answer format:** Free-form search results (strings)

**Note:** Requires web search capabilities for execution.

#### 3. Group Travel Planning

```python
{
    "id": 0,
    "base_person": {
        "name": "Jennifer",
        "query": "I am Jennifer. Please help me plan a trip from St. Petersburg to Rockford spanning 3 days...",
        "daily_plans": [
            {
                "days": 1,
                "current_city": "from St. Petersburg to Rockford",
                "transportation": "Flight Number: F1234567, from St. Petersburg to Chicago, Departure Time: 08:00, Arrival Time: 10:30",
                "breakfast": "Sunshine Cafe, St. Petersburg",
                "attraction": "Starved Rock State Park, Utica; ...",
                "lunch": "Burger King, Rockford",
                "dinner": "Olympic Tavern, Rockford",
                "accommodation": "Candlewood Suites Rockford, an IHG Hotel, Rockford"
            },
            # ... days 2, 3
        ]
    },
    "questions": [
        "I am Eric.\n I'm joining Jennifer for this trip. [JOIN/RELATION constraints]",
        "I am Emma.\n I'm traveling with Jennifer and Eric. [constraints referencing Eric/Jennifer]",
        # ... 5-8 additional travelers
    ],
    "answers": [
        [  # Eric's complete itinerary
            {"days": 1, "current_city": "...", "transportation": "...", ...},
            {"days": 2, ...},
            {"days": 3, ...}
        ],
        [  # Emma's complete itinerary
            {"days": 1, ...},
            # ...
        ],
        # ... itineraries for all additional travelers
    ]
}
```

**Answer format:** Each answer is a full multi-day itinerary (list of daily plan dicts)

**Background:** The `base_person` field provides the foundation itinerary that subsequent travelers reference.

**Note:** Requires TravelPlanner database (cities, flights, hotels, restaurants, attractions) for execution.

#### 4. Formal Reasoning (Math)

```python
{
    "id": 0,
    "paper_name": "optimal_convergence_learning_theory",
    "backgrounds": [
        "Definition 1: A function f is L-smooth if ...\nLemma 2: For any convex function ...",
        "Notation: Let ∇f(x) denote the gradient of f at x.\nTheorem 3 (from subtask 1): ...",
        # ... one background per subtask, cumulative context
    ],
    "questions": [
        "Prove that under L-smoothness, gradient descent with step size 1/L converges at rate O(1/k).",
        "Using the result from subtask 1, show that the accelerated method achieves O(1/k²).",
        # ... 2-16 interdependent questions
    ],
    "answers": [
        "By L-smoothness: f(x_{k+1}) ≤ f(x_k) + ⟨∇f(x_k), x_{k+1}-x_k⟩ + (L/2)||x_{k+1}-x_k||². Substituting step size η=1/L and using descent lemma yields f(x_k) - f(x_{k+1}) ≥ (1/2L)||∇f(x_k)||². Summing from k=0 to K-1 and rearranging gives min_{k} ||∇f(x_k)||² ≤ 2L(f(x_0)-f*)/K = O(1/K).",
        "From subtask 1, standard GD gives O(1/k). Nesterov acceleration applies momentum correction ...",
        # ... expert-verified mathematical proofs
    ]
}
```

**Background format:** Each subtask gets cumulative context (definitions, prior results, notation)

**Answer format:** Mathematical derivations / proofs (free-form text with LaTeX notation)

**Domains covered:** Pure mathematics, Optimization theory, Learning theory

#### 5. Formal Reasoning (Physics)

Same schema as Math, but with physics content:

**Domains covered:**
- High energy theory (quantum field theory, string theory)
- High energy phenomenology (particle physics, collider predictions)
- High energy lattice (lattice QCD, numerical methods)
- Condensed matter theory

## Example Usage

### Iterating Through Tasks

```python
from datasets import load_dataset

# Load formal math reasoning (easiest to integrate with RAGE)
math_data = load_dataset("ZexueHe/memoryarena", "formal_reasoning_math")

for task in math_data['test']:
    task_id = task['id']
    paper_name = task['paper_name']
    num_subtasks = len(task['questions'])
    
    print(f"\n=== Task {task_id}: {paper_name} ===")
    print(f"Subtasks: {num_subtasks}")
    
    # Process subtasks sequentially (as in MemoryArena eval)
    for i, (question, answer, background) in enumerate(zip(
        task['questions'],
        task['answers'],
        task['backgrounds']
    )):
        print(f"\n--- Subtask {i+1} ---")
        print(f"Background:\n{background[:200]}...")
        print(f"\nQuestion: {question}")
        print(f"\nGround truth: {answer[:100]}...")
```

### Measuring Task Depth

```python
# Compute subtask depth distribution
depths = [len(task['questions']) for task in math_data['test']]
print(f"Min depth: {min(depths)}")
print(f"Max depth: {max(depths)}")
print(f"Mean depth: {sum(depths)/len(depths):.1f}")
```

## Integration with RAGE

### Recommended Starting Point: Formal Reasoning

**Why start here:**
1. ✅ No external environment dependencies (pure reasoning)
2. ✅ Clear subtask structure maps to RAGE phases
3. ✅ Background contexts = addressing targets
4. ✅ Verifiable ground-truth answers
5. ✅ Aligns with RAGE's inductive learning goals

### Mapping to RAGE Concepts

```python
# Conceptual mapping (not actual code)
class MemoryArenaTask:
    def to_rage_phases(self):
        phases = []
        for i, (question, background) in enumerate(zip(self.questions, self.backgrounds)):
            phase = {
                'id': f"{self.paper_name}_phase_{i}",
                'query': question,
                'context_addresses': self.parse_background_refs(background),
                'ground_truth': self.answers[i],
                'dependencies': [f"{self.paper_name}_phase_{j}" for j in range(i)]
            }
            phases.append(phase)
        return phases
```

### Evaluation Metrics for RAGE

Implement these MemoryArena metrics:

1. **Success Rate (SR)**: Fraction of tasks with all subtasks correct
2. **Progress Score (PS)**: Average fraction of correct subtasks per task
3. **SR@k**: Success rate at subtask depth k (decay analysis)

```python
def compute_metrics(predictions, ground_truth):
    """
    predictions: list[list[str]] - one list per task, one str per subtask
    ground_truth: list[list[str]] - same structure
    """
    task_success = []
    task_progress = []
    sr_at_depth = {}
    
    for pred_task, gt_task in zip(predictions, ground_truth):
        # Binary correctness per subtask
        correct = [p == g for p, g in zip(pred_task, gt_task)]
        
        # Task-level metrics
        task_success.append(all(correct))
        task_progress.append(sum(correct) / len(correct))
        
        # SR@k: track success at each depth
        for k in range(1, len(correct) + 1):
            if k not in sr_at_depth:
                sr_at_depth[k] = []
            sr_at_depth[k].append(all(correct[:k]))
    
    return {
        'SR': sum(task_success) / len(task_success),
        'PS': sum(task_progress) / len(task_progress),
        'SR@k': {k: sum(v)/len(v) for k, v in sr_at_depth.items()}
    }
```

## Data Preprocessing for RAGE

### Parsing Mathematical Backgrounds

Formal reasoning backgrounds contain:
- **Definitions**: Extract for substrate fact storage
- **Lemmas/Theorems**: Reference via addressing
- **Notation**: Track for context consistency

Example parser:

```python
import re

def parse_formal_background(background_text):
    """Extract structured components from math/physics background."""
    
    # Extract definitions
    definitions = re.findall(r'Definition \d+:(.*?)(?=\n[A-Z]|\Z)', background_text, re.DOTALL)
    
    # Extract lemmas/theorems (prior results to reference)
    lemmas = re.findall(r'Lemma \d+:(.*?)(?=\n[A-Z]|\Z)', background_text, re.DOTALL)
    theorems = re.findall(r'Theorem \d+:(.*?)(?=\n[A-Z]|\Z)', background_text, re.DOTALL)
    
    # Extract notation
    notation = re.findall(r'Notation:(.*?)(?=\n[A-Z]|\Z)', background_text, re.DOTALL)
    
    return {
        'definitions': [d.strip() for d in definitions],
        'lemmas': [l.strip() for l in lemmas],
        'theorems': [t.strip() for t in theorems],
        'notation': [n.strip() for n in notation]
    }
```

## Known Limitations

1. **No official evaluation code**: Dataset only, no harness released yet
2. **Environment dependencies**: Bundled shopping, travel planning, web search require external simulators
3. **Answer verification**: Some answers are free-form text (harder to auto-verify)
4. **No training split**: Evaluation only (766 test tasks total)

## Download & Cache

HuggingFace datasets caches downloaded data automatically:

```bash
# Default cache location
~/.cache/huggingface/datasets/

# Customize cache directory
export HF_DATASETS_CACHE="/path/to/cache"
```

## License

**Creative Commons Attribution 4.0 International (CC-BY-4.0)**

You are free to:
- ✅ Share — copy and redistribute
- ✅ Adapt — remix, transform, build upon
- ✅ Commercial use allowed

Under the terms:
- **Attribution** — Must give appropriate credit

## Next Steps for RAGE Integration

1. **Prototype formal reasoning evaluator** using this dataset
2. **Implement SR@k metric** to track decay with phase depth
3. **Compare RAGE addressing** vs. external memory systems (Table 3 benchmarks)
4. **Extend to other domains** once environment simulators are available

---

**Dataset source:** https://huggingface.co/datasets/ZexueHe/memoryarena  
**Documentation:** https://memoryarena.github.io/  
**Last updated:** 2026-02-22
