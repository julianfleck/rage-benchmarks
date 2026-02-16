# RAGE Benchmarks

Simple retrieval benchmarks comparing RAGE (Retrieval-Augmented Graph Embeddings) against vanilla RAG baselines.

## Purpose

RAGE claims to provide context-aware retrieval through:
- Temporal filtering (recent frames get higher relevance)
- Structural traversal (following parent/child relationships)
- Multi-hop reasoning across the knowledge graph

This benchmark suite tests whether those claims hold against a simple baseline: chunking + embedding + cosine similarity.

## What We Measure

For each test query:
- **Recall@1, Recall@5**: Did the system retrieve the expected frames?
- **Latency (ms)**: How long did retrieval take?
- **Tokens used**: How many tokens were consumed?

## Benchmark Categories

- **Direct**: Simple factual queries ("What is X?")
- **Temporal**: Time-sensitive queries ("What happened yesterday?")
- **Structural**: Graph-aware queries ("What is the parent of X?")

## Setup

```bash
# Clone the repo
git clone https://github.com/julianfleck/rage-benchmarks
cd rage-benchmarks

# Install dependencies
pip install -e .

# Ensure rage-substrate is available (local or installed)
# The benchmark will use the local RAGE database
```

## Running Benchmarks

```bash
# Run the simple retrieval benchmark
python -m benchmarks.simple_retrieval

# Output goes to stdout (markdown table) and results/*.json
```

## Expected Output

```
Category   | System       | Recall@1 | Recall@5 | Latency (ms) | Tokens
-----------|-------------|----------|----------|--------------|-------
direct     | RAGE        | 1.00     | 1.00     | 45.2         | 1234
direct     | Baseline    | 0.67     | 1.00     | 12.3         | 890
temporal   | RAGE        | 1.00     | 1.00     | 52.1         | 1456
temporal   | Baseline    | 0.00     | 0.33     | 15.7         | 920
structural | RAGE        | 1.00     | 1.00     | 48.9         | 1312
structural | Baseline    | 0.33     | 0.67     | 13.4         | 905
```

## Hypothesis

We expect:
- RAGE and baseline perform similarly on **direct** queries (simple keyword matching works)
- RAGE significantly outperforms on **temporal** queries (baseline has no time awareness)
- RAGE outperforms on **structural** queries (baseline can't traverse relationships)

## Test Data

See `data/queries.json` for the test queries and expected results.

Queries are intentionally simple to isolate retrieval quality from LLM reasoning.

## Baseline Implementation

The vanilla RAG baseline (`benchmarks/baseline_rag.py`):
- Chunks all frame content into 500-token segments
- Embeds with the same model RAGE uses (OpenAI text-embedding-3-small)
- Returns top-k by cosine similarity
- No temporal filtering, no graph traversal, no context fusion

## Future Work

- [ ] Add more query categories (multi-hop, negation, etc.)
- [ ] Test with larger corpora
- [ ] Add retrieval latency percentiles (p50, p95, p99)
- [ ] Compare different embedding models
- [ ] Add ablation tests (RAGE without temporal, RAGE without traversal, etc.)

## License

MIT
