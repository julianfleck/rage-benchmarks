"""Simple retrieval benchmark: RAGE vs Baseline RAG."""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add rage-substrate to path
RAGE_SUBSTRATE_PATH = Path(__file__).parent.parent.parent / "rage-substrate"
if RAGE_SUBSTRATE_PATH.exists():
    sys.path.insert(0, str(RAGE_SUBSTRATE_PATH))

try:
    from rage.retrieval import context
except ImportError:
    print("Error: Could not import rage.retrieval", file=sys.stderr)
    print("Make sure rage-substrate is installed or in ../rage-substrate", file=sys.stderr)
    sys.exit(1)

from benchmarks.baseline_rag import BaselineRAG
from benchmarks.metrics import evaluate_retrieval, count_tokens


def load_queries(path: Path = None) -> List[Dict[str, Any]]:
    """Load test queries from JSON file."""
    if path is None:
        path = Path(__file__).parent.parent / "data" / "queries.json"
    
    with open(path) as f:
        return json.load(f)


def run_rage_retrieval(query: str, effort: str = "medium") -> tuple[List[Dict], float]:
    """
    Run RAGE retrieval and measure latency.
    
    Returns:
        (frames, latency_ms)
    """
    start = time.time()
    
    # Call RAGE context() function
    result = context(query, effort=effort)
    
    latency_ms = (time.time() - start) * 1000
    
    # Parse frames from result
    # context() returns a string with frame content
    # We need to parse it back into frames
    # For now, we'll just count this as a single "frame"
    # TODO: Improve this by actually parsing the markdown output
    frames = [{
        "title": "RAGE Context Result",
        "content": result,
        "source": "rage"
    }]
    
    return frames, latency_ms


def run_baseline_retrieval(baseline: BaselineRAG, query: str) -> tuple[List[Dict], float]:
    """
    Run baseline RAG retrieval and measure latency.
    
    Returns:
        (frames, latency_ms)
    """
    start = time.time()
    frames = baseline.retrieve_as_frames(query, k=5)
    latency_ms = (time.time() - start) * 1000
    
    return frames, latency_ms


def run_benchmark(queries: List[Dict[str, Any]], baseline: BaselineRAG) -> Dict[str, Any]:
    """
    Run benchmark on all queries.
    
    Returns:
        Results dict with per-query and aggregate metrics
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "queries": []
    }
    
    for i, query_spec in enumerate(queries, 1):
        query = query_spec["query"]
        category = query_spec.get("category", "unknown")
        
        print(f"\n[{i}/{len(queries)}] {category}: {query}")
        
        # Run RAGE retrieval
        print("  Running RAGE...", end=" ", flush=True)
        rage_frames, rage_latency = run_rage_retrieval(query)
        print(f"✓ ({rage_latency:.1f}ms)")
        
        # Run baseline retrieval
        print("  Running baseline...", end=" ", flush=True)
        baseline_frames, baseline_latency = run_baseline_retrieval(baseline, query)
        print(f"✓ ({baseline_latency:.1f}ms)")
        
        # Evaluate RAGE
        rage_metrics = evaluate_retrieval(
            rage_frames,
            expected_titles=query_spec.get("expected_frame_titles"),
            expected_contains=query_spec.get("expected_contains")
        )
        
        # Evaluate baseline
        baseline_metrics = evaluate_retrieval(
            baseline_frames,
            expected_titles=query_spec.get("expected_frame_titles"),
            expected_contains=query_spec.get("expected_contains")
        )
        
        # Store results
        query_result = {
            "query": query,
            "category": category,
            "rage": {
                "latency_ms": rage_latency,
                "metrics": rage_metrics,
                "num_frames": len(rage_frames)
            },
            "baseline": {
                "latency_ms": baseline_latency,
                "metrics": baseline_metrics,
                "num_frames": len(baseline_frames)
            }
        }
        
        results["queries"].append(query_result)
        
        # Print quick summary
        print(f"    RAGE:     R@1={rage_metrics.get('recall@1_title', 0):.2f} | "
              f"R@5={rage_metrics.get('recall@5_title', 0):.2f} | "
              f"tokens={rage_metrics.get('total_tokens', 0)}")
        print(f"    Baseline: R@1={baseline_metrics.get('recall@1_title', 0):.2f} | "
              f"R@5={baseline_metrics.get('recall@5_title', 0):.2f} | "
              f"tokens={baseline_metrics.get('total_tokens', 0)}")
    
    return results


def print_summary_table(results: Dict[str, Any]):
    """Print markdown table summary of results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80 + "\n")
    
    # Group by category
    by_category = {}
    for q in results["queries"]:
        cat = q["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(q)
    
    # Print table header
    print(f"{'Category':<12} | {'System':<10} | {'Recall@1':<9} | {'Recall@5':<9} | "
          f"{'Latency':<10} | {'Tokens':<8}")
    print("-" * 80)
    
    # Print rows by category
    for category in sorted(by_category.keys()):
        queries = by_category[category]
        
        # Calculate averages
        rage_r1 = sum(q["rage"]["metrics"].get("recall@1_title", 0) for q in queries) / len(queries)
        rage_r5 = sum(q["rage"]["metrics"].get("recall@5_title", 0) for q in queries) / len(queries)
        rage_lat = sum(q["rage"]["latency_ms"] for q in queries) / len(queries)
        rage_tok = sum(q["rage"]["metrics"].get("total_tokens", 0) for q in queries) / len(queries)
        
        base_r1 = sum(q["baseline"]["metrics"].get("recall@1_title", 0) for q in queries) / len(queries)
        base_r5 = sum(q["baseline"]["metrics"].get("recall@5_title", 0) for q in queries) / len(queries)
        base_lat = sum(q["baseline"]["latency_ms"] for q in queries) / len(queries)
        base_tok = sum(q["baseline"]["metrics"].get("total_tokens", 0) for q in queries) / len(queries)
        
        # Print RAGE row
        print(f"{category:<12} | {'RAGE':<10} | {rage_r1:>9.2f} | {rage_r5:>9.2f} | "
              f"{rage_lat:>8.1f}ms | {int(rage_tok):>8}")
        
        # Print baseline row
        print(f"{'':<12} | {'Baseline':<10} | {base_r1:>9.2f} | {base_r5:>9.2f} | "
              f"{base_lat:>8.1f}ms | {int(base_tok):>8}")
        print()
    
    print("=" * 80 + "\n")


def save_results(results: Dict[str, Any], output_dir: Path = None):
    """Save results to JSON file."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"retrieval_benchmark_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def main():
    """Run the benchmark."""
    print("=" * 80)
    print("RAGE Retrieval Benchmark")
    print("=" * 80)
    
    # Load queries
    print("\nLoading test queries...")
    queries = load_queries()
    print(f"Loaded {len(queries)} queries")
    
    # Initialize baseline
    print("\nInitializing baseline RAG...")
    baseline = BaselineRAG()
    baseline.load_from_rage_db()
    baseline.embed_chunks()
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = run_benchmark(queries, baseline)
    
    # Print summary
    print_summary_table(results)
    
    # Save results
    save_results(results)
    
    print("✓ Benchmark complete!")


if __name__ == "__main__":
    main()
