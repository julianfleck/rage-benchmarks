"""Simple retrieval benchmark: RAGE vs Baseline RAG."""

import sys
import gc
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add rage-substrate to path
RAGE_SUBSTRATE_PATH = Path(__file__).parent.parent.parent / "rage-substrate"
if RAGE_SUBSTRATE_PATH.exists():
    sys.path.insert(0, str(RAGE_SUBSTRATE_PATH))

try:
    from rage_substrate.core.substrate import Substrate
except ImportError:
    print("Error: Could not import rage_substrate", file=sys.stderr)
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


def run_rage_retrieval(query: str, effort: str = "low", db_path: str = None) -> tuple[List[Dict], float]:
    """
    Run RAGE retrieval and measure latency.
    
    Args:
        query: The search query
        effort: Effort level - "low", "medium", or "high"
        db_path: Path to substrate database
    
    Returns:
        (frames, latency_ms)
    """
    if db_path is None:
        db_path = str(Path(__file__).parent.parent.parent / "rage-substrate" / "substrate.db")
    
    substrate = Substrate("benchmark", db_path=db_path)
    
    start = time.time()
    
    # Use tools.execute_sync for context
    result = substrate.tools.execute_sync("context", {
        "query": query,
        "effort": effort,
        "limit": 10
    })
    
    latency_ms = (time.time() - start) * 1000
    
    substrate.close()
    
    # Parse frames from result
    frames = []
    if result.success and result.data:
        for frame in result.data.get("frames", []):
            frames.append({
                "title": frame.get("title", ""),
                "content": frame.get("content", frame.get("summary", "")),
                "similarity": frame.get("similarity", 0),
                "source": "rage"
            })
    
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


def run_benchmark(
    queries: List[Dict[str, Any]], 
    baseline: BaselineRAG,
    efforts: List[str] = None
) -> Dict[str, Any]:
    """
    Run benchmark on all queries, testing each effort level.
    
    Args:
        queries: List of query specifications
        baseline: Initialized baseline RAG system
        efforts: List of effort levels to test (default: ["low", "medium", "high"])
    
    Returns:
        Results dict with per-query and aggregate metrics
    """
    if efforts is None:
        efforts = ["low", "medium", "high"]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "efforts_tested": efforts,
        "queries": []
    }
    
    for i, query_spec in enumerate(queries, 1):
        query = query_spec["query"]
        category = query_spec.get("category", "unknown")
        
        print(f"\n[{i}/{len(queries)}] {category}: {query}")
        
        query_result = {
            "query": query,
            "category": category,
        }
        
        # Run RAGE retrieval for each effort level
        for effort in efforts:
            print(f"  Running RAGE ({effort})...", end=" ", flush=True)
            rage_frames, rage_latency = run_rage_retrieval(query, effort=effort)
            print(f"✓ ({rage_latency:.1f}ms)")
            
            # Evaluate RAGE
            rage_metrics = evaluate_retrieval(
                rage_frames,
                expected_titles=query_spec.get("expected_frame_titles"),
                expected_contains=query_spec.get("expected_contains")
            )
            
            query_result[f"rage_{effort}"] = {
                "latency_ms": rage_latency,
                "metrics": rage_metrics,
                "num_frames": len(rage_frames)
            }
            
            # Print quick summary
            print(f"    RAGE({effort}): R@1={rage_metrics.get('recall@1_title', 0):.2f} | "
                  f"R@5={rage_metrics.get('recall@5_title', 0):.2f} | "
                  f"tokens={rage_metrics.get('total_tokens', 0)}")
        
        # Run baseline retrieval (once per query)
        print("  Running baseline...", end=" ", flush=True)
        baseline_frames, baseline_latency = run_baseline_retrieval(baseline, query)
        print(f"✓ ({baseline_latency:.1f}ms)")
        
        # Evaluate baseline
        baseline_metrics = evaluate_retrieval(
            baseline_frames,
            expected_titles=query_spec.get("expected_frame_titles"),
            expected_contains=query_spec.get("expected_contains")
        )
        
        query_result["baseline"] = {
            "latency_ms": baseline_latency,
            "metrics": baseline_metrics,
            "num_frames": len(baseline_frames)
        }
        
        print(f"    Baseline: R@1={baseline_metrics.get('recall@1_title', 0):.2f} | "
              f"R@5={baseline_metrics.get('recall@5_title', 0):.2f} | "
              f"tokens={baseline_metrics.get('total_tokens', 0)}")
        
        results["queries"].append(query_result)
        
        # Memory cleanup between queries
        gc.collect()
    
    return results


def print_summary_table(results: Dict[str, Any]):
    """Print markdown table summary of results."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100 + "\n")
    
    efforts = results.get("efforts_tested", ["low", "medium", "high"])
    
    # Group by category
    by_category = {}
    for q in results["queries"]:
        cat = q["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(q)
    
    # Print table header
    header = f"{'Category':<12} | {'System':<14} | {'Recall@1':<9} | {'Recall@5':<9} | {'Latency':<10} | {'Tokens':<8}"
    print(header)
    print("-" * len(header))
    
    # Print rows by category
    for category in sorted(by_category.keys()):
        queries = by_category[category]
        n = len(queries)
        
        # Calculate averages for each effort level
        for effort in efforts:
            key = f"rage_{effort}"
            r1 = sum(q[key]["metrics"].get("recall@1_title", 0) for q in queries) / n
            r5 = sum(q[key]["metrics"].get("recall@5_title", 0) for q in queries) / n
            lat = sum(q[key]["latency_ms"] for q in queries) / n
            tok = sum(q[key]["metrics"].get("total_tokens", 0) for q in queries) / n
            
            label = f"RAGE({effort})"
            cat_label = category if effort == efforts[0] else ""
            print(f"{cat_label:<12} | {label:<14} | {r1:>9.2f} | {r5:>9.2f} | {lat:>8.1f}ms | {int(tok):>8}")
        
        # Baseline
        base_r1 = sum(q["baseline"]["metrics"].get("recall@1_title", 0) for q in queries) / n
        base_r5 = sum(q["baseline"]["metrics"].get("recall@5_title", 0) for q in queries) / n
        base_lat = sum(q["baseline"]["latency_ms"] for q in queries) / n
        base_tok = sum(q["baseline"]["metrics"].get("total_tokens", 0) for q in queries) / n
        
        print(f"{'':<12} | {'Baseline':<14} | {base_r1:>9.2f} | {base_r5:>9.2f} | {base_lat:>8.1f}ms | {int(base_tok):>8}")
        print()
    
    # Print overall averages
    print("-" * len(header))
    print("OVERALL AVERAGES:")
    all_queries = results["queries"]
    n = len(all_queries)
    
    for effort in efforts:
        key = f"rage_{effort}"
        r1 = sum(q[key]["metrics"].get("recall@1_title", 0) for q in all_queries) / n
        r5 = sum(q[key]["metrics"].get("recall@5_title", 0) for q in all_queries) / n
        lat = sum(q[key]["latency_ms"] for q in all_queries) / n
        tok = sum(q[key]["metrics"].get("total_tokens", 0) for q in all_queries) / n
        
        label = f"RAGE({effort})"
        print(f"{'':<12} | {label:<14} | {r1:>9.2f} | {r5:>9.2f} | {lat:>8.1f}ms | {int(tok):>8}")
    
    base_r1 = sum(q["baseline"]["metrics"].get("recall@1_title", 0) for q in all_queries) / n
    base_r5 = sum(q["baseline"]["metrics"].get("recall@5_title", 0) for q in all_queries) / n
    base_lat = sum(q["baseline"]["latency_ms"] for q in all_queries) / n
    base_tok = sum(q["baseline"]["metrics"].get("total_tokens", 0) for q in all_queries) / n
    
    print(f"{'':<12} | {'Baseline':<14} | {base_r1:>9.2f} | {base_r5:>9.2f} | {base_lat:>8.1f}ms | {int(base_tok):>8}")
    
    print("\n" + "=" * 100 + "\n")


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RAGE Retrieval Benchmark")
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of queries to run (default: all)"
    )
    parser.add_argument(
        "--efforts", "-e",
        type=str,
        default="low,medium,high",
        help="Comma-separated effort levels to test (default: low,medium,high)"
    )
    parser.add_argument(
        "--queries", "-q",
        type=str,
        default=None,
        help="Path to queries JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    return parser.parse_args()


def main():
    """Run the benchmark."""
    args = parse_args()
    
    # Parse effort levels
    efforts = [e.strip() for e in args.efforts.split(",")]
    valid_efforts = {"low", "medium", "high"}
    for e in efforts:
        if e not in valid_efforts:
            print(f"Error: Invalid effort level '{e}'. Must be one of: {valid_efforts}", file=sys.stderr)
            sys.exit(1)
    
    print("=" * 80)
    print("RAGE Retrieval Benchmark")
    print("=" * 80)
    print(f"Effort levels: {', '.join(efforts)}")
    
    # Load queries
    print("\nLoading test queries...")
    queries_path = Path(args.queries) if args.queries else None
    queries = load_queries(queries_path)
    
    # Apply limit
    if args.limit:
        queries = queries[:args.limit]
        print(f"Limited to first {args.limit} queries")
    
    print(f"Loaded {len(queries)} queries")
    
    # Initialize baseline
    print("\nInitializing baseline RAG...")
    baseline = BaselineRAG()
    baseline.load_from_rage_db()
    baseline.embed_chunks()
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = run_benchmark(queries, baseline, efforts=efforts)
    
    # Print summary
    print_summary_table(results)
    
    # Save results
    output_dir = Path(args.output) if args.output else None
    save_results(results, output_dir)
    
    print("✓ Benchmark complete!")


if __name__ == "__main__":
    main()
