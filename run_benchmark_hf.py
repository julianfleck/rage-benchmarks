"""
RAGE Retrieval Benchmark with HuggingFace Embeddings.

Runs the full benchmark suite using:
- RAGE substrate for RAGE retrieval (unchanged, uses OpenRouter internally)
- HuggingFace sentence-transformers for the baseline (local, no API key)

Usage:
    cd rage-benchmarks
    uv run --directory ../rage-substrate python run_benchmark_hf.py
    
    Or with options:
    uv run --directory ../rage-substrate python run_benchmark_hf.py --limit 10
    uv run --directory ../rage-substrate python run_benchmark_hf.py --efforts medium,high
"""

import sys
import gc
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add rage-substrate to path
RAGE_SUBSTRATE_PATH = Path(__file__).parent.parent / "rage-substrate"
if RAGE_SUBSTRATE_PATH.exists():
    sys.path.insert(0, str(RAGE_SUBSTRATE_PATH))

# Add rage-benchmarks to path
RAGE_BENCHMARKS_PATH = Path(__file__).parent
sys.path.insert(0, str(RAGE_BENCHMARKS_PATH))

try:
    from rage_substrate.core.substrate import Substrate
except ImportError:
    print("Error: Could not import rage_substrate", file=sys.stderr)
    print(f"Tried: {RAGE_SUBSTRATE_PATH}", file=sys.stderr)
    sys.exit(1)

from benchmarks.baseline_rag_hf import BaselineRAGHF
from benchmarks.metrics import evaluate_retrieval


def load_queries(path: Path = None) -> List[Dict[str, Any]]:
    if path is None:
        path = RAGE_BENCHMARKS_PATH / "data" / "queries.json"
    with open(path) as f:
        return json.load(f)


def run_rage_retrieval(query: str, effort: str = "medium", db_path: str = None):
    if db_path is None:
        db_path = str(RAGE_SUBSTRATE_PATH / "substrate.db")
    
    substrate = Substrate("benchmark", db_path=db_path)
    start = time.time()
    result = substrate.tools.execute_sync("context", {
        "query": query,
        "effort": effort,
        "limit": 10
    })
    latency_ms = (time.time() - start) * 1000
    substrate.close()
    
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


def run_baseline_retrieval(baseline: BaselineRAGHF, query: str):
    start = time.time()
    frames = baseline.retrieve_as_frames(query, k=5)
    latency_ms = (time.time() - start) * 1000
    return frames, latency_ms


def run_benchmark(queries, baseline, efforts=None, db_path=None):
    if efforts is None:
        efforts = ["low", "medium", "high"]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "embedding_backend": "huggingface/sentence-transformers/all-MiniLM-L6-v2",
        "efforts_tested": efforts,
        "queries": []
    }
    
    for i, query_spec in enumerate(queries, 1):
        query = query_spec["query"]
        category = query_spec.get("category", "unknown")
        
        print(f"\n[{i}/{len(queries)}] {category}: {query}")
        
        query_result = {"query": query, "category": category}
        
        for effort in efforts:
            print(f"  Running RAGE ({effort})...", end=" ", flush=True)
            rage_frames, rage_latency = run_rage_retrieval(query, effort=effort, db_path=db_path)
            print(f"✓ ({rage_latency:.1f}ms, {len(rage_frames)} frames)")
            
            rage_metrics = evaluate_retrieval(
                rage_frames,
                expected_titles=query_spec.get("expected_frame_titles"),
                expected_contains=query_spec.get("expected_content_contains") or query_spec.get("expected_contains")
            )
            
            query_result[f"rage_{effort}"] = {
                "latency_ms": rage_latency,
                "metrics": rage_metrics,
                "num_frames": len(rage_frames)
            }
            
            print(f"    RAGE({effort}): R@1={rage_metrics.get('recall@1_title', 0):.2f} | "
                  f"R@5={rage_metrics.get('recall@5_title', 0):.2f} | "
                  f"tokens={rage_metrics.get('total_tokens', 0)}")
        
        print("  Running HF baseline...", end=" ", flush=True)
        baseline_frames, baseline_latency = run_baseline_retrieval(baseline, query)
        print(f"✓ ({baseline_latency:.1f}ms)")
        
        baseline_metrics = evaluate_retrieval(
            baseline_frames,
            expected_titles=query_spec.get("expected_frame_titles"),
            expected_contains=query_spec.get("expected_content_contains") or query_spec.get("expected_contains")
        )
        
        query_result["baseline_hf"] = {
            "latency_ms": baseline_latency,
            "metrics": baseline_metrics,
            "num_frames": len(baseline_frames)
        }
        
        print(f"    HF Baseline: R@1={baseline_metrics.get('recall@1_title', 0):.2f} | "
              f"R@5={baseline_metrics.get('recall@5_title', 0):.2f} | "
              f"tokens={baseline_metrics.get('total_tokens', 0)}")
        
        results["queries"].append(query_result)
        gc.collect()
    
    return results


def print_summary_table(results: Dict[str, Any]):
    efforts = results.get("efforts_tested", ["low", "medium", "high"])
    
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS — HuggingFace Embeddings Baseline")
    print(f"Embedding backend: {results.get('embedding_backend', 'unknown')}")
    print("=" * 100 + "\n")
    
    by_category = {}
    for q in results["queries"]:
        cat = q["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(q)
    
    header = f"{'Category':<12} | {'System':<20} | {'Recall@1':<9} | {'Recall@5':<9} | {'Latency':<10} | {'Tokens':<8}"
    print(header)
    print("-" * len(header))
    
    for category in sorted(by_category.keys()):
        queries = by_category[category]
        n = len(queries)
        
        for effort in efforts:
            key = f"rage_{effort}"
            if key not in queries[0]:
                continue
            r1 = sum(q[key]["metrics"].get("recall@1_title", 0) for q in queries) / n
            r5 = sum(q[key]["metrics"].get("recall@5_title", 0) for q in queries) / n
            lat = sum(q[key]["latency_ms"] for q in queries) / n
            tok = sum(q[key]["metrics"].get("total_tokens", 0) for q in queries) / n
            
            label = f"RAGE({effort})"
            cat_label = category if effort == efforts[0] else ""
            print(f"{cat_label:<12} | {label:<20} | {r1:>9.2f} | {r5:>9.2f} | {lat:>8.1f}ms | {int(tok):>8}")
        
        base_r1 = sum(q["baseline_hf"]["metrics"].get("recall@1_title", 0) for q in queries) / n
        base_r5 = sum(q["baseline_hf"]["metrics"].get("recall@5_title", 0) for q in queries) / n
        base_lat = sum(q["baseline_hf"]["latency_ms"] for q in queries) / n
        base_tok = sum(q["baseline_hf"]["metrics"].get("total_tokens", 0) for q in queries) / n
        
        print(f"{'':<12} | {'HF Baseline':<20} | {base_r1:>9.2f} | {base_r5:>9.2f} | {base_lat:>8.1f}ms | {int(base_tok):>8}")
        print()
    
    print("-" * len(header))
    print("OVERALL AVERAGES:")
    all_queries = results["queries"]
    n = len(all_queries)
    
    for effort in efforts:
        key = f"rage_{effort}"
        if key not in all_queries[0]:
            continue
        r1 = sum(q[key]["metrics"].get("recall@1_title", 0) for q in all_queries) / n
        r5 = sum(q[key]["metrics"].get("recall@5_title", 0) for q in all_queries) / n
        lat = sum(q[key]["latency_ms"] for q in all_queries) / n
        tok = sum(q[key]["metrics"].get("total_tokens", 0) for q in all_queries) / n
        
        label = f"RAGE({effort})"
        print(f"{'':<12} | {label:<20} | {r1:>9.2f} | {r5:>9.2f} | {lat:>8.1f}ms | {int(tok):>8}")
    
    base_r1 = sum(q["baseline_hf"]["metrics"].get("recall@1_title", 0) for q in all_queries) / n
    base_r5 = sum(q["baseline_hf"]["metrics"].get("recall@5_title", 0) for q in all_queries) / n
    base_lat = sum(q["baseline_hf"]["latency_ms"] for q in all_queries) / n
    base_tok = sum(q["baseline_hf"]["metrics"].get("total_tokens", 0) for q in all_queries) / n
    
    print(f"{'':<12} | {'HF Baseline':<20} | {base_r1:>9.2f} | {base_r5:>9.2f} | {base_lat:>8.1f}ms | {int(base_tok):>8}")
    print("\n" + "=" * 100 + "\n")


def save_results(results: Dict[str, Any], output_dir: Path = None):
    if output_dir is None:
        output_dir = RAGE_BENCHMARKS_PATH / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"retrieval_hf_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    return output_file


def parse_args():
    parser = argparse.ArgumentParser(description="RAGE Retrieval Benchmark (HuggingFace Embeddings)")
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--efforts", "-e", type=str, default="medium")
    parser.add_argument("--queries", "-q", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--hf-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args()


def main():
    args = parse_args()
    
    efforts = [e.strip() for e in args.efforts.split(",")]
    
    print("=" * 80)
    print("RAGE Retrieval Benchmark — HuggingFace Embeddings")
    print("=" * 80)
    print(f"Effort levels: {', '.join(efforts)}")
    print(f"HF model: {args.hf_model}")
    
    print("\nLoading test queries...")
    queries_path = Path(args.queries) if args.queries else None
    queries = load_queries(queries_path)
    
    if args.limit:
        queries = queries[:args.limit]
        print(f"Limited to first {args.limit} queries")
    
    print(f"Loaded {len(queries)} queries")
    
    print("\nInitializing HuggingFace baseline RAG...")
    baseline = BaselineRAGHF(model_name=args.hf_model)
    baseline.load_from_rage_db(db_path=args.db)
    baseline.embed_chunks()
    
    print("\nRunning benchmark...")
    results = run_benchmark(queries, baseline, efforts=efforts, db_path=args.db)
    
    print_summary_table(results)
    
    output_dir = Path(args.output) if args.output else None
    outfile = save_results(results, output_dir)
    
    print("✓ Benchmark complete!")
    return str(outfile)


if __name__ == "__main__":
    main()
