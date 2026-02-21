#!/usr/bin/env python3
"""
Run memU LoCoMo benchmark on conv-26 only.

This script wraps the memU-experiment/locomo_test.py to run only on conv-26
(sample index 0) from the LoCoMo benchmark.
"""

import sys
import os
import subprocess
import json
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Paths
BENCHMARK_ROOT = Path(__file__).parent.parent.parent
MEMU_EXPERIMENT = BENCHMARK_ROOT / "external" / "memU-experiment"
LOCOMO_DATA = MEMU_EXPERIMENT / "data" / "locomo10.json"

# OpenRouter configuration (OpenAI-compatible API)
# Set OPENROUTER_API_KEY environment variable or use .env file
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def run_memu_benchmark(
    sample_index: int = 0,  # conv-26 is index 0
    model: str = "gpt-4o-mini",
    memory_dir: str = None,
    max_workers: int = 5,
    category: str = None,
    force_resum: bool = False,
    analyze_on: str = "wrong"
):
    """
    Run memU benchmark using the official locomo_test.py script.

    Args:
        sample_index: Which sample to run (0 = conv-26)
        model: The chat model to use
        memory_dir: Directory for memory files
        max_workers: Number of parallel workers
        category: Category filter (e.g., "1,2,3")
        force_resum: Force regenerate memories
        analyze_on: Analysis mode ("wrong", "all", "none")
    """
    if memory_dir is None:
        memory_dir = str(BENCHMARK_ROOT / "benchmarks" / "memu" / "memory")

    # Ensure memory dir exists
    Path(memory_dir).mkdir(parents=True, exist_ok=True)

    # Ensure data file exists in memU-experiment/data
    data_dir = MEMU_EXPERIMENT / "data"
    data_dir.mkdir(exist_ok=True)

    # Copy locomo data if not present
    if not (data_dir / "locomo10.json").exists():
        src = BENCHMARK_ROOT / "external" / "locomo" / "data" / "locomo10.json"
        if src.exists():
            import shutil
            shutil.copy(src, data_dir / "locomo10.json")
            print(f"Copied locomo10.json to {data_dir}")
        else:
            print(f"ERROR: LoCoMo data not found at {src}")
            return None

    # Build command
    cmd = [
        sys.executable,
        str(MEMU_EXPERIMENT / "locomo_test.py"),
        f"--sample-use=[{sample_index}]",  # locomo_test accepts list notation for specific indices
        f"--chat-deployment={model}",
        f"--memory-dir={memory_dir}",
        f"--max-workers={max_workers}",
        f"--analyze-on={analyze_on}"
    ]

    if category:
        cmd.append(f"--category={category}")

    if force_resum:
        cmd.append("--force-resum")

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Set it via: export OPENROUTER_API_KEY=your-key")
        print("Or add it to a .env file in the project root")
        return None

    print(f"Running memU benchmark...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {MEMU_EXPERIMENT}")
    print()

    # Run the command with OpenRouter configuration
    env = os.environ.copy()
    env["PYTHONPATH"] = str(MEMU_EXPERIMENT) + ":" + env.get("PYTHONPATH", "")
    env["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    env["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL

    result = subprocess.run(
        cmd,
        cwd=str(MEMU_EXPERIMENT),
        env=env,
        capture_output=False  # Stream output
    )

    if result.returncode != 0:
        print(f"\nBenchmark failed with exit code {result.returncode}")
        return None

    # Find and parse the most recent results file
    results_pattern = "enhanced_memory_test_results_*.json"
    results_files = sorted(
        MEMU_EXPERIMENT.glob(results_pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if results_files:
        latest_results = results_files[0]
        print(f"\nResults file: {latest_results}")

        with open(latest_results) as f:
            return json.load(f)

    return None


def print_comparison_table(memu_results: dict):
    """Print a comparison table of results by category."""
    if not memu_results:
        print("No results to display")
        return

    summary = memu_results.get("summary", {})

    print("\n" + "=" * 70)
    print("MEMU LOCOMO BENCHMARK RESULTS (conv-26)")
    print("=" * 70)

    print(f"\nOverall Accuracy: {summary.get('overall_accuracy', 0):.1%}")
    print(f"Total Questions: {summary.get('total_questions', 0)}")
    print(f"Correct: {summary.get('total_correct', 0)}")
    print(f"Total Time: {summary.get('total_time', 0):.1f}s")

    cat_stats = summary.get("category_stats", {})
    cat_acc = summary.get("category_accuracies", {})

    print("\nCategory Breakdown:")
    print("-" * 50)
    print(f"{'Category':<12} {'Correct/Total':<15} {'Accuracy':<12}")
    print("-" * 50)

    # LoCoMo categories:
    # 1: Single-hop QA
    # 2: Multi-hop QA
    # 3: Temporal reasoning
    # 4: Open-domain QA
    # 5: Adversarial QA
    category_names = {
        "1": "Single-hop",
        "2": "Multi-hop",
        "3": "Temporal",
        "4": "Open-domain",
        "5": "Adversarial"
    }

    for cat in sorted(cat_stats.keys(), key=lambda x: int(x)):
        stats = cat_stats[cat]
        acc = cat_acc.get(cat, 0)
        name = category_names.get(cat, f"Cat-{cat}")
        print(f"{name:<12} {stats['correct']}/{stats['total']:<12} {acc:.1%}")

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run memU benchmark on conv-26")
    parser.add_argument("--model", default="gpt-4o-mini", help="Chat model")
    parser.add_argument("--memory-dir", help="Memory directory")
    parser.add_argument("--max-workers", type=int, default=5, help="Parallel workers")
    parser.add_argument("--category", help="Category filter (e.g., '1,2,3')")
    parser.add_argument("--force-resum", action="store_true", help="Force regenerate memories")
    parser.add_argument("--analyze-on", default="wrong", choices=["wrong", "all", "none"])
    args = parser.parse_args()

    results = run_memu_benchmark(
        sample_index=0,  # conv-26
        model=args.model,
        memory_dir=args.memory_dir,
        max_workers=args.max_workers,
        category=args.category,
        force_resum=args.force_resum,
        analyze_on=args.analyze_on
    )

    if results:
        print_comparison_table(results)
