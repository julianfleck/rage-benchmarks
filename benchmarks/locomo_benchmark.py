"""LoCoMo benchmark: Test RAGE retrieval + LLM answering on long-conversation QA."""

import sys
import json
import time
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

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

from openai import OpenAI
import os

# LoCoMo category mapping
CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal", 
    3: "commonsense",
    4: "multi-hop",
    5: "adversarial"
}


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score between prediction and ground truth.
    """
    pred_tokens = set(normalize_answer(prediction).split())
    truth_tokens = set(normalize_answer(ground_truth).split())
    
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    
    common = pred_tokens & truth_tokens
    
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Check if normalized prediction exactly matches ground truth."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def load_locomo_data(data_path: Path = None) -> List[Dict[str, Any]]:
    """Load LoCoMo dataset from JSON file."""
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "locomo" / "locomo10.json"
    
    with open(data_path) as f:
        return json.load(f)


def extract_qa_pairs(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract all QA pairs from LoCoMo data.
    
    Returns list of QA dicts with:
        - question: str
        - answer: str
        - evidence: list of dia_ids
        - category: int
        - category_name: str
        - sample_id: str
        - is_adversarial: bool
    
    Note: Category 5 (adversarial) questions use 'adversarial_answer' instead of 'answer'.
    These are trick questions that may be unanswerable or about the wrong speaker.
    """
    qa_pairs = []
    
    for item in data:
        sample_id = item.get("sample_id", "unknown")
        
        for qa in item.get("qa", []):
            # Handle adversarial questions which use 'adversarial_answer'
            answer = qa.get("answer") or qa.get("adversarial_answer", "")
            
            qa_pairs.append({
                "question": qa["question"],
                "answer": str(answer),
                "evidence": qa.get("evidence", []),
                "category": qa["category"],
                "category_name": CATEGORY_NAMES.get(qa["category"], "unknown"),
                "sample_id": sample_id,
                "is_adversarial": "adversarial_answer" in qa
            })
    
    return qa_pairs


class RAGEQARunner:
    """Run QA using RAGE tools and an LLM."""
    
    # System prompt for the LLM
    SYSTEM_PROMPT = """You are a helpful assistant answering questions about conversations between people.

You have access to a memory substrate with the following tools:
- search(query): Search for relevant information
- context(query, effort): Get contextual information with low/medium/high effort  
- find(filter): Find frames matching criteria

Use the tools to find relevant information, then provide a concise answer.
Base your answer ONLY on the information retrieved. If you cannot find relevant information, say "I don't know".

Keep answers brief and factual. Do not include tool calls in your final answer."""

    def __init__(
        self,
        substrate: Substrate,
        model: str = "anthropic/claude-sonnet-4-20250514",
        api_key: str = None,
        verbose: bool = False
    ):
        """
        Initialize QA runner.
        
        Args:
            substrate: RAGE substrate instance
            model: LLM model name for OpenRouter
            api_key: API key (defaults to OPENROUTER_API_KEY env var)
            verbose: Print verbose output
        """
        self.substrate = substrate
        self.model = model
        self.verbose = verbose
        
        api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Define tools for the LLM
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the conversation memory for relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "context",
                    "description": "Get contextual information with specified retrieval effort",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query to get context for"
                            },
                            "effort": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "description": "Retrieval effort level"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a RAGE tool and return result as string."""
        if name == "search":
            result = self.substrate.tools.execute_sync("search", {
                "query": args.get("query", ""),
                "limit": 10
            })
        elif name == "context":
            result = self.substrate.tools.execute_sync("context", {
                "query": args.get("query", ""),
                "effort": args.get("effort", "medium"),
                "limit": 10
            })
        else:
            return f"Unknown tool: {name}"
        
        if not result.success:
            return f"Error: {result.error}"
        
        # Format frames as readable text
        frames = result.data.get("frames", [])
        if not frames:
            return "No relevant information found."
        
        output = []
        for frame in frames[:10]:
            title = frame.get("title", "Untitled")
            content = frame.get("content", "") or frame.get("summary", "")
            output.append(f"[{title}]\n{content}")
        
        return "\n\n".join(output)
    
    def answer_question(self, question: str, max_turns: int = 3) -> Tuple[str, Dict[str, Any]]:
        """
        Answer a question using RAGE tools and LLM.
        
        Args:
            question: The question to answer
            max_turns: Maximum tool call turns
        
        Returns:
            (answer, metadata) tuple
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        
        metadata = {
            "tool_calls": [],
            "tokens_used": 0,
            "turns": 0
        }
        
        for turn in range(max_turns):
            metadata["turns"] = turn + 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            metadata["tokens_used"] += response.usage.total_tokens if response.usage else 0
            
            # Check if we got tool calls
            if message.tool_calls:
                messages.append(message)
                
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    if self.verbose:
                        print(f"    Tool: {func_name}({func_args})")
                    
                    result = self._execute_tool(func_name, func_args)
                    
                    metadata["tool_calls"].append({
                        "name": func_name,
                        "args": func_args,
                        "result_len": len(result)
                    })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                # No more tool calls, we have our answer
                return message.content or "", metadata
        
        # Max turns reached, get final answer
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages + [{"role": "user", "content": "Based on the information retrieved, please provide your final answer."}]
        )
        
        metadata["tokens_used"] += response.usage.total_tokens if response.usage else 0
        return response.choices[0].message.content or "", metadata


def run_benchmark(
    qa_pairs: List[Dict[str, Any]],
    runner: RAGEQARunner,
    limit: int = None,
    categories: List[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run benchmark on QA pairs.
    
    Args:
        qa_pairs: List of QA dicts
        runner: RAGEQARunner instance
        limit: Max number of questions to evaluate
        categories: Filter to specific categories (1-5)
        verbose: Print verbose output
    
    Returns:
        Results dict with per-question and aggregate metrics
    """
    # Filter by category if specified
    if categories:
        qa_pairs = [q for q in qa_pairs if q["category"] in categories]
    
    # Apply limit
    if limit:
        qa_pairs = qa_pairs[:limit]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": runner.model,
        "total_questions": len(qa_pairs),
        "questions": [],
        "by_category": defaultdict(lambda: {"f1_sum": 0, "em_sum": 0, "count": 0})
    }
    
    total_f1 = 0.0
    total_em = 0.0
    
    for i, qa in enumerate(qa_pairs, 1):
        question = qa["question"]
        ground_truth = qa["answer"]
        category = qa["category"]
        category_name = qa["category_name"]
        
        if verbose:
            print(f"\n[{i}/{len(qa_pairs)}] {category_name}: {question}")
        else:
            print(f"[{i}/{len(qa_pairs)}] {category_name}: {question[:50]}...", end=" ", flush=True)
        
        start = time.time()
        predicted, meta = runner.answer_question(question)
        latency_ms = (time.time() - start) * 1000
        
        # Compute metrics
        f1 = compute_f1(predicted, ground_truth)
        em = compute_exact_match(predicted, ground_truth)
        
        if verbose:
            print(f"  Ground truth: {ground_truth}")
            print(f"  Predicted:    {predicted[:100]}...")
            print(f"  F1: {f1:.3f}, EM: {em:.1f}, Latency: {latency_ms:.0f}ms")
        else:
            print(f"F1={f1:.2f}")
        
        # Store result
        q_result = {
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "evidence": qa["evidence"],
            "category": category,
            "category_name": category_name,
            "sample_id": qa["sample_id"],
            "f1": f1,
            "exact_match": em,
            "latency_ms": latency_ms,
            "tool_calls": meta["tool_calls"],
            "tokens_used": meta["tokens_used"]
        }
        results["questions"].append(q_result)
        
        # Update aggregates
        total_f1 += f1
        total_em += em
        
        results["by_category"][category_name]["f1_sum"] += f1
        results["by_category"][category_name]["em_sum"] += em
        results["by_category"][category_name]["count"] += 1
    
    # Compute averages
    n = len(qa_pairs)
    results["avg_f1"] = total_f1 / n if n > 0 else 0
    results["avg_em"] = total_em / n if n > 0 else 0
    
    # Convert defaultdict to regular dict with averages
    by_cat = {}
    for cat_name, stats in results["by_category"].items():
        count = stats["count"]
        by_cat[cat_name] = {
            "avg_f1": stats["f1_sum"] / count if count > 0 else 0,
            "avg_em": stats["em_sum"] / count if count > 0 else 0,
            "count": count
        }
    results["by_category"] = by_cat
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print summary table of results."""
    print("\n" + "=" * 70)
    print("LOCOMO BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Model: {results['model']}")
    print(f"Total questions: {results['total_questions']}")
    print()
    
    # Header
    header = f"{'Category':<15} | {'Count':>6} | {'Avg F1':>8} | {'Avg EM':>8}"
    print(header)
    print("-" * len(header))
    
    # By category
    for cat_name in ["single-hop", "temporal", "commonsense", "multi-hop", "adversarial"]:
        if cat_name in results["by_category"]:
            stats = results["by_category"][cat_name]
            print(f"{cat_name:<15} | {stats['count']:>6} | {stats['avg_f1']:>8.3f} | {stats['avg_em']:>8.3f}")
    
    print("-" * len(header))
    print(f"{'OVERALL':<15} | {results['total_questions']:>6} | {results['avg_f1']:>8.3f} | {results['avg_em']:>8.3f}")
    print("=" * 70)


def save_results(results: Dict[str, Any], output_dir: Path = None):
    """Save results to JSON file."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"locomo_benchmark_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    """Run LoCoMo benchmark."""
    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark on RAGE")
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to locomo10.json"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to substrate database (should be pre-ingested with locomo_ingest.py)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="anthropic/claude-sonnet-4-20250514",
        help="LLM model for answering (OpenRouter format)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of questions to evaluate"
    )
    parser.add_argument(
        "--categories", "-c",
        type=str,
        default=None,
        help="Comma-separated category numbers to test (1-5)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LoCoMo Benchmark")
    print("=" * 70)
    
    # Load QA data
    print("\nLoading LoCoMo data...")
    data_path = Path(args.data) if args.data else None
    data = load_locomo_data(data_path)
    qa_pairs = extract_qa_pairs(data)
    print(f"Loaded {len(qa_pairs)} QA pairs from {len(data)} conversations")
    
    # Parse categories
    categories = None
    if args.categories:
        categories = [int(c.strip()) for c in args.categories.split(",")]
        print(f"Filtering to categories: {categories}")
    
    # Initialize substrate
    if args.db:
        db_path = args.db
    else:
        db_path = str(Path(__file__).parent.parent / "data" / "locomo" / "locomo_substrate.db")
    
    print(f"Using database: {db_path}")
    
    if not Path(db_path).exists():
        print(f"\nError: Database not found at {db_path}", file=sys.stderr)
        print("Run locomo_ingest.py first to create and populate the database.", file=sys.stderr)
        sys.exit(1)
    
    substrate = Substrate("locomo_benchmark", db_path=db_path)
    
    # Initialize QA runner
    print(f"Using model: {args.model}")
    runner = RAGEQARunner(substrate, model=args.model, verbose=args.verbose)
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = run_benchmark(
        qa_pairs,
        runner,
        limit=args.limit,
        categories=categories,
        verbose=args.verbose
    )
    
    substrate.close()
    
    # Print and save results
    print_summary(results)
    
    output_dir = Path(args.output) if args.output else None
    save_results(results, output_dir)
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
