"""LLM-based exploration benchmark: Test LLM tool calling with RAGE substrate."""

import sys
import json
import time
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import httpx

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


# OpenRouter configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"


def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Get your key at https://openrouter.ai/keys"
        )
    return key


def load_queries(path: Path = None) -> List[Dict[str, Any]]:
    """Load test queries from JSON file."""
    if path is None:
        path = Path(__file__).parent.parent / "data" / "queries_llm.json"
    
    with open(path) as f:
        return json.load(f)


def build_system_prompt(substrate: Substrate, territories: List[str] = None) -> str:
    """Build system prompt with available territories for filtering."""
    territories_list = "\n".join(f"  - {t}" for t in (territories or []))
    
    return f"""You have access to a 'context' tool for searching a knowledge base.

To answer questions, call the 'context' tool with:
- query: your search query (required)
- filter: territory to search in (optional, see list below)
- types: frame types to filter by (optional, e.g. "observation", "claim", "decision")
- since: temporal filter (optional, e.g. "today", "yesterday", "2w", "last-week")

Available territories:
{territories_list}

Call the context tool ONCE with appropriate filters to find relevant information."""


# Only expose context tool for benchmarking
BENCHMARK_TOOLS = {"context"}


def call_openrouter(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    api_key: str,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Call OpenRouter API with tool support."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://rage-substrate.dev",  # Optional
        "X-Title": "RAGE Benchmark",  # Optional
    }
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "required",  # Force tool use - no answering from training data
        "temperature": 0.1,  # Low temperature for consistent evaluation
        "max_tokens": 2048,
    }
    
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def execute_tool_call(
    substrate: Substrate,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> str:
    """Execute a tool call and return the result as string."""
    try:
        result = substrate.tools.execute_sync(tool_name, tool_args)
        if result.success:
            # Format result data as JSON for the LLM
            return json.dumps(result.data, indent=2, default=str)
        else:
            return json.dumps({"error": result.error}, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def run_llm_exploration(
    query: str,
    substrate: Substrate,
    api_key: str,
    max_tool_rounds: int = 1,  # Only need 1 round - single context call
) -> Dict[str, Any]:
    """
    Run LLM exploration for a single query.
    
    Args:
        query: The user query
        substrate: Initialized Substrate instance
        api_key: OpenRouter API key
        max_tool_rounds: Maximum number of tool calling rounds (1 = single context call)
        
    Returns:
        Dict with response, tool_calls, latency_ms, etc.
    """
    # Get tool specs (only context tool for benchmarking)
    all_tools = substrate.tools.openai_specs()
    tools = [t for t in all_tools if t["function"]["name"] in BENCHMARK_TOOLS]
    
    # Remove 'effort' from context tool params - we control it externally
    for tool in tools:
        if tool["function"]["name"] == "context":
            props = tool["function"]["parameters"].get("properties", {})
            if "effort" in props:
                del props["effort"]
            required = tool["function"]["parameters"].get("required", [])
            if "effort" in required:
                required.remove("effort")
    
    # Get available territories for the system prompt
    territories_result = substrate.tools.execute_sync("territories", {})
    territories = territories_result.data.get("territories", []) if territories_result.success else []
    
    # Build messages
    system_prompt = build_system_prompt(substrate, territories)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    
    tool_calls_made = []
    total_start = time.time()
    llm_time_ms = 0
    tool_time_ms = 0
    
    for round_num in range(max_tool_rounds):
        # Call LLM
        llm_start = time.time()
        try:
            response = call_openrouter(messages, tools, api_key)
        except httpx.HTTPStatusError as e:
            return {
                "response": f"API error: {e.response.status_code}",
                "tool_calls": tool_calls_made,
                "total_latency_ms": (time.time() - total_start) * 1000,
                "llm_time_ms": llm_time_ms,
                "tool_time_ms": tool_time_ms,
                "rounds": round_num,
                "error": str(e),
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": tool_calls_made,
                "total_latency_ms": (time.time() - total_start) * 1000,
                "llm_time_ms": llm_time_ms,
                "tool_time_ms": tool_time_ms,
                "rounds": round_num,
                "error": str(e),
            }
        
        llm_time_ms += (time.time() - llm_start) * 1000
        
        # Parse response
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        
        # Add assistant message to history
        messages.append(message)
        
        # Check if LLM wants to call tools
        if message.get("tool_calls"):
            tool_results = []
            
            for tool_call in message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                try:
                    tool_args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}
                
                # Execute tool
                tool_start = time.time()
                result = execute_tool_call(substrate, tool_name, tool_args)
                tool_time_ms += (time.time() - tool_start) * 1000
                
                # Record tool call
                tool_calls_made.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result_preview": result[:500] if len(result) > 500 else result,
                })
                
                # Add tool result to messages
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                })
            
            # Add all tool results to messages
            messages.extend(tool_results)
            
        else:
            # No more tool calls - LLM has final answer
            final_response = message.get("content", "")
            return {
                "response": final_response,
                "tool_calls": tool_calls_made,
                "total_latency_ms": (time.time() - total_start) * 1000,
                "llm_time_ms": llm_time_ms,
                "tool_time_ms": tool_time_ms,
                "rounds": round_num + 1,
            }
    
    # Max rounds reached - get final content
    final_content = messages[-1].get("content", "") if messages else ""
    return {
        "response": final_content,
        "tool_calls": tool_calls_made,
        "total_latency_ms": (time.time() - total_start) * 1000,
        "llm_time_ms": llm_time_ms,
        "tool_time_ms": tool_time_ms,
        "rounds": max_tool_rounds,
        "max_rounds_reached": True,
    }


def evaluate_response(
    response: str,
    expected_frame_titles: List[str] = None,
    expected_contains: List[str] = None,
    used_tools: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate if the response contains expected content.
    
    Args:
        response: The LLM's response text
        expected_frame_titles: Titles expected to be mentioned
        expected_contains: Keywords expected in response
        used_tools: Whether tools were actually called
        
    Returns:
        Dict with match scores and details.
    """
    results = {
        "title_matches": [],
        "content_matches": [],
        "title_score": 0.0,
        "content_score": 0.0,
        "used_tools": used_tools,
        "skipped": False,
    }
    
    # If no tools were used, mark as skipped (answer came from training data)
    if not used_tools:
        results["skipped"] = True
        results["skip_reason"] = "no_tool_calls"
        results["combined_score"] = 0.0
        return results
    
    response_lower = response.lower()
    
    # Check for expected frame titles
    if expected_frame_titles:
        matches = []
        for title in expected_frame_titles:
            # Flexible matching - check if title words appear
            title_words = title.lower().split()
            # Match if majority of words found
            found_words = sum(1 for w in title_words if w in response_lower)
            match_ratio = found_words / len(title_words) if title_words else 0
            if match_ratio >= 0.5:
                matches.append(title)
        
        results["title_matches"] = matches
        results["title_score"] = len(matches) / len(expected_frame_titles)
    
    # Check for expected content keywords
    if expected_contains:
        matches = []
        for keyword in expected_contains:
            if keyword.lower() in response_lower:
                matches.append(keyword)
        
        results["content_matches"] = matches
        results["content_score"] = len(matches) / len(expected_contains) if expected_contains else 0
    
    # Combined score
    scores = [results["title_score"], results["content_score"]]
    valid_scores = [s for s in scores if s > 0]
    results["combined_score"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    return results


def run_benchmark(
    queries: List[Dict[str, Any]],
    db_path: str = None,
    max_tool_rounds: int = 1,  # Single context call per query
) -> Dict[str, Any]:
    """
    Run LLM exploration benchmark on all queries.
    
    Args:
        queries: List of query specifications
        db_path: Path to substrate database
        max_tool_rounds: Max tool calling rounds per query (1 = single context call)
        
    Returns:
        Results dict with per-query and aggregate metrics.
    """
    api_key = get_openrouter_api_key()
    
    if db_path is None:
        db_path = str(RAGE_SUBSTRATE_PATH / "substrate.db")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": OPENROUTER_MODEL,
        "max_tool_rounds": max_tool_rounds,
        "queries": [],
    }
    
    # Initialize substrate once for all queries
    substrate = Substrate("benchmark-llm", db_path=db_path)
    
    try:
        for i, query_spec in enumerate(queries, 1):
            query = query_spec["query"]
            category = query_spec.get("category", "unknown")
            difficulty = query_spec.get("difficulty", "unknown")
            
            print(f"\n[{i}/{len(queries)}] {category}/{difficulty}: {query[:60]}...")
            
            # Run LLM exploration
            print("  Running LLM exploration...", end=" ", flush=True)
            exploration_result = run_llm_exploration(
                query, substrate, api_key, max_tool_rounds
            )
            
            num_tools = len(exploration_result.get("tool_calls", []))
            latency = exploration_result.get("total_latency_ms", 0)
            print(f"✓ ({latency:.0f}ms, {num_tools} tool calls)")
            
            # Evaluate response (only count if tools were used)
            used_tools = num_tools > 0
            evaluation = evaluate_response(
                exploration_result.get("response", ""),
                expected_frame_titles=query_spec.get("expected_frame_titles"),
                expected_contains=query_spec.get("expected_content_contains"),
                used_tools=used_tools,
            )
            
            # Build result entry
            query_result = {
                "query": query,
                "category": category,
                "difficulty": difficulty,
                "response": exploration_result.get("response", ""),
                "tool_calls": exploration_result.get("tool_calls", []),
                "num_tool_calls": num_tools,
                "rounds": exploration_result.get("rounds", 0),
                "total_latency_ms": latency,
                "llm_time_ms": exploration_result.get("llm_time_ms", 0),
                "tool_time_ms": exploration_result.get("tool_time_ms", 0),
                "evaluation": evaluation,
            }
            
            if exploration_result.get("error"):
                query_result["error"] = exploration_result["error"]
            
            results["queries"].append(query_result)
            
            # Print quick summary
            print(f"    Tools: {[tc['tool'] for tc in exploration_result.get('tool_calls', [])]}")
            print(f"    Score: title={evaluation['title_score']:.2f} content={evaluation['content_score']:.2f}")
            
    finally:
        substrate.close()
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print benchmark summary."""
    print("\n" + "=" * 80)
    print("LLM EXPLORATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Model: {results['model']}")
    print(f"Queries: {len(results['queries'])}")
    
    # Tool usage tracking
    all_queries = results["queries"]
    n = len(all_queries)
    used_tools_count = sum(1 for q in all_queries if q["num_tool_calls"] > 0)
    skipped_count = sum(1 for q in all_queries if q["evaluation"].get("skipped", False))
    
    print(f"\nTool Usage: {used_tools_count}/{n} queries used tools ({100*used_tools_count/n:.0f}%)")
    if skipped_count > 0:
        print(f"⚠️  Skipped (no tool calls): {skipped_count} queries")
    print()
    
    # Group by category
    by_category: Dict[str, List] = {}
    for q in all_queries:
        cat = q["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(q)
    
    # Print per-category stats
    header = f"{'Category':<12} | {'Count':<6} | {'Tools Used':<10} | {'Title Score':<11} | {'Content Score':<13} | {'Avg Latency':<11}"
    print(header)
    print("-" * len(header))
    
    for category in sorted(by_category.keys()):
        queries = by_category[category]
        cat_n = len(queries)
        cat_used_tools = sum(1 for q in queries if q["num_tool_calls"] > 0)
        
        # Only score queries that used tools
        scored_queries = [q for q in queries if q["num_tool_calls"] > 0]
        if scored_queries:
            title_score = sum(q["evaluation"]["title_score"] for q in scored_queries) / len(scored_queries)
            content_score = sum(q["evaluation"]["content_score"] for q in scored_queries) / len(scored_queries)
        else:
            title_score = 0.0
            content_score = 0.0
        avg_latency = sum(q["total_latency_ms"] for q in queries) / cat_n
        
        print(f"{category:<12} | {cat_n:<6} | {cat_used_tools:>3}/{cat_n:<6} | {title_score:>11.2f} | {content_score:>13.2f} | {avg_latency:>9.0f}ms")
    
    print("-" * len(header))
    
    # Overall stats (only for queries that used tools)
    if n > 0:
        scored_queries = [q for q in all_queries if q["num_tool_calls"] > 0]
        if scored_queries:
            title_score = sum(q["evaluation"]["title_score"] for q in scored_queries) / len(scored_queries)
            content_score = sum(q["evaluation"]["content_score"] for q in scored_queries) / len(scored_queries)
        else:
            title_score = 0.0
            content_score = 0.0
        avg_latency = sum(q["total_latency_ms"] for q in all_queries) / n
        
        print(f"{'OVERALL':<12} | {n:<6} | {used_tools_count:>3}/{n:<6} | {title_score:>11.2f} | {content_score:>13.2f} | {avg_latency:>9.0f}ms")
    
    print("\n" + "=" * 80)
    
    # Tool usage breakdown
    print("\nTool Usage Breakdown:")
    tool_counts: Dict[str, int] = {}
    for q in all_queries:
        for tc in q.get("tool_calls", []):
            tool = tc["tool"]
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    if tool_counts:
        for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            print(f"  {tool}: {count}")
    else:
        print("  (no tool calls made)")
    
    # List queries without tool calls
    no_tool_queries = [q for q in all_queries if q["num_tool_calls"] == 0]
    if no_tool_queries:
        print(f"\n⚠️  Queries without tool calls ({len(no_tool_queries)}):")
        for q in no_tool_queries:
            print(f"  - {q['query'][:60]}...")
    
    print()


def save_results(results: Dict[str, Any], output_dir: Path = None):
    """Save results to JSON file."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"llm_benchmark_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_file}")
    return output_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAGE LLM Exploration Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.llm_exploration --limit 5
  python -m benchmarks.llm_exploration --queries data/queries_llm.json
  python -m benchmarks.llm_exploration --category direct --limit 10
        """,
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of queries to run (default: all)"
    )
    parser.add_argument(
        "--queries", "-q",
        type=str,
        default=None,
        help="Path to queries JSON file (default: data/queries_llm.json)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to substrate database (default: ../rage-substrate/substrate.db)"
    )
    parser.add_argument(
        "--max-rounds", "-r",
        type=int,
        default=1,
        help="Maximum tool calling rounds per query (default: 1 = single context call)"
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help="Filter queries by category (direct, structural, temporal, negative)"
    )
    return parser.parse_args()


def main():
    """Run the LLM exploration benchmark."""
    args = parse_args()
    
    print("=" * 80)
    print("RAGE LLM Exploration Benchmark")
    print("=" * 80)
    print(f"Model: {OPENROUTER_MODEL}")
    print(f"Max tool rounds: {args.max_rounds}")
    
    # Load queries
    print("\nLoading test queries...")
    queries_path = Path(args.queries) if args.queries else None
    queries = load_queries(queries_path)
    
    # Filter by category
    if args.category:
        queries = [q for q in queries if q.get("category") == args.category]
        print(f"Filtered to category '{args.category}': {len(queries)} queries")
    
    # Apply limit
    if args.limit:
        queries = queries[:args.limit]
        print(f"Limited to first {args.limit} queries")
    
    print(f"Total queries to run: {len(queries)}")
    
    if not queries:
        print("No queries to run!")
        return
    
    # Run benchmark
    print("\nRunning LLM exploration benchmark...")
    results = run_benchmark(
        queries,
        db_path=args.db,
        max_tool_rounds=args.max_rounds,
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_dir = Path(args.output) if args.output else None
    save_results(results, output_dir)
    
    print("✓ Benchmark complete!")


if __name__ == "__main__":
    main()
