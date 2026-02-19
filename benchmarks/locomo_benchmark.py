"""LoCoMo benchmark: Test RAGE retrieval + LLM answering on long-conversation QA."""

import sys
import json
import time
import argparse
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich import box

log = logging.getLogger(__name__)
console = Console()

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


def llm_judge(prediction: str, ground_truth: str, client: OpenAI, model: str = "openai/gpt-4o-mini") -> int:
    """
    Use LLM as judge to determine if prediction matches ground truth semantically.
    
    Returns 1 if correct, 0 if wrong.
    This is the metric used by MemU, MemMachine, Mem0 etc.
    """
    prompt = f"""You are evaluating whether a predicted answer matches the ground truth answer.
The answers don't need to be word-for-word identical, but they must convey the same information.

Ground Truth: {ground_truth}
Predicted Answer: {prediction}

Does the predicted answer convey the same information as the ground truth?
Reply with only "CORRECT" or "WRONG"."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        verdict = response.choices[0].message.content.strip().upper()
        return 1 if "CORRECT" in verdict else 0
    except Exception as e:
        log.warning(f"LLM judge failed: {e}")
        return 0


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
    
    # Valid modes
    MODES = ["autonomous", "fixed-low", "fixed-medium", "fixed-high"]

    def __init__(
        self,
        substrate: Substrate,
        model: str = "openai/gpt-4o",
        api_key: str = None,
        verbose: bool = False,
        mode: str = "autonomous"
    ):
        """
        Initialize QA runner.
        
        Args:
            substrate: RAGE substrate instance
            model: LLM model name for OpenRouter
            api_key: API key (defaults to OPENROUTER_API_KEY env var)
            verbose: Print verbose output
            mode: Benchmark mode - "autonomous" (full tools) or "fixed-{low,medium,high}" (inject context call)
        """
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {self.MODES}")
        
        self.substrate = substrate
        self.model = model
        self.verbose = verbose
        self.mode = mode
        
        api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Get tools from substrate (same as frontend)
        self.tools = substrate.tools.openai_specs()
        
        # Build system prompt: RAGE tools + QA-specific instructions
        rage_instructions = substrate.tools.instructions()
        qa_instructions = """

## Answer Format

Write answers as SHORT PHRASES only. Use exact words from the retrieved context whenever possible.

## Temporal Reasoning

When content mentions relative times like "yesterday", "last week", "last year", compute the ACTUAL DATE using the frame's created_at timestamp.

Example: If a frame has created_at="2023-05-08" and content says "yesterday", the answer should be "7 May 2023" (not "yesterday").

Always give specific dates, not relative terms."""
        
        self.system_prompt = rage_instructions + qa_instructions
    
    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a RAGE tool and return result as string."""
        # Execute tool through substrate (generic - handles any tool)
        result = self.substrate.tools.execute_sync(name, args)
        
        if not result.success:
            return f"Error: {result.error}"
        
        # Format output based on result data
        data = result.data or {}
        
        # Handle frames (from find, context, etc.)
        frames = data.get("frames", [])
        if frames:
            output = []
            for frame in frames[:10]:
                title = frame.get("title", "Untitled")
                created_at = frame.get("created_at", "")
                content = frame.get("content", "") or frame.get("summary", "")
                # Include timestamp for temporal reasoning
                frame_str = f"[{title}]"
                if created_at:
                    frame_str += f"\ncreated_at: {created_at}"
                frame_str += f"\n{content}"
                output.append(frame_str)
            return "\n\n".join(output)
        
        # Handle other data types
        if data:
            return json.dumps(data, indent=2, default=str)
        
        return "No relevant information found."
    
    def answer_question(self, question: str, max_turns: int = 10) -> Tuple[str, Dict[str, Any]]:
        """
        Answer a question using RAGE tools and LLM.
        
        Args:
            question: The question to answer
            max_turns: Maximum tool call turns
        
        Returns:
            (answer, metadata) tuple
        """
        # Dispatch based on mode
        if self.mode.startswith("fixed-"):
            return self._answer_fixed_effort(question)
        else:
            return self._answer_autonomous(question, max_turns)
    
    def _answer_fixed_effort(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """Answer using fixed effort level (no tool autonomy)."""
        # Extract effort level from mode
        effort = self.mode.split("-")[1]  # "low", "medium", or "high"
        
        # Call context directly
        if self.verbose:
            print(f"    [Fixed] context(query='{question}', effort='{effort}')")
        
        context_result = self._execute_tool("context", {"query": question, "effort": effort})
        
        metadata = {
            "tool_calls": [{"name": "context", "args": {"query": question, "effort": effort}}],
            "context_result": context_result if context_result else "",  # Full context for analysis
            "tokens_used": 0,
            "turns": 1,
            "mode": self.mode
        }
        
        # Now ask LLM to answer based on context
        user_prompt = f"""Based on the following context, answer the question with a SHORT PHRASE only.

Context:
{context_result}

Question: {question}

Short answer:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer questions concisely using only the provided context. Give SHORT PHRASE answers."},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100
        )
        
        metadata["tokens_used"] = response.usage.total_tokens if response.usage else 0
        return response.choices[0].message.content or "", metadata
    
    def _answer_autonomous(self, question: str, max_turns: int = 10) -> Tuple[str, Dict[str, Any]]:
        """Answer using autonomous tool selection."""
        # Format question with LoCoMo-style prompt for short answers
        user_prompt = f"""Answer this question using the knowledge tools. Give a SHORT PHRASE answer only.

Question: {question}

Short answer:"""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        metadata = {
            "tool_calls": [],
            "tokens_used": 0,
            "turns": 0,
            "mode": "autonomous"
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
                
                # Inject system message with remaining calls (like frontend does)
                calls_remaining = max_turns - turn - 1
                if calls_remaining > 0:
                    messages.append({
                        "role": "system", 
                        "content": f"[{calls_remaining} tool calls remaining. Use them to gather more information if needed before answering.]"
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


def _render_question_panel(
    index: int,
    total: int,
    category_name: str,
    question: str,
    context: str,
    llm_score: Optional[int],
    f1: float,
    latency_s: float,
    batch_judge: bool = True
) -> Panel:
    """Render a Rich panel for a single question result."""
    # Build the content table
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", style="bold cyan", width=10)
    table.add_column("Value")
    
    # Truncate question if too long
    q_display = question if len(question) <= 70 else question[:67] + "..."
    table.add_row("Query:", q_display)
    
    # Context preview (first 200 chars)
    ctx_preview = context[:200] + "..." if len(context) > 200 else context
    ctx_preview = ctx_preview.replace("\n", " ")
    table.add_row("Context:", f"[dim]{ctx_preview}[/dim]")
    
    # Metrics row
    if batch_judge:
        metrics = f"[yellow]Judge: pending[/yellow] │ F1: {f1:.2f} │ Latency: {latency_s:.1f}s"
    else:
        judge_color = "green" if llm_score == 1 else "red"
        metrics = f"[{judge_color}]Judge: {llm_score}[/{judge_color}] │ F1: {f1:.2f} │ Latency: {latency_s:.1f}s"
    table.add_row("Metrics:", metrics)
    
    # Title with progress
    title = f"[bold][{index}/{total}] {category_name}[/bold]"
    
    return Panel(table, title=title, border_style="blue", box=box.ROUNDED)


def run_benchmark(
    qa_pairs: List[Dict[str, Any]],
    runner: RAGEQARunner,
    limit: int = None,
    categories: List[int] = None,
    verbose: bool = False,
    batch_judge: bool = True
) -> Dict[str, Any]:
    """
    Run benchmark on QA pairs.
    
    Args:
        qa_pairs: List of QA dicts
        runner: RAGEQARunner instance
        limit: Max number of questions to evaluate
        categories: Filter to specific categories (1-5)
        verbose: Print verbose output
        batch_judge: If True, run LLM judging as batch at end (more efficient)
    
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
        "mode": runner.mode,
        "total_questions": len(qa_pairs),
        "questions": [],
        "by_category": defaultdict(lambda: {"f1_sum": 0, "em_sum": 0, "llm_sum": 0, "count": 0})
    }
    
    total_f1 = 0.0
    total_em = 0.0
    total_llm = 0
    
    console.print()  # Blank line before panels
    
    for i, qa in enumerate(qa_pairs, 1):
        question = qa["question"]
        ground_truth = qa["answer"]
        category = qa["category"]
        category_name = qa["category_name"]
        
        start = time.time()
        predicted, meta = runner.answer_question(question)
        latency_s = time.time() - start
        latency_ms = latency_s * 1000
        
        # Compute metrics (LLM judge deferred if batch_judge=True)
        f1 = compute_f1(predicted, ground_truth)
        em = compute_exact_match(predicted, ground_truth)
        llm_score = None if batch_judge else llm_judge(predicted, ground_truth, runner.client)
        
        # Get context from tool calls for display
        # Get context from tool results or metadata
        context_preview = meta.get("context_result", "")
        if not context_preview:
            # Fallback: show tool call info
            for tc in meta.get("tool_calls", []):
                if tc.get("name") == "context":
                    context_preview = tc.get("result", f"[{tc['args'].get('effort', 'auto')}] {tc['args'].get('query', '')[:100]}")
                    break
        if not context_preview:
            context_preview = "(no context retrieved)"
        
        # Render rich panel
        panel = _render_question_panel(
            index=i,
            total=len(qa_pairs),
            category_name=category_name,
            question=question,
            context=context_preview,
            llm_score=llm_score,
            f1=f1,
            latency_s=latency_s,
            batch_judge=batch_judge
        )
        console.print(panel)
        
        # Verbose: show ground truth and prediction
        if verbose:
            console.print(f"  [dim]Ground truth:[/dim] {ground_truth}")
            console.print(f"  [dim]Predicted:[/dim] {predicted[:150]}...")
            console.print()
        
        # Store result
        q_result = {
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "context": meta.get("context_result", ""),  # Full context sent to LLM
            "evidence": qa["evidence"],
            "category": category,
            "category_name": category_name,
            "sample_id": qa["sample_id"],
            "f1": f1,
            "exact_match": em,
            "llm_score": llm_score,  # None if batch_judge
            "latency_ms": latency_ms,
            "tool_calls": meta["tool_calls"],
            "tokens_used": meta["tokens_used"]
        }
        results["questions"].append(q_result)
        
        # Update aggregates (LLM done in batch later if batch_judge)
        total_f1 += f1
        total_em += em
        if llm_score is not None:
            total_llm += llm_score
        
        results["by_category"][category_name]["f1_sum"] += f1
        results["by_category"][category_name]["em_sum"] += em
        if llm_score is not None:
            results["by_category"][category_name]["llm_sum"] += llm_score
        results["by_category"][category_name]["count"] += 1
    
    # Batch LLM judging phase (if enabled)
    if batch_judge:
        console.print()
        console.rule("[bold cyan]LLM Judging Phase[/bold cyan]")
        
        total_llm = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Judging answers...", total=len(results["questions"]))
            
            for i, q in enumerate(results["questions"], 1):
                llm_score = llm_judge(q["predicted"], q["ground_truth"], runner.client)
                q["llm_score"] = llm_score
                total_llm += llm_score
                
                # Update category stats
                cat_name = q["category_name"]
                results["by_category"][cat_name]["llm_sum"] += llm_score
                
                progress.update(task, advance=1, description=f"[cyan]Judging... ({total_llm}/{i} correct)")
        
        console.print(f"[green]✓[/green] Judging complete: [bold]{total_llm}/{len(results['questions'])}[/bold] correct ({total_llm/len(results['questions'])*100:.1f}%)")
    
    # Compute overall averages
    n = len(qa_pairs)
    results["avg_f1"] = total_f1 / n if n > 0 else 0
    results["avg_em"] = total_em / n if n > 0 else 0
    results["avg_llm"] = total_llm / n if n > 0 else 0
    results["llm_score_pct"] = (total_llm / n * 100) if n > 0 else 0
    
    # Convert defaultdict to regular dict with averages
    by_cat = {}
    for cat_name, stats in results["by_category"].items():
        count = stats["count"]
        by_cat[cat_name] = {
            "avg_f1": stats["f1_sum"] / count if count > 0 else 0,
            "avg_em": stats["em_sum"] / count if count > 0 else 0,
            "avg_llm": stats["llm_sum"] / count if count > 0 else 0,
            "llm_pct": (stats["llm_sum"] / count * 100) if count > 0 else 0,
            "count": count
        }
    results["by_category"] = by_cat
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print summary table of results using Rich."""
    console.print()
    console.rule("[bold green]LoCoMo Benchmark Results[/bold green]")
    console.print()
    
    # Config info
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="bold")
    config_table.add_column("Value")
    config_table.add_row("Model", results['model'])
    config_table.add_row("Mode", results.get('mode', 'autonomous'))
    config_table.add_row("Questions", str(results['total_questions']))
    console.print(config_table)
    console.print()
    
    # Results table
    table = Table(title="Results by Category", box=box.ROUNDED)
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Avg F1", justify="right")
    table.add_column("Avg EM", justify="right")
    table.add_column("LLM %", justify="right", style="bold")
    
    # By category (in order)
    for cat_name in ["single-hop", "temporal", "commonsense", "multi-hop", "adversarial"]:
        if cat_name in results["by_category"]:
            stats = results["by_category"][cat_name]
            llm_pct = stats['llm_pct']
            llm_style = "green" if llm_pct >= 70 else "yellow" if llm_pct >= 50 else "red"
            table.add_row(
                cat_name,
                str(stats['count']),
                f"{stats['avg_f1']:.3f}",
                f"{stats['avg_em']:.3f}",
                f"[{llm_style}]{llm_pct:.1f}%[/{llm_style}]"
            )
    
    # Overall row
    table.add_section()
    overall_pct = results['llm_score_pct']
    overall_style = "green" if overall_pct >= 70 else "yellow" if overall_pct >= 50 else "red"
    table.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold]{results['total_questions']}[/bold]",
        f"[bold]{results['avg_f1']:.3f}[/bold]",
        f"[bold]{results['avg_em']:.3f}[/bold]",
        f"[bold {overall_style}]{overall_pct:.1f}%[/bold {overall_style}]"
    )
    
    console.print(table)
    console.print()
    
    # Highlight the main metric
    console.print(Panel(
        f"[bold white]🎯 LLM Judge Score: {results['llm_score_pct']:.1f}%[/bold white]\n"
        f"[dim](comparable to MemU, MemMachine, etc.)[/dim]",
        border_style="green" if overall_pct >= 70 else "yellow",
        box=box.DOUBLE
    ))


def save_results(results: Dict[str, Any], output_path: Path = None):
    """Save results to JSON file."""
    if output_path is None:
        # Default: auto-generate path in results directory
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"locomo_benchmark_{timestamp}.json"
    else:
        # Ensure parent directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[green]✓[/green] Results saved to: [bold]{output_path}[/bold]")


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
        default="openai/gpt-4o",
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
        "--conversation",
        type=int,
        default=None,
        help="Filter to specific conversation index (0-9)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path for JSON results (e.g., results/bench-low.json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="autonomous",
        choices=["autonomous", "fixed-low", "fixed-medium", "fixed-high"],
        help="Benchmark mode: autonomous (full tools) or fixed-{low,medium,high} (inject context call)"
    )
    parser.add_argument(
        "--no-batch-judge",
        action="store_true",
        help="Disable batch judging (judge inline instead)"
    )
    
    args = parser.parse_args()
    
    console.rule("[bold blue]LoCoMo Benchmark[/bold blue]")
    
    # Load QA data
    console.print("\n[cyan]Loading LoCoMo data...[/cyan]")
    data_path = Path(args.data) if args.data else None
    data = load_locomo_data(data_path)
    qa_pairs = extract_qa_pairs(data)
    console.print(f"  Loaded [bold]{len(qa_pairs)}[/bold] QA pairs from [bold]{len(data)}[/bold] conversations")
    
    # Filter by conversation if specified
    if args.conversation is not None:
        conv_sample_id = data[args.conversation].get("sample_id")
        qa_pairs = [q for q in qa_pairs if q["sample_id"] == conv_sample_id]
        console.print(f"  Filtered to conversation [bold]{args.conversation}[/bold] ({conv_sample_id}): [bold]{len(qa_pairs)}[/bold] questions")
    
    # Parse categories
    categories = None
    if args.categories:
        categories = [int(c.strip()) for c in args.categories.split(",")]
        console.print(f"  Filtering to categories: [bold]{categories}[/bold]")
    
    # Initialize substrate
    if args.db:
        db_path = args.db
    else:
        db_path = str(Path(__file__).parent.parent / "data" / "locomo" / "locomo_substrate.db")
    
    console.print(f"  Database: [dim]{db_path}[/dim]")
    
    if not Path(db_path).exists():
        console.print(f"\n[red]Error:[/red] Database not found at {db_path}", style="red")
        console.print("[dim]Run locomo_ingest.py first to create and populate the database.[/dim]")
        sys.exit(1)
    
    substrate = Substrate("locomo_benchmark", db_path=db_path)
    
    # Show config
    console.print()
    config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    config_table.add_column("Key", style="bold cyan")
    config_table.add_column("Value")
    config_table.add_row("Model", args.model)
    config_table.add_row("Mode", args.mode)
    config_table.add_row("Batch judging", "yes" if not args.no_batch_judge else "no")
    if args.output:
        config_table.add_row("Output", args.output)
    console.print(config_table)
    
    # Initialize QA runner
    runner = RAGEQARunner(substrate, model=args.model, verbose=args.verbose, mode=args.mode)
    
    # Run benchmark
    batch_judge = not args.no_batch_judge
    console.print()
    console.rule("[cyan]Running Benchmark[/cyan]")
    results = run_benchmark(
        qa_pairs,
        runner,
        limit=args.limit,
        categories=categories,
        batch_judge=batch_judge,
        verbose=args.verbose
    )
    
    # Add config to results for JSON export
    results["config"] = {
        "model": args.model,
        "mode": args.mode,
        "conversation": args.conversation,
        "categories": categories,
        "limit": args.limit,
        "batch_judge": batch_judge,
        "db_path": db_path,
        "data_path": str(data_path) if data_path else None
    }
    
    substrate.close()
    
    # Print and save results
    print_summary(results)
    
    # Save to JSON if --output specified (or always with default path)
    output_path = Path(args.output) if args.output else None
    save_results(results, output_path)
    
    console.print("\n[bold green]✓ Benchmark complete![/bold green]")


if __name__ == "__main__":
    main()
