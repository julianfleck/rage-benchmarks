#!/usr/bin/env python3
"""Re-judge existing benchmark results with fixed adversarial scoring.

This script re-evaluates LLM judge scores on existing result files,
using the corrected adversarial scoring logic where abstaining/refusing
is the CORRECT response for unanswerable questions.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import box

import os

console = Console()


def llm_judge(prediction: str, ground_truth: str, client: OpenAI, model: str = "openai/gpt-4o-mini") -> int:
    """Standard LLM judge for normal questions."""
    prompt = f"""You are evaluating whether a predicted answer matches the ground truth answer.
The answers don't need to be word-for-word identical, but they must convey the same information.

DATE EQUIVALENCE - these are CORRECT:
- Different formats: "10 July 2023" = "July 10, 2023" = "2023-07-10"
- Computed dates: "The friday before 15 July 2023" = "July 7, 2023"
- With day-of-week: "Saturday, May 20, 2023" = "May 20, 2023"
- Relative times: "10 years ago" = "ten years ago"

SEMANTIC EQUIVALENCE - these are CORRECT:
- Synonyms: "transgender woman" = "trans woman"
- Extra context is OK: "10 years ago" matches "ten years ago from June 2023, so June 2013"
- Verbose answers containing the correct info are CORRECT

Ground Truth: {ground_truth}
Predicted Answer: {prediction}

Does the predicted answer convey the same information as the ground truth? A verbose answer that CONTAINS the ground truth info is CORRECT.
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
        console.print(f"[yellow]Warning: LLM judge failed: {e}[/yellow]")
        return 0


def llm_judge_adversarial(prediction: str, wrong_answer: str, client: OpenAI, model: str = "openai/gpt-4o-mini") -> int:
    """
    Judge adversarial (unanswerable) questions with INVERTED scoring.
    
    For adversarial questions, the correct behavior is recognizing the question
    cannot be answered. Returns 1 if model abstained, 0 if it gave wrong answer.
    """
    prompt = f"""You are evaluating an "unanswerable" question - the question CANNOT be answered from the context because it asks about the wrong person or wrong situation.

The CORRECT response is to recognize the question cannot be answered:
- Saying "I don't know", "not mentioned", "N/A", "cannot be determined"
- Expressing uncertainty or that the information isn't available
- Refusing to answer or saying it's unanswerable

The WRONG response is to give this incorrect answer (which confuses speakers/context):
Wrong answer to avoid: {wrong_answer}

Model's response: {prediction}

Did the model CORRECTLY recognize the question was unanswerable (expressed uncertainty, said "don't know", etc.)?
Or did it INCORRECTLY give the wrong answer or something similar to it?

Reply with only "CORRECT" (model appropriately abstained) or "WRONG" (model gave incorrect answer)."""

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
        console.print(f"[yellow]Warning: LLM adversarial judge failed: {e}[/yellow]")
        return 0


def load_locomo_qa_info(data_path: Path = None) -> dict:
    """Load adversarial answers from original LoCoMo data."""
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "locomo" / "locomo10.json"
    
    with open(data_path) as f:
        data = json.load(f)
    
    # Build lookup: question -> adversarial_answer
    qa_info = {}
    for item in data:
        for qa in item.get("qa", []):
            question = qa["question"]
            if "adversarial_answer" in qa:
                qa_info[question] = {
                    "is_adversarial": True,
                    "adversarial_answer": str(qa["adversarial_answer"])
                }
            else:
                qa_info[question] = {
                    "is_adversarial": False,
                    "adversarial_answer": None
                }
    
    return qa_info


def rejudge_results(results_path: Path, output_path: Path = None, dry_run: bool = False) -> dict:
    """Re-judge a results file with corrected adversarial scoring."""
    
    # Load existing results
    with open(results_path) as f:
        results = json.load(f)
    
    # Load original QA data for adversarial answers
    qa_info = load_locomo_qa_info()
    
    # Setup OpenAI client
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    
    console.print(f"\n[cyan]Re-judging {len(results['questions'])} questions from {results_path.name}[/cyan]")
    
    # Track changes
    old_scores = {"total": 0, "adversarial": 0, "normal": 0}
    new_scores = {"total": 0, "adversarial": 0, "normal": 0}
    changes = []
    
    # Count question types
    adversarial_count = 0
    normal_count = 0
    
    for q in results["questions"]:
        question = q["question"]
        info = qa_info.get(question, {"is_adversarial": False})
        
        # Enrich with adversarial info if missing
        if "is_adversarial" not in q:
            q["is_adversarial"] = info["is_adversarial"]
        if "adversarial_answer" not in q and info["is_adversarial"]:
            q["adversarial_answer"] = info["adversarial_answer"]
        
        if q.get("is_adversarial"):
            adversarial_count += 1
        else:
            normal_count += 1
    
    console.print(f"  [dim]Found {adversarial_count} adversarial, {normal_count} normal questions[/dim]")
    
    if dry_run:
        console.print("\n[yellow]DRY RUN - showing what would change without making API calls[/yellow]")
        
        # Just show which questions would be re-judged
        for q in results["questions"]:
            if q.get("is_adversarial"):
                old = q.get("llm_score", 0)
                console.print(f"  [dim]Would re-judge adversarial:[/dim] {q['question'][:60]}...")
                console.print(f"    [dim]Old score: {old}, Predicted: {q['predicted'][:80]}...[/dim]")
        return results
    
    # Re-judge all questions
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Re-judging...", total=len(results["questions"]))
        
        for q in results["questions"]:
            old_score = q.get("llm_score", 0)
            old_scores["total"] += old_score
            
            if q.get("is_adversarial"):
                old_scores["adversarial"] += old_score
                # Use adversarial judge
                new_score = llm_judge_adversarial(q["predicted"], q["adversarial_answer"], client)
            else:
                old_scores["normal"] += old_score
                # Re-judge normal questions too for consistency
                new_score = llm_judge(q["predicted"], q["ground_truth"], client)
            
            new_scores["total"] += new_score
            if q.get("is_adversarial"):
                new_scores["adversarial"] += new_score
            else:
                new_scores["normal"] += new_score
            
            # Track changes
            if old_score != new_score:
                changes.append({
                    "question": q["question"][:60],
                    "is_adversarial": q.get("is_adversarial", False),
                    "old": old_score,
                    "new": new_score,
                    "predicted": q["predicted"][:80]
                })
            
            q["llm_score"] = new_score
            progress.advance(task)
    
    # Recalculate aggregates
    by_cat = defaultdict(lambda: {"llm_sum": 0, "f1_sum": 0, "em_sum": 0, "count": 0})
    
    for q in results["questions"]:
        cat = q["category_name"]
        by_cat[cat]["llm_sum"] += q["llm_score"]
        by_cat[cat]["f1_sum"] += q.get("f1", 0)
        by_cat[cat]["em_sum"] += q.get("exact_match", 0)
        by_cat[cat]["count"] += 1
    
    # Convert to averages
    results["by_category"] = {}
    for cat_name, stats in by_cat.items():
        count = stats["count"]
        results["by_category"][cat_name] = {
            "avg_f1": stats["f1_sum"] / count if count > 0 else 0,
            "avg_em": stats["em_sum"] / count if count > 0 else 0,
            "avg_llm": stats["llm_sum"] / count if count > 0 else 0,
            "llm_pct": (stats["llm_sum"] / count * 100) if count > 0 else 0,
            "count": count
        }
    
    # Update overall
    n = len(results["questions"])
    results["avg_llm"] = new_scores["total"] / n if n > 0 else 0
    results["llm_score_pct"] = (new_scores["total"] / n * 100) if n > 0 else 0
    results["rejudged_at"] = datetime.now().isoformat()
    results["rejudge_note"] = "Re-judged with corrected adversarial scoring"
    
    # Print summary
    console.print()
    console.rule("[bold green]Re-judge Results[/bold green]")
    
    # Changes summary
    console.print(f"\n[bold]Score Changes:[/bold]")
    console.print(f"  Overall: {old_scores['total']}/{n} → {new_scores['total']}/{n} ({old_scores['total']/n*100:.1f}% → {new_scores['total']/n*100:.1f}%)")
    console.print(f"  Adversarial ({adversarial_count}): {old_scores['adversarial']} → {new_scores['adversarial']} ({old_scores['adversarial']/adversarial_count*100:.1f}% → {new_scores['adversarial']/adversarial_count*100:.1f}%)")
    console.print(f"  Normal ({normal_count}): {old_scores['normal']} → {new_scores['normal']} ({old_scores['normal']/normal_count*100:.1f}% → {new_scores['normal']/normal_count*100:.1f}%)")
    
    # Show some specific changes
    if changes:
        console.print(f"\n[bold]{len(changes)} questions changed score:[/bold]")
        for c in changes[:10]:
            arrow = "✓" if c["new"] > c["old"] else "✗"
            adv = "[adversarial] " if c["is_adversarial"] else ""
            console.print(f"  {arrow} {adv}{c['question']}... ({c['old']} → {c['new']})")
        if len(changes) > 10:
            console.print(f"  [dim]... and {len(changes) - 10} more[/dim]")
    
    # Results by category
    console.print()
    table = Table(title="Results by Category", box=box.ROUNDED)
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("LLM %", justify="right", style="bold")
    
    for cat_name in ["single-hop", "temporal", "commonsense", "multi-hop", "adversarial"]:
        if cat_name in results["by_category"]:
            stats = results["by_category"][cat_name]
            llm_pct = stats['llm_pct']
            llm_style = "green" if llm_pct >= 70 else "yellow" if llm_pct >= 50 else "red"
            table.add_row(
                cat_name,
                str(stats['count']),
                f"[{llm_style}]{llm_pct:.1f}%[/{llm_style}]"
            )
    
    table.add_section()
    overall_pct = results['llm_score_pct']
    overall_style = "green" if overall_pct >= 70 else "yellow" if overall_pct >= 50 else "red"
    table.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold]{n}[/bold]",
        f"[bold {overall_style}]{overall_pct:.1f}%[/bold {overall_style}]"
    )
    console.print(table)
    
    # Save results
    if output_path is None:
        output_path = results_path.parent / f"{results_path.stem}_rejudged.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[green]✓[/green] Saved to: [bold]{output_path}[/bold]")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Re-judge benchmark results with fixed adversarial scoring")
    parser.add_argument("results", type=str, help="Path to results JSON file")
    parser.add_argument("-o", "--output", type=str, help="Output path (default: {input}_rejudged.json)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without making API calls")
    parser.add_argument("--all", action="store_true", help="Re-judge all results in results/ directory")
    
    args = parser.parse_args()
    
    if args.all:
        results_dir = Path(__file__).parent.parent / "results"
        for results_file in results_dir.glob("*.json"):
            if "_rejudged" not in results_file.name:
                console.rule(f"[bold]{results_file.name}[/bold]")
                rejudge_results(results_file, dry_run=args.dry_run)
    else:
        results_path = Path(args.results)
        output_path = Path(args.output) if args.output else None
        rejudge_results(results_path, output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
