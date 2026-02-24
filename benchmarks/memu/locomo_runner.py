"""
memU LoCoMo Benchmark Runner

Runs memU against LoCoMo conv-26 questions and compares with RAGE.
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add memU-experiment to path
MEMU_EXPERIMENT_PATH = Path(__file__).parent.parent.parent / "external" / "memU-experiment"
sys.path.insert(0, str(MEMU_EXPERIMENT_PATH))

import dotenv
dotenv.load_dotenv()


@dataclass
class BenchmarkResult:
    """Result for a single question."""
    question: str
    expected_answer: str
    generated_answer: str
    is_correct: bool
    category: int
    latency_ms: float
    retrieved_content: str = ""
    evidence: List[str] = field(default_factory=list)


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark run."""
    total_questions: int
    correct_answers: int
    accuracy: float
    category_stats: Dict[int, Dict[str, Any]]
    total_time_seconds: float
    avg_latency_ms: float


class MemUBenchmarkRunner:
    """
    Runs memU benchmark against LoCoMo data.
    """

    def __init__(
        self,
        chat_model: str = "gpt-4o-mini",
        memory_dir: str = "benchmarks/memu/memory",
        max_workers: int = 5
    ):
        self.chat_model = chat_model
        self.memory_dir = Path(memory_dir)
        self.max_workers = max_workers

        # These will be initialized when needed
        self.mem_agent = None
        self.response_agent = None
        self.evaluate_agent = None

    def _init_agents(self):
        """Initialize memU agents."""
        from mem_agent import MemAgent
        from response_agent import ResponseAgent
        from evaluate_agent import EvaluateAgent

        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.mem_agent = MemAgent(
            chat_deployment=self.chat_model,
            memory_dir=str(self.memory_dir)
        )

        self.response_agent = ResponseAgent(
            chat_deployment=self.chat_model,
            memory_dir=str(self.memory_dir)
        )

        self.evaluate_agent = EvaluateAgent(
            chat_deployment="gpt-4o"  # Use stronger model for evaluation
        )

        print(f"memU agents initialized with model: {self.chat_model}")
        print(f"Memory directory: {self.memory_dir}")

    def load_conv26_data(self) -> Dict[str, Any]:
        """Load conv-26 data from extracted files."""
        data_dir = Path(__file__).parent.parent / "locomo" / "data"

        with open(data_dir / "conv-26-sessions.json") as f:
            sessions_data = json.load(f)

        with open(data_dir / "conv-26-questions.json") as f:
            questions = json.load(f)

        return {
            "sessions": sessions_data,
            "questions": questions
        }

    def process_sessions(self, sessions_data: Dict[str, Any], force_reprocess: bool = False) -> bool:
        """
        Process conversation sessions to build memory.

        Args:
            sessions_data: The sessions data dict with speaker info and sessions
            force_reprocess: If True, clear existing memories first

        Returns:
            True if processing succeeded
        """
        if self.mem_agent is None:
            self._init_agents()

        speaker_a = sessions_data["speaker_a"]
        speaker_b = sessions_data["speaker_b"]
        characters = [speaker_a, speaker_b]

        # Check if memories already exist
        profile_files = list(self.memory_dir.glob("*_profile.txt"))
        events_files = list(self.memory_dir.glob("*_events.txt"))

        if profile_files or events_files:
            if force_reprocess:
                print("Clearing existing memories...")
                self.mem_agent.clear_character_memory(characters)
            else:
                print(f"Memories already exist for {len(profile_files)} characters. Use force_reprocess=True to rebuild.")
                return True

        sessions = sessions_data["sessions"]
        print(f"\nProcessing {len(sessions)} sessions for {characters}...")

        for session_name in sorted(sessions.keys(), key=lambda x: int(x.replace("session_", ""))):
            session = sessions[session_name]
            session_date = session["date"]
            utterances = session["utterances"]

            print(f"  Processing {session_name} ({session['turns']} turns, {session_date})...")

            # Convert utterances to format expected by memU
            session_data = []
            for utt in utterances:
                if isinstance(utt, dict):
                    session_data.append(utt)
                else:
                    # Handle if utterances are strings
                    session_data.append({"speaker": "unknown", "text": str(utt)})

            # Process session with MemAgent
            try:
                self.mem_agent.update_character_memory(
                    session_data=session_data,
                    session_date=session_date,
                    characters=characters
                )
            except Exception as e:
                print(f"    Warning: Error processing session: {e}")

        print(f"\nSession processing complete.")
        return True

    def answer_question(self, question: str, characters: List[str]) -> Dict[str, Any]:
        """
        Answer a single question using memU.

        Returns:
            Dict with 'answer' and 'retrieved_content' keys
        """
        if self.response_agent is None:
            self._init_agents()

        start_time = time.time()

        try:
            result = self.response_agent.answer_question(
                question=question,
                characters=characters,
                max_iterations=3
            )

            latency_ms = (time.time() - start_time) * 1000

            return {
                "answer": result.get("answer", ""),
                "retrieved_content": result.get("retrieved_content", ""),
                "latency_ms": latency_ms,
                "success": True
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                "answer": f"Error: {e}",
                "retrieved_content": "",
                "latency_ms": latency_ms,
                "success": False
            }

    def evaluate_answer(self, question: str, generated_answer: str, expected_answer: str) -> Dict[str, Any]:
        """
        Evaluate if the generated answer is correct.

        Returns:
            Dict with 'is_correct', 'explanation' keys
        """
        if self.evaluate_agent is None:
            self._init_agents()

        try:
            result = self.evaluate_agent.evaluate(
                question=question,
                generated_answer=generated_answer,
                expected_answer=expected_answer
            )
            return {
                "is_correct": result.get("is_correct", False),
                "explanation": result.get("explanation", "")
            }
        except Exception as e:
            # Fall back to simple string matching
            gen_lower = generated_answer.lower().strip()
            exp_lower = expected_answer.lower().strip()
            is_correct = exp_lower in gen_lower or gen_lower in exp_lower
            return {
                "is_correct": is_correct,
                "explanation": f"Fallback evaluation (agent error: {e})"
            }

    def run_benchmark(
        self,
        questions: List[Dict[str, Any]],
        characters: List[str],
        category_filter: Optional[List[int]] = None
    ) -> List[BenchmarkResult]:
        """
        Run benchmark on all questions.

        Args:
            questions: List of question dicts with 'question', 'answer', 'category', 'evidence'
            characters: List of character names
            category_filter: If provided, only run questions in these categories

        Returns:
            List of BenchmarkResult objects
        """
        results = []

        # Filter questions if needed
        if category_filter:
            questions = [q for q in questions if q["category"] in category_filter]

        print(f"\nRunning benchmark on {len(questions)} questions...")

        for i, q in enumerate(questions):
            question = q["question"]
            expected_answer = q["answer"]
            category = q["category"]
            evidence = q.get("evidence", [])

            print(f"\n[{i+1}/{len(questions)}] Cat {category}: {question[:60]}...")

            # Get answer from memU
            answer_result = self.answer_question(question, characters)
            generated_answer = answer_result["answer"]
            latency_ms = answer_result["latency_ms"]
            retrieved_content = answer_result["retrieved_content"]

            print(f"  Answer: {generated_answer[:80]}...")
            print(f"  Expected: {expected_answer}")
            print(f"  Latency: {latency_ms:.0f}ms")

            # Evaluate answer
            eval_result = self.evaluate_answer(question, generated_answer, expected_answer)
            is_correct = eval_result["is_correct"]

            status = "✓" if is_correct else "✗"
            print(f"  Result: {status}")

            results.append(BenchmarkResult(
                question=question,
                expected_answer=expected_answer,
                generated_answer=generated_answer,
                is_correct=is_correct,
                category=category,
                latency_ms=latency_ms,
                retrieved_content=retrieved_content,
                evidence=evidence
            ))

        return results

    def compute_summary(self, results: List[BenchmarkResult], total_time: float) -> BenchmarkSummary:
        """Compute summary statistics from results."""
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)

        # Category breakdown
        category_stats = {}
        for r in results:
            cat = r.category
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "correct": 0, "latencies": []}
            category_stats[cat]["total"] += 1
            if r.is_correct:
                category_stats[cat]["correct"] += 1
            category_stats[cat]["latencies"].append(r.latency_ms)

        # Compute accuracies and avg latencies per category
        for cat, stats in category_stats.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            stats["avg_latency_ms"] = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0
            del stats["latencies"]  # Don't include raw latencies in summary

        avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0

        return BenchmarkSummary(
            total_questions=total,
            correct_answers=correct,
            accuracy=correct / total if total > 0 else 0,
            category_stats=category_stats,
            total_time_seconds=total_time,
            avg_latency_ms=avg_latency
        )

    def print_summary(self, summary: BenchmarkSummary):
        """Print formatted summary."""
        print("\n" + "=" * 70)
        print("MEMU BENCHMARK RESULTS")
        print("=" * 70)

        print(f"\nOverall Accuracy: {summary.accuracy:.1%} ({summary.correct_answers}/{summary.total_questions})")
        print(f"Total Time: {summary.total_time_seconds:.1f}s")
        print(f"Avg Latency: {summary.avg_latency_ms:.0f}ms")

        print("\nCategory Breakdown:")
        print("-" * 50)
        print(f"{'Category':<12} {'Correct':<12} {'Accuracy':<12} {'Avg Latency':<12}")
        print("-" * 50)

        for cat in sorted(summary.category_stats.keys()):
            stats = summary.category_stats[cat]
            acc = stats["accuracy"]
            correct = stats["correct"]
            total = stats["total"]
            latency = stats["avg_latency_ms"]
            print(f"{cat:<12} {correct}/{total:<10} {acc:.1%}        {latency:.0f}ms")

        print("=" * 70)

    def save_results(self, results: List[BenchmarkResult], summary: BenchmarkSummary, output_path: Optional[Path] = None):
        """Save results to JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path("results") / f"memu_benchmark_{timestamp}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.chat_model,
            "summary": {
                "total_questions": summary.total_questions,
                "correct_answers": summary.correct_answers,
                "accuracy": summary.accuracy,
                "total_time_seconds": summary.total_time_seconds,
                "avg_latency_ms": summary.avg_latency_ms,
                "category_stats": summary.category_stats
            },
            "results": [
                {
                    "question": r.question,
                    "expected_answer": r.expected_answer,
                    "generated_answer": r.generated_answer,
                    "is_correct": r.is_correct,
                    "category": r.category,
                    "latency_ms": r.latency_ms,
                    "evidence": r.evidence
                }
                for r in results
            ]
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main():
    """Run memU benchmark on conv-26."""
    import argparse

    parser = argparse.ArgumentParser(description="Run memU LoCoMo benchmark")
    parser.add_argument("--model", default="gpt-4o-mini", help="Chat model to use")
    parser.add_argument("--memory-dir", default="benchmarks/memu/memory", help="Memory directory")
    parser.add_argument("--max-workers", type=int, default=5, help="Max parallel workers")
    parser.add_argument("--category", type=str, help="Comma-separated category filter (e.g., '1,2,3')")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocess sessions")
    parser.add_argument("--skip-sessions", action="store_true", help="Skip session processing")
    args = parser.parse_args()

    # Parse category filter
    category_filter = None
    if args.category:
        category_filter = [int(c.strip()) for c in args.category.split(",")]

    # Initialize runner
    runner = MemUBenchmarkRunner(
        chat_model=args.model,
        memory_dir=args.memory_dir,
        max_workers=args.max_workers
    )

    # Load data
    print("Loading conv-26 data...")
    data = runner.load_conv26_data()

    characters = [data["sessions"]["speaker_a"], data["sessions"]["speaker_b"]]
    print(f"Characters: {characters}")

    # Process sessions if needed
    if not args.skip_sessions:
        runner.process_sessions(data["sessions"], force_reprocess=args.force_reprocess)
    else:
        print("Skipping session processing...")

    # Run benchmark
    questions = data["questions"]
    if args.limit:
        questions = questions[:args.limit]

    start_time = time.time()
    results = runner.run_benchmark(questions, characters, category_filter)
    total_time = time.time() - start_time

    # Compute and display summary
    summary = runner.compute_summary(results, total_time)
    runner.print_summary(summary)

    # Save results
    runner.save_results(results, summary)

    return summary


if __name__ == "__main__":
    main()
