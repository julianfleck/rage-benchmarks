"""Metrics for retrieval evaluation."""

from typing import List, Dict, Any
import tiktoken


def recall_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    """
    Calculate recall@k.
    
    Args:
        retrieved: List of retrieved frame titles/IDs
        expected: List of expected frame titles/IDs
        k: Number of top results to consider
        
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not expected:
        return 1.0  # No expected results means perfect recall
    
    retrieved_at_k = set(retrieved[:k])
    expected_set = set(expected)
    
    # Count how many expected items were found
    found = len(retrieved_at_k.intersection(expected_set))
    return found / len(expected_set)


def contains_any(text: str, keywords: List[str]) -> bool:
    """
    Check if text contains any of the keywords (case-insensitive).
    
    Args:
        text: Text to search in
        keywords: List of keywords to search for
        
    Returns:
        True if any keyword is found
    """
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer (default: gpt-4)
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def evaluate_retrieval(
    retrieved_frames: List[Dict[str, Any]],
    expected_titles: List[str] = None,
    expected_contains: List[str] = None,
    k_values: List[int] = [1, 5]
) -> Dict[str, Any]:
    """
    Evaluate retrieval results against expectations.
    
    Args:
        retrieved_frames: List of frame dicts with 'title' and 'content'
        expected_titles: Expected frame titles
        expected_contains: Keywords that should appear in content
        k_values: List of k values to compute recall for
        
    Returns:
        Dict with recall@k scores and other metrics
    """
    results = {}
    
    # Extract titles from retrieved frames
    retrieved_titles = [f.get("title", "") for f in retrieved_frames]
    
    # Calculate recall@k for titles
    if expected_titles:
        for k in k_values:
            results[f"recall@{k}_title"] = recall_at_k(retrieved_titles, expected_titles, k)
    
    # Check content containment
    if expected_contains and retrieved_frames:
        # Concatenate all retrieved content
        all_content = "\n".join(f.get("content", "") for f in retrieved_frames[:5])
        results["contains_expected"] = contains_any(all_content, expected_contains)
    
    # Count total tokens
    total_tokens = sum(
        count_tokens(f.get("content", "")) for f in retrieved_frames[:5]
    )
    results["total_tokens"] = total_tokens
    
    return results
