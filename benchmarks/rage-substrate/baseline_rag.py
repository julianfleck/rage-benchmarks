"""Vanilla RAG baseline: simple chunking + embedding + cosine similarity."""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import time

import numpy as np
from openai import OpenAI

# Add rage-substrate to path if it exists locally
RAGE_SUBSTRATE_PATH = Path(__file__).parent.parent.parent / "rage-substrate"
if RAGE_SUBSTRATE_PATH.exists():
    sys.path.insert(0, str(RAGE_SUBSTRATE_PATH))

try:
    from rage_substrate.core.db import SubstrateDB
except ImportError:
    print("Warning: Could not import rage_substrate.core.db - baseline will use mock data", file=sys.stderr)
    SubstrateDB = None


class BaselineRAG:
    """Simple RAG baseline: chunk + embed + retrieve by cosine similarity."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        embedding_model: str = "text-embedding-3-small",
        api_key: str = None
    ):
        """
        Initialize baseline RAG.
        
        Args:
            chunk_size: Target chunk size in tokens
            embedding_model: OpenAI embedding model name
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.chunk_size = chunk_size
        self.embedding_model = embedding_model
        
        # Use OpenRouter API if OPENAI_API_KEY not set
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        base_url = None
        if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            base_url = "https://openrouter.ai/api/v1"
            self.embedding_model = "openai/text-embedding-3-small"
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller segments.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dicts with text and metadata
        """
        # Simple word-based chunking (rough approximation)
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "metadata": metadata or {},
                "chunk_index": i // self.chunk_size
            })
        
        return chunks
    
    def load_from_rage_db(self, db_path: str = None):
        """Load all frames from RAGE database and chunk them."""
        if SubstrateDB is None:
            print("Warning: RAGE db not available, using empty corpus", file=sys.stderr)
            return
        
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "rage-substrate" / "substrate.db")
        
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT id, title, content, summary FROM frames LIMIT 10000")
        rows = cursor.fetchall()
        frames = [dict(row) for row in rows]
        conn.close()
        
        print(f"Loading {len(frames)} frames from RAGE database...", file=sys.stderr)
        
        for frame in frames:
            fid = frame.get("id", "")
            title = frame.get("title", "")
            content = frame.get("content", "") or frame.get("summary", "")
            
            # Combine title and content
            full_text = f"{title}\n\n{content}"
            
            # Chunk the frame
            frame_chunks = self.chunk_text(
                full_text,
                metadata={"fid": fid, "title": title}
            )
            
            self.chunks.extend(frame_chunks)
        
        print(f"Created {len(self.chunks)} chunks", file=sys.stderr)
    
    def embed_chunks(self):
        """Embed all chunks using OpenAI API."""
        if not self.chunks:
            print("Warning: No chunks to embed", file=sys.stderr)
            return
        
        print(f"Embedding {len(self.chunks)} chunks...", file=sys.stderr)
        
        # Embed in batches of 100
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            texts = [chunk["text"] for chunk in batch]
            
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            batch_embeddings = [e.embedding for e in response.data]
            all_embeddings.extend(batch_embeddings)
        
        self.embeddings = np.array(all_embeddings)
        print(f"Embeddings shape: {self.embeddings.shape}", file=sys.stderr)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks by cosine similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of top-k chunks with scores
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        
        # Embed query
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        query_embedding = np.array(response.data[0].embedding)
        
        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return top-k chunks with scores
        results = []
        for idx in top_k_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(similarities[idx])
            results.append(chunk)
        
        return results
    
    def retrieve_as_frames(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve results formatted as frames (for comparison with RAGE).
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of frame-like dicts with title and content
        """
        chunks = self.retrieve(query, k=k * 3)  # Get more chunks to deduplicate
        
        # Group chunks by frame
        frames_dict = {}
        for chunk in chunks:
            fid = chunk["metadata"].get("fid", "unknown")
            if fid not in frames_dict:
                frames_dict[fid] = {
                    "title": chunk["metadata"].get("title", "Untitled"),
                    "content": chunk["text"],
                    "fid": fid,
                    "score": chunk["score"]
                }
        
        # Return top-k frames by score
        frames = sorted(frames_dict.values(), key=lambda f: f["score"], reverse=True)
        return frames[:k]


def main():
    """Test the baseline RAG."""
    baseline = BaselineRAG()
    baseline.load_from_rage_db()
    baseline.embed_chunks()
    
    # Test query
    query = "What is the BNF grammar for RAGE addresses?"
    results = baseline.retrieve_as_frames(query, k=5)
    
    print(f"\nQuery: {query}\n")
    for i, frame in enumerate(results, 1):
        print(f"{i}. {frame['title']} (score: {frame['score']:.4f})")
        print(f"   {frame['content'][:200]}...\n")


if __name__ == "__main__":
    main()
