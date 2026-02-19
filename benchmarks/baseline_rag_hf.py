"""
HuggingFace-powered RAG baseline.

Same logic as baseline_rag.py but uses sentence-transformers instead of OpenAI.
Fully local — no API key required.
"""

import sys
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from benchmarks.hf_embeddings import get_embedder

# Add rage-substrate to path
RAGE_SUBSTRATE_PATH = Path(__file__).parent.parent.parent / "rage-substrate"
if RAGE_SUBSTRATE_PATH.exists():
    sys.path.insert(0, str(RAGE_SUBSTRATE_PATH))


class BaselineRAGHF:
    """
    Vanilla RAG baseline using HuggingFace sentence-transformers.
    
    Drops-in for BaselineRAG but with local HF embeddings.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.chunk_size = chunk_size
        self.model_name = model_name
        self.embedder = get_embedder(model_name)
        
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk text into ~chunk_size-word segments."""
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
        if db_path is None:
            db_path = str(RAGE_SUBSTRATE_PATH / "substrate.db")
        
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
            full_text = f"{title}\n\n{content}"
            
            frame_chunks = self.chunk_text(
                full_text,
                metadata={"fid": fid, "title": title}
            )
            self.chunks.extend(frame_chunks)
        
        print(f"Created {len(self.chunks)} chunks", file=sys.stderr)
    
    def embed_chunks(self):
        """Embed all chunks using HuggingFace sentence-transformers."""
        if not self.chunks:
            print("Warning: No chunks to embed", file=sys.stderr)
            return
        
        print(f"Embedding {len(self.chunks)} chunks with HF model ({self.model_name})...",
              file=sys.stderr)
        
        texts = [chunk["text"] for chunk in self.chunks]
        self.embeddings = self.embedder.embed_texts(texts)
        
        print(f"Embeddings shape: {self.embeddings.shape}", file=sys.stderr)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks by cosine similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        
        # Embed query (normalize=True so dot product = cosine sim)
        query_emb = self.embedder.embed_texts([query])[0]
        
        # Cosine similarity (embeddings are normalized)
        similarities = self.embeddings @ query_emb
        
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(similarities[idx])
            results.append(chunk)
        
        return results
    
    def retrieve_as_frames(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve results formatted as frames."""
        chunks = self.retrieve(query, k=k * 3)
        
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
        
        frames = sorted(frames_dict.values(), key=lambda f: f["score"], reverse=True)
        return frames[:k]
