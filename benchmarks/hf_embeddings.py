"""
HuggingFace embeddings for RAGE benchmarks.

Uses sentence-transformers locally (no API key needed).
Model: sentence-transformers/all-MiniLM-L6-v2
- 384-dim, fast, good quality for benchmark purposes
- Runs fully locally via PyTorch
"""

import numpy as np
from typing import List, Optional


class HFEmbedder:
    """
    HuggingFace sentence-transformers embedder.
    
    Wraps SentenceTransformer with an OpenAI-compatible interface
    so it can be swapped into BaselineRAG.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedder.
        
        Args:
            model_name: HuggingFace model ID. Defaults to all-MiniLM-L6-v2
                        which is fast, free, and works offline after first download.
        """
        from sentence_transformers import SentenceTransformer
        print(f"Loading HF model: {model_name}...", flush=True)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ HF model loaded (dim={self.dim})", flush=True)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            numpy array of shape (len(texts), dim)
        """
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    
    def embed_one(self, text: str) -> List[float]:
        """Embed a single text and return as list."""
        emb = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return emb[0].tolist()


# Singleton instance (avoid reloading the model on every call)
_embedder: Optional[HFEmbedder] = None


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HFEmbedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None or _embedder.model_name != model_name:
        _embedder = HFEmbedder(model_name)
    return _embedder
