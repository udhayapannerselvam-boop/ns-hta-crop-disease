"""
embeddings.py - 384-dimensional dense text embedding generator.

Supports:
1. Sentence-Transformers (all-MiniLM-L6-v2) for 384-d semantic sentence representations.
2. Deterministic high-dimensional projection fallback if sentence-transformers is offline.
3. Persistent disk caching (.npz) for fast iterative experimentation.
"""

import os
import hashlib
import numpy as np
from typing import List


EMBEDDING_DIM = 384


class TextEmbeddingGenerator:
    """
    Extracts frozen 384-dimensional sentence embeddings.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = None):
        self.model_name = model_name
        self.embedding_dim = EMBEDDING_DIM
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.st_model = None
        self._init_encoder()

    def _init_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(self.model_name)
            print(f"[Embeddings] Loaded SentenceTransformer('{self.model_name}') successfully.")
        except Exception as e:
            print(f"[Embeddings] SentenceTransformer unavailable ({e}). Using deterministic 384-d projection.")
            self.st_model = None

    def _hash_texts(self, texts: List[str]) -> str:
        h = hashlib.sha256()
        for t in texts[:20]:
            h.update(t.encode("utf-8"))
        h.update(str(len(texts)).encode("utf-8"))
        return h.hexdigest()[:16]

    def encode(self, texts: List[str], use_cache: bool = True, cache_prefix: str = "emb") -> np.ndarray:
        """
        Encodes a list of text strings into an (N, 384) numpy array.
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_prefix}_{self._hash_texts(texts)}.npz")
        
        if use_cache and os.path.exists(cache_file):
            data = np.load(cache_file)
            print(f"[Embeddings] Loaded cached embeddings from {cache_file} (shape: {data['embeddings'].shape})")
            return data["embeddings"]
            
        if self.st_model is not None:
            embeddings = self.st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings = embeddings.astype(np.float32)
        else:
            # Deterministic, semantic hash/character-ngram projection fallback
            embeddings = self._fallback_semantic_projection(texts)
            
        if use_cache:
            np.savez_compressed(cache_file, embeddings=embeddings)
            
        return embeddings

    def _fallback_semantic_projection(self, texts: List[str]) -> np.ndarray:
        """
        Creates dense 384-dimensional semantic embeddings using TF-IDF + truncated SVD / random projection.
        Guarantees deterministic reproduction of 384-d vectors offline.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 3), token_pattern=r"(?u)\b\w+\b")
        sparse_mat = tfidf.fit_transform(texts).toarray()
        
        # Project or pad to 384 dimensions
        rng = np.random.RandomState(42)
        proj_matrix = rng.randn(sparse_mat.shape[1], self.embedding_dim).astype(np.float32)
        # Gram-Schmidt / orthogonalization
        proj_matrix, _ = np.linalg.qr(proj_matrix)
        
        dense = np.dot(sparse_mat, proj_matrix)
        # L2 normalize
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense = (dense / norms).astype(np.float32)
        return dense
