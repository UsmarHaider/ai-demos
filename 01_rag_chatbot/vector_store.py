"""A tiny vector store, written out in full so there is no magic.

A "vector database" sounds heavy. At this scale it is three things:
  1. a list of text chunks,
  2. a matrix of their embeddings,
  3. cosine similarity to find the closest rows.

That is all this file does. In production you would swap this for Chroma,
pgvector, Pinecone or FAISS - the interface below is deliberately the same
shape (add / search / save / load) so the swap is a one-file change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Hit:
    """One search result."""
    text: str
    source: str
    chunk_id: int
    score: float          # cosine similarity, 1.0 = identical direction


class SimpleVectorStore:
    def __init__(self, backend: str = "unknown") -> None:
        self.texts: list[str] = []
        self.sources: list[str] = []
        self.vectors: np.ndarray | None = None
        # Which embedding backend built this index. Searching an index with a
        # different backend would compare meaningless numbers, so we record it.
        self.backend = backend

    # ------------------------------------------------------------------ build
    def add(self, texts: list[str], sources: list[str], embeddings: list[list[float]]) -> None:
        self.texts.extend(texts)
        self.sources.extend(sources)
        new = np.array(embeddings, dtype=np.float32)
        self.vectors = new if self.vectors is None else np.vstack([self.vectors, new])

    # ----------------------------------------------------------------- search
    def search(self, query_embedding: list[float], k: int = 3) -> list[Hit]:
        """Cosine similarity = dot product of unit-length vectors."""
        if self.vectors is None or len(self.texts) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-10

        m = self.vectors
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-10)

        scores = m @ q                                  # one number per chunk
        top = np.argsort(scores)[::-1][:k]               # highest first

        return [
            Hit(text=self.texts[i], source=self.sources[i], chunk_id=int(i), score=float(scores[i]))
            for i in top
        ]

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "backend": self.backend,
                    "texts": self.texts,
                    "sources": self.sources,
                    "vectors": self.vectors.tolist() if self.vectors is not None else [],
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "SimpleVectorStore":
        data = json.loads(path.read_text(encoding="utf-8"))
        store = cls(backend=data.get("backend", "unknown"))
        store.texts = data["texts"]
        store.sources = data["sources"]
        store.vectors = np.array(data["vectors"], dtype=np.float32) if data["vectors"] else None
        return store

    def __len__(self) -> int:
        return len(self.texts)
