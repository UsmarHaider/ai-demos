"""Where embeddings come from.

Embeddings are a separate model from the chat model, so RAG needs its
embeddings from somewhere else. Two backends, chosen by EMBED_BACKEND in .env:

  gemini  - Google's gemini-embedding-001. Real semantic embeddings: two
            passages that *mean* the same thing land near each other even if
            they share no words. 3072 dimensions. Needs a key and network.

  local   - a keyword embedding computed here in numpy. No network, no key,
            no quota, nothing to install. It matches on shared WORDS rather
            than shared meaning, so it is weaker - but it never goes down,
            which makes it a good insurance policy for a live demo.

Both expose the same two methods, `embed_documents` and `embed_query`, so the
rest of the code does not care which one it is talking to.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from shared import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")
LOCAL_DIM = 2048


class LocalKeywordEmbeddings:
    """A hashing TF embedding, written out so students can see there is no magic.

    For each piece of text we:
      1. lowercase it and pull out the words,
      2. add word pairs too, so "annual leave" differs from "leave" alone,
      3. hash every term into one of 2048 buckets,
      4. count them, damp the counts with 1 + log(count),
      5. scale the vector to length 1 so cosine similarity behaves.

    This is the classic "hashing trick". It needs no vocabulary file, so a
    question asked later is embedded exactly the same way as the documents
    were during ingestion.
    """

    def _terms(self, text: str) -> list[str]:
        words = _TOKEN_RE.findall(text.lower())
        pairs = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return words + pairs

    def _vector(self, text: str) -> list[float]:
        counts: Counter[int] = Counter()
        for term in self._terms(text):
            digest = hashlib.md5(term.encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % LOCAL_DIM
            counts[bucket] += 1

        vec = np.zeros(LOCAL_DIM, dtype=np.float32)
        for bucket, count in counts.items():
            vec[bucket] = 1.0 + math.log(count)

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def get_embedder():
    """Return the embedding backend named in .env."""
    config.require_embeddings()

    if config.EMBED_BACKEND == "local":
        return LocalKeywordEmbeddings()

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL)
