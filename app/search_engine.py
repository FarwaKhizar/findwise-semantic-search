"""
search_engine.py
----------------
Three ways to search a product catalog, side by side:

  1. Keyword  (BM25)          — classic term matching. Fast, but literal.
  2. Semantic (embeddings)    — matches by meaning, catches synonyms/paraphrase.
  3. Hybrid   (RRF fusion)    — fuses keyword + semantic rankings, then the
                                combined order surfaces the best of both.

Everything runs locally and offline:
- Embeddings: sentence-transformers (no API key, no cost).
- Keyword: a compact BM25 implementation (no external service).
- Fusion: Reciprocal Rank Fusion (RRF) — a simple, robust way to combine
          ranked lists without tuning score scales.

Design note: this mirrors production hybrid search (e.g. Elasticsearch BM25 +
a vector index), just distilled to the essentials so the ranking behavior is
easy to see and explain.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ------------------------------- Data types -------------------------------- #

@dataclass
class Product:
    id: int
    title: str
    description: str
    category: str
    price: float

    @property
    def text(self) -> str:
        return f"{self.title}. {self.description}"


@dataclass
class Result:
    product: Product
    score: float          # method-specific score (normalized 0..1 for display)
    rank: int             # 1-indexed position in this method's list


# --------------------------- Tokenizer / BM25 ------------------------------ #

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Minimal BM25 keyword ranker (Okapi variant)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.N, 1)
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for d in docs:
            freq: dict[str, int] = {}
            for term in d:
                freq[term] = freq.get(term, 0) + 1
            self.tf.append(freq)
            for term in freq:
                self.df[term] = self.df.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        # standard BM25 idf with +1 smoothing to stay non-negative
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.N, dtype="float32")
        for i, freq in enumerate(self.tf):
            dl = sum(freq.values())
            s = 0.0
            for term in query:
                if term not in freq:
                    continue
                idf = self._idf(term)
                tf = freq[term]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * (tf * (self.k1 + 1)) / denom
            out[i] = s
        return out


# ------------------------------- The engine -------------------------------- #

class SearchEngine:
    """Builds keyword + semantic indexes over a product catalog and searches
    them three ways."""

    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2"):
        self._embedder = SentenceTransformer(embed_model_name)
        self.products: list[Product] = []
        self._bm25: BM25 | None = None
        self._embeddings: np.ndarray | None = None

    # ---- Load ---- #

    def load_csv(self, path: str) -> int:
        df = pd.read_csv(path)
        self.products = [
            Product(
                id=int(r.id), title=str(r.title), description=str(r.description),
                category=str(r.category), price=float(r.price),
            )
            for r in df.itertuples(index=False)
        ]
        self._build()
        return len(self.products)

    def _build(self) -> None:
        texts = [p.text for p in self.products]
        # keyword index
        self._bm25 = BM25([_tokenize(t) for t in texts])
        # semantic index (normalized for cosine via dot product)
        self._embeddings = self._embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    # ---- Individual methods ---- #

    def keyword_search(self, query: str, k: int = 5) -> list[Result]:
        scores = self._bm25.scores(_tokenize(query))
        return self._rank(scores, k)

    def semantic_search(self, query: str, k: int = 5) -> list[Result]:
        q = self._embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        scores = (self._embeddings @ q[0])  # cosine similarity
        return self._rank(scores, k)

    def hybrid_search(
        self, query: str, k: int = 5, rrf_k: int = 60,
        w_keyword: float = 1.0, w_semantic: float = 2.0,
    ) -> list[Result]:
        """Weighted Reciprocal Rank Fusion of keyword + semantic rankings.

        RRF score for a doc = sum over methods of weight / (rrf_k + rank).
        It needs only the *ranks* from each method, so it's robust to the two
        methods having totally different score scales. Semantic is weighted
        higher here because on natural-language queries it is usually the
        stronger ranker, with keyword adding exact-term precision — a common
        production default.
        """
        kw = self._bm25.scores(_tokenize(query))
        q = self._embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        sem = (self._embeddings @ q[0])

        kw_order = np.argsort(-kw)
        sem_order = np.argsort(-sem)
        kw_rank = {idx: r for r, idx in enumerate(kw_order)}
        sem_rank = {idx: r for r, idx in enumerate(sem_order)}

        fused = np.zeros(len(self.products), dtype="float32")
        for i in range(len(self.products)):
            fused[i] = (
                w_keyword / (rrf_k + kw_rank[i])
                + w_semantic / (rrf_k + sem_rank[i])
            )
        return self._rank(fused, k)

    # ---- Shared ranking / normalization ---- #

    def _rank(self, scores: np.ndarray, k: int) -> list[Result]:
        order = np.argsort(-scores)[:k]
        top = scores[order]
        # normalize to 0..1 for readable display (relative within this result set)
        lo, hi = float(top.min()), float(top.max())
        span = (hi - lo) or 1.0
        results = []
        for rank, idx in enumerate(order, start=1):
            norm = (float(scores[idx]) - lo) / span
            results.append(Result(product=self.products[idx], score=norm, rank=rank))
        return results

    def search_all(self, query: str, k: int = 5) -> dict[str, list[Result]]:
        """Run all three methods for a side-by-side comparison."""
        return {
            "Keyword (BM25)": self.keyword_search(query, k),
            "Semantic": self.semantic_search(query, k),
            "Hybrid (RRF)": self.hybrid_search(query, k),
        }

    @property
    def size(self) -> int:
        return len(self.products)
