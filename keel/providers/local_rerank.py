"""Local cross-encoder reranking through fastembed (ONNX on the CPU). Implements `Reranker`.

The model loads on first use. Scores are the cross-encoder's raw relevance logits (higher is more
relevant; ms-marco MiniLM scores relevant passages near or above zero and unrelated ones well below).
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from keel.providers.base import ChunkHit

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class FastembedReranker:
    """Cross-encoder reranker; returns hits sorted by the model's score, highest first."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        *,
        local_files_only: bool = False,
        batch_size: int = 32,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.name = f"fastembed:{model_name}"
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self.threads = threads
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            kwargs: dict[str, Any] = {}
            if self.threads is not None:
                kwargs["threads"] = self.threads
            if self.local_files_only:
                kwargs["local_files_only"] = True
            self._model = TextCrossEncoder(self.model_name, **kwargs)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Cross-encoder score for each (query, text) pair, in input order."""
        if not texts:
            return []
        model = self._load()
        return [float(s) for s in model.rerank(query, list(texts), batch_size=self.batch_size)]

    def rerank(self, query: str, hits: list[ChunkHit]) -> list[ChunkHit]:
        """Re-score hits against the query and return copies sorted by that score, highest first.
        Ties keep their input order."""
        if not hits:
            return []
        scores = self.score(query, [hit.text for hit in hits])
        rescored = [replace(hit, score=score) for hit, score in zip(hits, scores, strict=True)]
        return sorted(rescored, key=lambda h: -h.score)
