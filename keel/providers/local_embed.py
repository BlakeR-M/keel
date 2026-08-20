"""Local embeddings through fastembed (ONNX on the CPU). Implements `EmbeddingProvider`.

The model loads on first use. Cached models load without network; air-gap mode restricts loading to
the local cache.
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "nomic-ai/nomic-embed-text-v1.5": 768,
}


class FastembedEmbeddings:
    """Dense text embeddings from a fastembed model, loaded lazily on the first call."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        local_files_only: bool = False,
        batch_size: int = 64,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.name = f"fastembed:{model_name}"
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self.threads = threads
        self._model: Any = None
        self._dim: int | None = _KNOWN_DIMS.get(model_name)

    @property
    def dim(self) -> int:
        """Vector dimension. Known models answer without loading; others load the model once."""
        if self._dim is None:
            self._dim = self._lookup_dim() or len(self.embed(["dimension probe"])[0])
        return self._dim

    def _lookup_dim(self) -> int | None:
        from fastembed import TextEmbedding

        for entry in TextEmbedding.list_supported_models():
            if entry.get("model") == self.model_name:
                return int(entry["dim"])
        return None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            kwargs: dict[str, Any] = {"lazy_load": False}
            if self.threads is not None:
                kwargs["threads"] = self.threads
            if self.local_files_only:
                kwargs["local_files_only"] = True
            self._model = TextEmbedding(self.model_name, **kwargs)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed passages. Returns one vector per input, in order."""
        if not texts:
            return []
        model = self._load()
        vectors = [vec.astype(float).tolist() for vec in model.embed(list(texts), batch_size=self.batch_size)]
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. bge-small needs no query prefix; fastembed applies one for models that do."""
        model = self._load()
        vector = next(iter(model.query_embed(text)))
        result = vector.astype(float).tolist()
        if self._dim is None:
            self._dim = len(result)
        return result
