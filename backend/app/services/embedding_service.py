# -*- coding: utf-8 -*-
"""
Standard text embedding service.

Volces is the default provider. DashScope is kept as an explicit
compatibility path for deployments that still opt into it.
"""

import logging
import time
from typing import List, Optional

import dashscope
import httpx
from dashscope import TextEmbedding

from app.core.config import settings

logger = logging.getLogger(__name__)

VOLCES_DEFAULT_EMBEDDING_MODEL = "doubao-embedding-text-240715"
LEGACY_DASHSCOPE_MODELS = {"text-embedding-v3", "text-embedding-v4"}


class EmbeddingService:
    def __init__(self):
        self.provider = settings.embedding_provider.lower()
        self.model = settings.embedding_model
        self.batch_size = settings.embedding_batch_size
        self.dimension = settings.embedding_dimension
        self.api_key = settings.volces_api_key
        self.base_url = settings.volces_base_url.rstrip("/")
        dashscope.api_key = settings.dashscope_api_key

    def embed_texts(self, texts: List[str], dimension: Optional[int] = None) -> List[List[float]]:
        if not texts:
            return []

        dim = dimension or self.dimension
        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            vectors = self._embed_batch(batch, dimension=dim)
            all_vectors.extend(vectors)

        return all_vectors

    def embed_query(self, text: str, dimension: Optional[int] = None) -> List[float]:
        dim = dimension or self.dimension
        vectors = self._embed_batch([text], dimension=dim)
        return vectors[0] if vectors else []

    def _embed_batch(
        self,
        texts: List[str],
        retry: int = 5,
        dimension: Optional[int] = None,
    ) -> List[List[float]]:
        if self.provider == "dashscope":
            return self._embed_batch_dashscope(texts, retry=retry, dimension=dimension)
        if self.provider == "volces":
            return self._embed_batch_volces(texts, retry=retry, dimension=dimension)
        raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _resolve_model(self, model: str) -> str:
        if self.provider == "volces" and model in LEGACY_DASHSCOPE_MODELS:
            logger.warning(
                "Remapping legacy DashScope embedding model %s to %s for Volces provider",
                model,
                VOLCES_DEFAULT_EMBEDDING_MODEL,
            )
            return VOLCES_DEFAULT_EMBEDDING_MODEL
        return model

    def _embed_batch_volces(
        self,
        texts: List[str],
        retry: int = 5,
        dimension: Optional[int] = None,
    ) -> List[List[float]]:
        payload = {
            "model": self._resolve_model(self.model),
            "input": texts,
            "encoding_format": "float",
        }
        if dimension is not None:
            payload["dimensions"] = dimension

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(retry):
            try:
                resp = httpx.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json().get("data") or []
                if not isinstance(data, list) or not data:
                    raise KeyError("data")
                data.sort(key=lambda item: item.get("index", item.get("text_index", 0)))
                return [item["embedding"] for item in data]
            except Exception as e:
                if attempt < retry - 1:
                    wait = min(2 ** attempt * 2, 60)
                    logger.warning(
                        "Embedding call failed, retrying in %ss (%s/%s): %s",
                        wait,
                        attempt + 1,
                        retry,
                        e,
                    )
                    time.sleep(wait)
                else:
                    logger.error("Embedding call failed permanently: %s", e)
                    raise
        return []

    def _embed_batch_dashscope(
        self,
        texts: List[str],
        retry: int = 5,
        dimension: Optional[int] = None,
    ) -> List[List[float]]:
        for attempt in range(retry):
            try:
                resp = TextEmbedding.call(
                    model=self.model,
                    input=texts,
                    dimension=dimension,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"DashScope embedding failed: {resp.message}")
                embeddings = resp.output["embeddings"]
                embeddings.sort(key=lambda item: item["text_index"])
                return [item["embedding"] for item in embeddings]
            except Exception as e:
                if attempt < retry - 1:
                    wait = min(2 ** attempt * 2, 60)
                    logger.warning(
                        "Embedding call failed, retrying in %ss (%s/%s): %s",
                        wait,
                        attempt + 1,
                        retry,
                        e,
                    )
                    time.sleep(wait)
                else:
                    logger.error("Embedding call failed permanently: %s", e)
                    raise
        return []


_instance = None


def get_embedding_service() -> EmbeddingService:
    global _instance
    if _instance is None:
        _instance = EmbeddingService()
    return _instance
