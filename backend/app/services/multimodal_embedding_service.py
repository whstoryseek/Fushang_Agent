# -*- coding: utf-8 -*-
"""
多模态 Embedding 服务
支持火山引擎 doubao-embedding-vision-251215 和 DashScope qwen3-vl-embedding。
通过 MULTIMODAL_EMBEDDING_PROVIDER 环境变量切换（默认 volces）。

注意：切换多模态嵌入模型后，已有 Milvus 中的多模态向量空间将不兼容，
      需要重新创建/索引多模态知识库。
"""
import base64
import logging
import time
from typing import List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# DashScope 保留模型
DASHSCOPE_MULTIMODAL_MODEL = "qwen3-vl-embedding"


class MultimodalEmbeddingService:
    """多模态嵌入服务（火山引擎 / DashScope 可配置）"""

    def __init__(self):
        self.provider = settings.multimodal_embedding_provider.lower()
        self.dimension = settings.multimodal_embedding_dimension

        if self.provider == "volces":
            self.model = settings.multimodal_embedding_model
            self.api_key = settings.volces_api_key
            self.base_url = settings.volces_base_url
            logger.info(f"[MultimodalEmbed] 使用火山引擎: {self.model}, dim={self.dimension}")

        if self.provider == "dashscope":
            import dashscope
            from dashscope import MultiModalEmbedding
            dashscope.api_key = settings.dashscope_api_key
            self.dashscope_model = DASHSCOPE_MULTIMODAL_MODEL
            self.MultiModalEmbedding = MultiModalEmbedding
            logger.info(f"[MultimodalEmbed] 使用 DashScope: {self.dashscope_model}, dim={self.dimension}")

    # ── 公共接口 ─────────────────────────────────────────────────────────────

    def embed_text(self, text: str, dimension: Optional[int] = None, retry: int = 3) -> List[float]:
        """对文本生成向量"""
        dim = dimension or self.dimension
        if self.provider == "volces":
            return self._embed_volces([{"type": "text", "text": text}], dimension=dim, retry=retry)
        return self._embed_dashscope([{"text": text}], dim, retry)

    def embed_texts(self, texts: List[str], dimension: Optional[int] = None) -> List[List[float]]:
        """批量文本向量化（逐条调用）"""
        return [self.embed_text(t, dimension) for t in texts]

    def embed_image(self, image_url: str, dimension: Optional[int] = None, retry: int = 3) -> Optional[List[float]]:
        """对图片 URL 生成向量"""
        dim = dimension or self.dimension
        try:
            if self.provider == "volces":
                return self._embed_volces([{
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }], dimension=dim, retry=retry)
            return self._embed_dashscope([{"image": image_url}], dim, retry)
        except Exception as e:
            logger.warning(f"[MultimodalEmbed] 图片向量化失败，跳过: {image_url[:80]}, error={e}")
            return None

    def embed_image_bytes(self, image_bytes: bytes, dimension: Optional[int] = None, retry: int = 3) -> Optional[List[float]]:
        """用 base64 data URI 直接传图片字节，不依赖外部 URL 可访问性"""
        dim = dimension or self.dimension
        try:
            data_uri = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
            if self.provider == "volces":
                return self._embed_volces([{
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                }], dimension=dim, retry=retry)
            return self._embed_dashscope([{"image": data_uri}], dim, retry)
        except Exception as e:
            logger.warning(f"[MultimodalEmbed] 图片字节向量化失败: {e}")
            return None

    def embed_text_and_image(
        self, text: str, image_url: str, dimension: Optional[int] = None
    ) -> dict:
        """同时生成文本和图片向量"""
        dim = dimension or self.dimension
        if self.provider == "volces":
            # 火山引擎多模态嵌入同时传入 text + image 返回联合向量
            vec = self._embed_volces([
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}}
            ], dimension=dim)
            return {"text": vec, "image": vec}
        return {
            "text": self._embed_dashscope([{"text": text}], dim),
            "image": self._embed_dashscope([{"image": image_url}], dim),
        }

    # ── 内部实现 ─────────────────────────────────────────────────────────────

    def _embed_volces(
        self,
        inputs: list,
        dimension: Optional[int] = None,
        retry: int = 3,
    ) -> List[float]:
        """调用火山引擎多模态嵌入（Seed1.6-Embedding）
        
        模型规格：
        - 默认维度 2048，支持 dimensions 参数降维到 1024
        - 支持文本、图像、视频任意混合输入
        - 图像格式：JPEG/PNG/WEBP/BMP/TIFF 等，边长 [10, 6000] px
        - 视频：≤50MB，MP4/AVI/MOV，支持 FPS 0.2-5
        - 上下文限制：128k
        """
        import httpx

        dim = dimension or self.dimension
        payload = {"model": self.model, "input": inputs}
        # 若配置的维度不是默认 2048，传入 dimensions 参数降维
        if dim != 2048:
            payload["dimensions"] = dim

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(retry):
            try:
                resp = httpx.post(
                    f"{self.base_url}/embeddings/multimodal",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                # OpenAI 兼容格式: data["data"][0]["embedding"]
                data_field = data.get("data")
                if isinstance(data_field, dict) and "embedding" in data_field:
                    return data_field["embedding"]
                if isinstance(data_field, list) and data_field:
                    return data_field[0]["embedding"]
                raise KeyError("embedding")
            except Exception as e:
                if attempt < retry - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"[MultimodalEmbed] Volces 调用失败，{wait}s 后重试 ({attempt+1}/{retry}): {e}"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"[MultimodalEmbed] Volces 最终失败: {e}")
                    raise
        return []

    def _embed_dashscope(self, inputs: list, dimension: int, retry: int = 3) -> List[float]:
        """调用 DashScope 多模态嵌入（保留兼容）"""
        for attempt in range(retry):
            try:
                resp = self.MultiModalEmbedding.call(
                    api_key=settings.dashscope_api_key,
                    model=self.dashscope_model,
                    input=inputs,
                    enable_fusion=False,  # 独立向量模式
                    dimension=dimension,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"qwen3-vl-embedding 调用失败: {resp.message}")
                return resp.output["embeddings"][0]["embedding"]
            except Exception as e:
                if attempt < retry - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"[MultimodalEmbed] DashScope 调用失败，{wait}s 后重试 ({attempt+1}/{retry}): {e}"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"[MultimodalEmbed] DashScope 最终失败: {e}")
                    raise
        return []


# 全局单例
_instance: Optional[MultimodalEmbeddingService] = None


def get_multimodal_embedding_service() -> MultimodalEmbeddingService:
    global _instance
    if _instance is None:
        _instance = MultimodalEmbeddingService()
    return _instance
