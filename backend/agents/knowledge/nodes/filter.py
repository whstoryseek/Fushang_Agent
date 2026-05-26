# -*- coding: utf-8 -*-
"""
Filter Node - 按相关性分数过滤切片（multi_doc 路径专用）
"""

from typing import Dict, Any
from datetime import datetime

from ..state import KnowledgeAgentState


def _score(chunk) -> float:
    """兼容 RetrievedChunk dataclass 和 dict 两种格式取分数"""
    if isinstance(chunk, dict):
        return chunk.get("score", 0.0) or 0.0
    return getattr(chunk, "score", 0.0) or 0.0


def filter_chunks(state: KnowledgeAgentState) -> Dict[str, Any]:
    """
    按相关性分数过滤切片（multi_doc 路径）

    single_doc 路径经 Milvus RRF 融合后直接进入 generate，不经过此节点。
    """
    merged_chunks = state["merged_chunks"]
    config = state["config"]

    print(f"\n[Filter] Filtering {len(merged_chunks)} chunks, threshold={config.vector_score_threshold}")

    try:
        # 根据检索策略选择对应的阈值
        retrieval_strategy = state.get("retrieval_strategy")
        if retrieval_strategy and retrieval_strategy.value == "keyword_only":
            min_score = getattr(config, "keyword_score_threshold", 0.3)
        else:
            min_score = getattr(config, "vector_score_threshold", 0.0)
        filtered = [c for c in merged_chunks if _score(c) >= min_score]

        print(f"[Filter] Kept {len(filtered)}/{len(merged_chunks)} chunks")

        metrics = state["metrics"]
        metrics.chunks_after_filter = len(filtered)

        return {
            "filtered_chunks": filtered,
            "metrics": metrics,
            "processing_log": [{
                "stage": "filter",
                "timestamp": datetime.now().isoformat(),
                "chunks_before": len(merged_chunks),
                "chunks_after": len(filtered),
                "threshold": min_score,
            }]
        }

    except Exception as e:
        print(f"[Filter] Error: {e}")
        return {"all_errors": [f"Filtering failed: {e}"]}
