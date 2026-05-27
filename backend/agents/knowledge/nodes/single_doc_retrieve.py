# -*- coding: utf-8 -*-
"""
Single Document Retrieve Node - 单文档查询
针对特定文档进行检索,返回top 10结果
"""

import logging
from datetime import datetime
from typing import List

from ..state import KnowledgeAgentState, RetrievedChunk, RetrievalStrategy
from ..services.retrieval import get_retrieval_service
from app.services.retrieval_bucket import search_with_bucket_fallback

logger = logging.getLogger(__name__)


async def single_doc_retrieve(state: KnowledgeAgentState) -> dict:
    """
    单文档查询节点（支持知识图谱并行检索）
    """
    import asyncio
    start_time = datetime.now()

    try:
        query = state["rewritten_query"]
        retrieval_strategy = state.get("retrieval_strategy", RetrievalStrategy.HYBRID)
        _cfg = state.get("config")
        collection = _cfg.collection if _cfg else None

        ranker       = _cfg.ranker          if _cfg else "RRF"
        rrf_k        = _cfg.rrf_k           if _cfg else 60
        hybrid_alpha = _cfg.hybrid_alpha     if _cfg else 0.5
        top_k        = _cfg.single_doc_top_k if _cfg else 20
        keyword_filter = _cfg.keyword_filter if _cfg else None

        # 多模态知识库走专用检索节点
        if getattr(_cfg, "kb_type", "standard") == "multimodal":
            from .multimodal_retrieve import multimodal_retrieve
            return await multimodal_retrieve(state)

        logger.info(f"[SingleDocRetrieve] 开始单文档检索: {query}")
        logger.info(f"[SingleDocRetrieve] 检索策略: {retrieval_strategy.value}, collection={collection}, ranker={ranker}, top_k={top_k}")

        retrieval_service = get_retrieval_service()
        kg_already_routed = state.get("kg_deep_traversal") is not None
        kg_enabled = getattr(_cfg, "kg_enabled", False) and not kg_already_routed

        # ── 并行执行 Milvus 检索 与 知识图谱检索 ────────────────────────────
        def _milvus_search(filter_expr=None):
            if retrieval_strategy == RetrievalStrategy.KEYWORD_ONLY:
                return retrieval_service.keyword_search(
                    query=query, top_k=top_k, collection=collection,
                    filter_expr=filter_expr,
                    keyword_filter=keyword_filter, ranker=ranker, rrf_k=rrf_k, hybrid_alpha=hybrid_alpha,
                )
            return retrieval_service.hybrid_search(
                query=query, top_k=top_k, collection=collection,
                filter_expr=filter_expr,
                ranker=ranker, rrf_k=rrf_k, hybrid_alpha=hybrid_alpha,
            )

        milvus_task = asyncio.to_thread(
            search_with_bucket_fallback,
            query=query,
            top_k=top_k,
            search_fn=_milvus_search,
        )

        if kg_enabled:
            from .graph_retrieve import async_graph_retrieve
            graph_task = async_graph_retrieve(state)
            (chunks, bucket_log), graph_result = await asyncio.gather(milvus_task, graph_task)
            graph_chunks = graph_result.get("kg_graph_chunks", [])
            graph_log = graph_result.get("processing_log", [])
            graph_warnings = graph_result.get("all_warnings", [])
        else:
            chunks, bucket_log = await milvus_task
            graph_chunks = []
            graph_log = []
            graph_warnings = []

        duration = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(f"[SingleDocRetrieve] 检索完成 ({duration:.0f}ms): Milvus={len(chunks)} 条, 图谱={len(graph_chunks)} 条")

        metrics = state["metrics"]
        metrics.retrieval_duration_ms = duration
        metrics.total_chunks_retrieved = len(chunks)

        result = {
            "merged_chunks": chunks,
            "total_candidates": len(chunks),
            "retrieval_strategy_used": retrieval_strategy,
            "metrics": metrics,
            "processing_log": [
                {"stage": "single_doc_retrieve", "duration_ms": duration,
                 "chunks_count": len(chunks), "strategy": retrieval_strategy.value,
                 "preferred_bucket": bucket_log.get("preferred_bucket"),
                 "bucket_fallback": bucket_log.get("bucket_fallback", False),
                 "fallback_reason": bucket_log.get("fallback_reason"),
                 "bucket_hits": bucket_log.get("bucket_hits", 0)}
            ] + graph_log,
        }
        if graph_chunks:
            result["kg_graph_chunks"] = graph_chunks
        if graph_warnings:
            result["all_warnings"] = graph_warnings
        return result

    except Exception as e:
        logger.error(f"[SingleDocRetrieve] 检索失败: {e}", exc_info=True)
        return {
            "merged_chunks": [],
            "total_candidates": 0,
            "all_errors": [f"单文档检索失败: {e}"],
            "error": str(e),
            "error_stage": "single_doc_retrieve",
        }
