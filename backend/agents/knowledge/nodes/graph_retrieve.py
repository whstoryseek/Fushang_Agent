# -*- coding: utf-8 -*-
"""
每次请求执行知识图谱检索（WhyHow SDK），结果写入 kg_graph_chunks（供 generate 分节组装）
"""
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from typing import Any, Dict

from ..state import KnowledgeAgentState

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kg_whyhow")


def _resolve_graph_id(collection: str, cfg) -> str | None:
    """从 retrieval_config 或 collection 名称推断 WhyHow graph_id"""
    # 优先用 retrieval_config 里的显式 graph_id
    explicit = getattr(cfg, "kg_graph_id", None)
    if explicit:
        return str(explicit)
    # 降级：用 collection 名称当 graph_name（WhyHow 按名称查找）
    return collection if collection else None


async def async_graph_retrieve(state: KnowledgeAgentState) -> dict:
    """异步执行知识图谱检索，供 retrieve 节点内部并行调用。"""
    import asyncio
    start = datetime.now()
    cfg = state.get("config")
    if not cfg or not getattr(cfg, "kg_enabled", False):
        return {
            "kg_graph_chunks": [],
            "processing_log": [{"stage": "graph_retrieve", "skipped": True, "reason": "kg_disabled"}],
        }

    collection = getattr(cfg, "collection", None) or ""
    graph_id = _resolve_graph_id(collection, cfg)
    if not graph_id:
        return {
            "kg_graph_chunks": [],
            "processing_log": [{"stage": "graph_retrieve", "skipped": True, "reason": "no_graph_id"}],
        }

    query = state.get("rewritten_query") or state.get("query") or ""
    top_k = int(getattr(cfg, "kg_top_k", 5) or 5)
    timeout = float(getattr(cfg, "kg_timeout_seconds", 2.0) or 2.0)

    from app.services.kg_whyhow_service import get_kg_retrieval_service
    svc = get_kg_retrieval_service()

    def _run():
        return svc.query_graph(
            graph_id=graph_id,
            query=query,
            top_k=top_k,
            timeout=timeout,
        )

    logger.info("[GraphRetrieve] START | graph=%s query=%r top_k=%d timeout=%.1fs",
                 graph_id, query[:80] if query else "(empty)", top_k, timeout)

    chunks: list[dict] = []
    try:
        chunks = await asyncio.wait_for(
            asyncio.to_thread(_run),
            timeout=timeout,
        )
        logger.info("[GraphRetrieve] SUCCESS | graph=%s query=%r → returned %d chunks",
                    graph_id, query[:80] if query else "(empty)", len(chunks))
        if chunks:
            logger.debug("[GraphRetrieve] chunks detail: %s",
                         [{"id": c.get("chunk_id"), "score": c.get("unified_score"),
                           "relations": c.get("relation_types")} for c in chunks])
    except asyncio.TimeoutError:
        logger.warning("[GraphRetrieve] WhyHow 查询超时 %.1fs graph=%s", timeout, graph_id)
        return {
            "kg_graph_chunks": [],
            "all_warnings": [f"graph_retrieve_timeout:{timeout}s"],
            "processing_log": [
                {
                    "stage": "graph_retrieve",
                    "duration_ms": (datetime.now() - start).total_seconds() * 1000,
                    "timeout": True,
                }
            ],
        }
    except Exception as e:
        logger.exception("[GraphRetrieve] WhyHow 查询失败")
        return {
            "kg_graph_chunks": [],
            "all_warnings": [f"graph_retrieve:{e}"],
        }

    ms = (datetime.now() - start).total_seconds() * 1000
    return {
        "kg_graph_chunks": chunks,
        "processing_log": [
            {
                "stage": "graph_retrieve",
                "duration_ms": ms,
                "chunk_count": len(chunks),
                "graph_id": graph_id,
            }
        ],
    }


def graph_retrieve(state: KnowledgeAgentState) -> dict:
    """同步包装（供 LangGraph 节点调用，保持向后兼容）。
    注意：新链路中 retrieve 节点内部直接调用 async_graph_retrieve 实现并行。"""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # 已有事件循环：创建 task 执行（但无法直接 await，返回空结果供兼容性）
        # 实际上在新链路中此节点不会被调用
        return {"kg_graph_chunks": [], "processing_log": [{"stage": "graph_retrieve", "skipped": True, "reason": "use_async_variant"}]}
    except RuntimeError:
        # 无事件循环：同步执行
        return asyncio.run(async_graph_retrieve(state))
