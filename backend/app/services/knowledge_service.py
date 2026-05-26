# -*- coding: utf-8 -*-
"""
Knowledge RAG 业务逻辑（Knowledge Agent 调用封装）
"""
import json
import re
import uuid
import logging
from typing import Optional, AsyncIterator, Dict, Any

from app.core.config import settings, SUPPORTED_MODELS
from app.core.exceptions import ValidationError, ExternalServiceError

logger = logging.getLogger(__name__)


def _sse(event: Optional[str], data: Dict[str, Any]) -> str:
    """单条 SSE 文本帧（UTF-8 由 StreamingResponse 编码）。"""
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n")
    return "\n".join(lines) + "\n"


def _extract_quality_level(answer_quality: Any) -> Optional[str]:
    """兼容 AnswerQuality 枚举和字符串（LangGraph state 序列化后可能变成 str）。"""
    if not answer_quality:
        return None
    if hasattr(answer_quality, "value"):
        return answer_quality.value
    if isinstance(answer_quality, str):
        return answer_quality
    return str(answer_quality)


def _persist_conversation_messages(
    session_id: str,
    query: str,
    answer_text: str,
    sources: list,
    confidence: Optional[float],
    query_image_oss_key: Optional[str],
    used_fallback: bool = False,
    fallback_reason: Optional[str] = None,
    quality_passed: Optional[bool] = None,
    quality_level: Optional[str] = None,
    kb_name: Optional[str] = None,
) -> None:
    """会话存在时写入 user/assistant 消息（与非流式 invoke 一致）。
    若触发 fallback，额外写入 unanswered_question 独立表（与会话脱钩，长期保留）。"""
    try:
        from app.db import get_conversation_repository
        conv_repo = get_conversation_repository()
        if not conv_repo.get_session(session_id):
            return
        phs = list(set(re.findall(r"<<IMAGE:[0-9a-f]+>>", answer_text or "")))
        conv_repo.add_message(
            session_id=session_id,
            role="user",
            content=query,
            query_image_oss_key=query_image_oss_key,
        )
        conv_repo.add_message(
            session_id=session_id,
            role="assistant",
            content=answer_text or "",
            sources=sources or [],
            confidence=confidence,
            image_placeholders=phs,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
            quality_passed=quality_passed,
            quality_level=quality_level,
        )
        conv_repo.touch_session(session_id)

        # 未回答问题独立存储（与会话脱钩，长期保留）
        if used_fallback:
            from app.db import get_unanswered_repository
            get_unanswered_repository().create(
                session_id=session_id,
                query=query,
                answer=answer_text,
                fallback_reason=fallback_reason,
                quality_level=quality_level,
                confidence=confidence,
                kb_name=kb_name,
                sources=sources,
            )
    except Exception as e:
        logger.warning(f"消息持久化失败（不影响回答）: {e}")


def persist_knowledge_result(
    *,
    session_id: str,
    query: str,
    result: Dict[str, Any],
    query_image_oss_key: Optional[str] = None,
    kb_name: Optional[str] = None,
) -> None:
    _persist_conversation_messages(
        session_id,
        query,
        result.get("answer") or "",
        result.get("sources") or [],
        result.get("confidence"),
        query_image_oss_key,
        used_fallback=result.get("used_fallback", False),
        fallback_reason=result.get("fallback_reason"),
        quality_passed=result.get("quality_passed"),
        quality_level=result.get("quality_level"),
        kb_name=kb_name,
    )


def _load_kb_retrieval(collection: Optional[str]):
    rc: dict = {}
    kb = None
    if not collection:
        return kb, rc
    try:
        from app.db import get_kb_repository
        kb = get_kb_repository().get_by_name(collection)
        if kb:
            rc = kb.get("retrieval_config") or {}
    except Exception as e:
        logger.warning(f"读取 kb retrieval_config 失败，使用默认值: {e}")
    return kb, rc


def _build_rag_config(
    model_name: str,
    kb: Optional[Dict[str, Any]],
    rc: Dict[str, Any],
    collection: Optional[str] = None,
    force_multi_doc: Optional[bool] = None,
    keyword_filter: Optional[str] = None,
    query_image_url: Optional[str] = None,
) -> "RAGConfig":
    """从 kb + retrieval_config 构建统一的 RAGConfig，供 invoke 和 stream 共用。"""
    from agents.knowledge import RAGConfig
    kg_enabled = _resolve_kb_graph_enabled(kb, rc)
    return RAGConfig(
        model=model_name,
        retrieval_strategy="hybrid",
        enable_llm_filter=True,
        enable_rerank=True,
        enable_citations=True,
        enable_fallback=True,
        collection=collection or None,
        ranker=rc.get("ranker", "RRF"),
        rrf_k=rc.get("rrf_k", 60),
        hybrid_alpha=rc.get("hybrid_alpha", 0.5),
        multi_doc_top_k=rc.get("multi_doc_top_k", 20),
        multi_doc_group_size=rc.get("multi_doc_group_size", 3),
        strict_group_size=rc.get("strict_group_size", False),
        single_doc_top_k=rc.get("single_doc_top_k", 20),
        llm_context_top_k=rc.get("llm_context_top_k", 10),
        memory_turns=rc.get("memory_turns", 2),
        kb_type=kb.get("kb_type", "standard") if kb else "standard",
        query_image_url=query_image_url,
        image_vector_dim=rc.get("image_vector_dim", 1024),
        force_multi_doc=force_multi_doc,
        keyword_filter=keyword_filter,
        kg_enabled=kg_enabled,
        kg_graph_id=rc.get("kg_graph_id"),
        kg_top_k=rc.get("kg_top_k", 5),
        kg_timeout_seconds=rc.get("kg_timeout_seconds", 2.0),
        rerank_enabled=rc.get("rerank_enabled", False),
        rerank_model_name=rc.get("rerank_model_name", "qwen3-rerank"),
        single_doc_rerank_top_k=rc.get("single_doc_rerank_top_k", 5),
        multi_doc_rerank_top_k=rc.get("multi_doc_rerank_top_k", 10),
    )


def _resolve_kb_graph_enabled(kb: Optional[Dict[str, Any]], rc: Optional[Dict[str, Any]]) -> bool:
    rc = rc or {}
    if "kg_enabled" in rc and rc.get("kg_enabled") is not None:
        return bool(rc.get("kg_enabled"))

    if rc.get("kg_graph_id"):
        return True

    if not kb or not kb.get("id"):
        return False

    try:
        from app.db import get_file_repository

        return get_file_repository().has_sync_graph_file(kb["id"])
    except Exception as e:
        logger.warning(f"检查知识图谱文件信号失败，按关闭处理: {e}")
        return False


async def invoke_knowledge_qa(
    query: str,
    model_name: str,
    session_id: str,
    collection: Optional[str] = None,
    force_multi_doc: Optional[bool] = None,
    keyword_filter: Optional[str] = None,
    query_image_url: Optional[str] = None,
    query_image_oss_key: Optional[str] = None,
    persist: bool = True,
) -> dict:
    """调用 Knowledge Agent 执行 RAG 问答。"""
    if model_name not in SUPPORTED_MODELS:
        raise ValidationError(f"Model '{model_name}' not supported. Available: {list(SUPPORTED_MODELS.keys())}")

    from agents.knowledge import get_knowledge_agent, create_initial_state, RAGConfig
    agent = get_knowledge_agent()

    request_id = str(uuid.uuid4())

    # 读取 kb 的 retrieval_config 注入到 RAGConfig
    kb, rc = _load_kb_retrieval(collection)

    initial_state = create_initial_state(
        query=query,
        user_id="api_user",
        session_id=session_id,
        config=_build_rag_config(
            model_name=model_name,
            kb=kb,
            rc=rc,
            collection=collection,
            force_multi_doc=force_multi_doc,
            keyword_filter=keyword_filter,
            query_image_url=query_image_url,
        ),
    )

    config = {
        "configurable": {
            "model": model_name,
            "session_id": session_id,
            "thread_id": session_id,  # LangGraph 用 session_id 作为 thread_id 实现对话记忆
        }
    }

    logger.info("处理 knowledge 请求", extra={"session_id": session_id, "request_id": request_id})

    try:
        result = await agent.ainvoke(initial_state, config=config)
    except Exception as e:
        raise ExternalServiceError(f"Knowledge Agent 调用失败: {e}") from e

    metrics = result.get("metrics", {})
    thoughts = {
        "query_analysis": {
            "intent": result.get("query_intent", ""),
            "complexity": result.get("query_complexity", ""),
            "keywords": result.get("query_keywords", []),
        },
        "retrieval": {
            "chunks_retrieved": metrics.total_chunks_retrieved if hasattr(metrics, "total_chunks_retrieved") else 0,
            "chunks_used": metrics.chunks_after_rerank if hasattr(metrics, "chunks_after_rerank") else 0,
        },
        "conversation_turns": 1,
    }

    quality_level = _extract_quality_level(result.get("answer_quality"))
    return_data = {
        "request_id": request_id,
        "session_id": session_id,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources": result["sources"],
        "model": model_name,
        "thoughts": thoughts,
        "image_map": result.get("image_map") or None,
        "finish_reason": "stop",
        "used_fallback": result.get("used_fallback", False),
        "fallback_reason": result.get("fallback_reason"),
        "quality_passed": result.get("quality_passed"),
        "quality_level": quality_level,
        "kb_name": kb.get("name") if kb else None,
    }

    if persist:
        persist_knowledge_result(
            session_id=session_id,
            query=query,
            result=return_data,
            query_image_oss_key=query_image_oss_key,
            kb_name=return_data.get("kb_name"),
        )

    return return_data


def _thoughts_from_state_values(vals: Dict[str, Any]) -> Dict[str, Any]:
    metrics = vals.get("metrics", {})
    return {
        "query_analysis": {
            "intent": vals.get("query_intent", ""),
            "complexity": vals.get("query_complexity", ""),
            "keywords": vals.get("query_keywords", []),
        },
        "retrieval": {
            "chunks_retrieved": metrics.total_chunks_retrieved if hasattr(metrics, "total_chunks_retrieved") else 0,
            "chunks_used": metrics.chunks_after_rerank if hasattr(metrics, "chunks_after_rerank") else 0,
        },
        "conversation_turns": 1,
    }


async def stream_knowledge_qa_sse(
    query: str,
    model_name: str,
    session_id: str,
    collection: Optional[str] = None,
    force_multi_doc: Optional[bool] = None,
    keyword_filter: Optional[str] = None,
    query_image_url: Optional[str] = None,
    query_image_oss_key: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Knowledge 问答 SSE：检索阶段走 LangGraph（interrupt 在 generate 前），
    生成阶段走 OpenAI 兼容流式，再以 precomputed_answer 恢复图执行 check_quality / finalize。
    """
    if model_name not in SUPPORTED_MODELS:
        yield _sse("error", {"message": f"Model '{model_name}' not supported. Available: {list(SUPPORTED_MODELS.keys())}"})
        return

    from agents.knowledge import get_knowledge_stream_prep_agent, create_initial_state, RAGConfig
    from agents.knowledge.nodes.generate import (
        prepare_generation_context,
        _sanitize_image_placeholders,
        build_sources_from_reranked,
    )
    from agents.knowledge.openai_stream import iter_openai_text_deltas

    agent = get_knowledge_stream_prep_agent()
    request_id = str(uuid.uuid4())
    kb, rc = _load_kb_retrieval(collection)

    initial_state = create_initial_state(
        query=query,
        user_id="api_user",
        session_id=session_id,
        config=_build_rag_config(
            model_name=model_name,
            kb=kb,
            rc=rc,
            collection=collection,
            force_multi_doc=force_multi_doc,
            keyword_filter=keyword_filter,
            query_image_url=query_image_url,
        ),
    )

    config = {
        "configurable": {
            "model": model_name,
            "session_id": session_id,
            "thread_id": session_id,
        }
    }

    logger.info("处理 knowledge 流式请求", extra={"session_id": session_id, "request_id": request_id})

    try:
        await agent.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.exception("Knowledge 检索阶段失败")
        yield _sse("error", {"message": f"Knowledge Agent 检索失败: {e}"})
        return

    try:
        snap = await agent.aget_state(config)
        vals = dict(snap.values)
    except Exception as e:
        logger.exception("aget_state 失败")
        yield _sse("error", {"message": str(e)})
        return

    ctx = prepare_generation_context(vals, config)
    sources_preview = build_sources_from_reranked(ctx["reranked_chunks"])

    yield _sse(
        "meta",
        {
            "request_id": request_id,
            "session_id": session_id,
            "model": model_name,
            "thoughts": _thoughts_from_state_values(vals),
            "sources": sources_preview,
            "image_map": ctx["image_map"] or {},
        },
    )

    raw_parts: list[str] = []
    try:
        async for piece in iter_openai_text_deltas(ctx["messages"], ctx["model_name"]):
            raw_parts.append(piece)
            yield _sse("delta", {"text": piece})
    except Exception as e:
        logger.exception("OpenAI 兼容流式调用失败")
        yield _sse("error", {"message": str(e)})
        return

    full = "".join(raw_parts)
    if ctx["is_image_mode"] or ctx["is_multimodal_kb"]:
        sanitized = _sanitize_image_placeholders(full, ctx["image_map"])
    else:
        sanitized = full

    try:
        await agent.aupdate_state(config, {"precomputed_answer": sanitized})
        await agent.ainvoke(None, config=config)
    except Exception as e:
        logger.exception("流式后恢复 LangGraph 失败")
        yield _sse("error", {"message": str(e)})
        return

    try:
        snap2 = await agent.aget_state(config)
        final = dict(snap2.values)
    except Exception as e:
        yield _sse("error", {"message": str(e)})
        return

    answer_final = final.get("answer") or ""
    yield _sse(
        "done",
        {
            "request_id": request_id,
            "session_id": session_id,
            "answer": answer_final,
            "confidence": final.get("confidence"),
            "sources": final.get("sources") or [],
            "model": model_name,
            "thoughts": _thoughts_from_state_values(final),
            "image_map": final.get("image_map") or ctx.get("image_map") or {},
            "finish_reason": "stop",
        },
    )

    _persist_conversation_messages(
        session_id,
        query,
        answer_final,
        final.get("sources") or [],
        final.get("confidence"),
        query_image_oss_key,
        used_fallback=final.get("used_fallback", False),
        fallback_reason=final.get("fallback_reason"),
        quality_passed=final.get("quality_passed"),
        quality_level=_extract_quality_level(final.get("answer_quality")),
        kb_name=kb.get("name") if kb else None,
    )
