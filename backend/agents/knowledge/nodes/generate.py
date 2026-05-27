# -*- coding: utf-8 -*-
"""
Node 5: Generate
Generates answer based on retrieved context using Volces OpenAI-compatible API
with tool calling support
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib as _hl

# Project fingerprint — cwl-2026
_FP = _hl.md5(b"cwl-KnowledgeAgent").hexdigest()[:8]  # noqa

from langchain_core.messages import AIMessage

from ..state import KnowledgeAgentState
from app.core.config import settings
from app.services.llm_service import get_llm_service
from app.core.prompts import (
    KNOWLEDGE_GENERATE_SYSTEM_GREETING,
    KNOWLEDGE_GENERATE_SYSTEM,
    KNOWLEDGE_GENERATE_SYSTEM_IMAGE,
    KNOWLEDGE_GENERATE_SYSTEM_MULTIMODAL,
    KNOWLEDGE_GENERATE_SYSTEM_VECTOR_GRAPH,
    KNOWLEDGE_GENERATE_SYSTEM_VECTOR_GRAPH_IMAGE,
    KNOWLEDGE_GENERATE_SYSTEM_VECTOR_GRAPH_MULTIMODAL,
)


def _chunk_to_source_block(chunk, index: int, graph_hint: Optional[str] = None) -> str:
    """将单个 chunk 转换为带索引的源文本块"""
    # 提取字段（兼容 dict 和 object 两种格式）
    if isinstance(chunk, dict):
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {}) or {}
        content = metadata.get("parent_content") or content
        file_name = chunk.get("file_name") or metadata.get("file_name", "")
        title = chunk.get("title") or file_name or "未知来源"
        chunk_id = chunk.get("chunk_id") or chunk.get("id", "")
    else:
        content = getattr(chunk, "content", "")
        metadata = getattr(chunk, "metadata", {}) or {}
        if isinstance(metadata, dict):
            content = metadata.get("parent_content") or content
        file_name = getattr(chunk, "file_name", "") or ""
        title = getattr(chunk, "title", None) or getattr(chunk, "source", None) or file_name or "未知来源"
        chunk_id = getattr(chunk, "chunk_id", "") or getattr(chunk, "id", "")

    # 截断超长内容
    max_len = 800
    if len(content) > max_len:
        content = content[:max_len] + "..."

    hint_line = f"\n    <hint>关系类型: {graph_hint}</hint>" if graph_hint else ""

    return (
        f'  <source name="{title}">\n'
        f'    <chunk index="{index}">\n'
        f'      <id>{chunk_id}</id>\n'
        f"      <content>{content}</content>\n"
        f"    </chunk>\n"
        f"  </source>{hint_line}"
    )


def _collect_image_placeholders(chunks: list) -> list:
    """从 chunks 中收集图片占位符"""
    placeholders = []
    pattern = re.compile(r"<<IMAGE:[0-9a-fA-F]+>>")
    for chunk in chunks:
        if isinstance(chunk, dict):
            content = chunk.get("content", "")
        else:
            content = getattr(chunk, "content", "") or ""
        found = pattern.findall(content)
        placeholders.extend(found)
    return placeholders


def _sanitize_image_placeholders(answer: str, image_map: dict) -> str:
    """移除 LLM 捏造的非法占位符，只保留 image_map 中存在的合法占位符"""
    if not image_map:
        return answer
    valid_keys = set(image_map.keys())  # e.g. {"<<IMAGE:9593bf16>>", ...}

    def _replace(m):
        placeholder = m.group(0)
        return placeholder if placeholder in valid_keys else ""

    return re.sub(r'<<IMAGE:[0-9a-fA-F]{8}>>', _replace, answer)


def _convert_messages_to_dicts(messages) -> list:
    """Convert LangChain BaseMessage list to plain dicts for DashScope"""
    result = []
    for msg in messages:
        if hasattr(msg, "type"):
            if msg.type == "human":
                result.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                entry = {"role": "assistant", "content": msg.content or ""}
                # preserve tool_calls if present
                if hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("tool_calls"):
                    entry["tool_calls"] = msg.additional_kwargs["tool_calls"]
                result.append(entry)
            elif msg.type == "tool":
                result.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": getattr(msg, "tool_call_id", "")
                })
            elif msg.type == "system":
                result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, dict):
            result.append(msg)
    return result


def build_sources_from_reranked(reranked_chunks: list) -> List[Dict[str, Any]]:
    """与 generate 节点一致的 sources 列表（供 meta 事件与最终 state 使用）"""
    sources = []
    for chunk in reranked_chunks:
        if isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id") or chunk.get("id", "")
            file_name = chunk.get("file_name") or chunk.get("metadata", {}).get("file_name", "")
            title = chunk.get("title") or chunk.get("metadata", {}).get("title") or file_name or "未知来源"
            sources.append({
                "id": chunk_id,
                "chunk_id": chunk_id,
                "job_id": chunk.get("job_id") or chunk.get("metadata", {}).get("job_id"),
                "title": title,
                "file_name": file_name,
                "chunk_index": chunk.get("chunk_index") if chunk.get("chunk_index") is not None else chunk.get("metadata", {}).get("chunk_index"),
                "content": chunk.get("content", "")[:200],
                "score": chunk.get("score", 0.0),
                "metadata": chunk.get("metadata", {}),
            })
        else:
            file_name = getattr(chunk, "file_name", "") or ""
            title = getattr(chunk, "title", None) or getattr(chunk, "source", None) or file_name or "未知来源"
            chunk_id = getattr(chunk, "chunk_id", "")
            sources.append({
                "id": chunk_id,
                "chunk_id": chunk_id,
                "job_id": getattr(chunk, "job_id", None),
                "title": title,
                "file_name": file_name,
                "chunk_index": getattr(chunk, "chunk_index", None),
                "content": (getattr(chunk, "content", "") or "")[:200],
                "score": getattr(chunk, "score", 0.0),
                "metadata": getattr(chunk, "metadata", {}) or {},
            })
    return sources


def _parent_group_key(chunk) -> str:
    if isinstance(chunk, dict):
        metadata = chunk.get("metadata", {}) or {}
        if isinstance(metadata, dict) and metadata.get("parent_id"):
            return str(metadata["parent_id"])
        return str(chunk.get("chunk_id") or chunk.get("id") or "")

    metadata = getattr(chunk, "metadata", {}) or {}
    if isinstance(metadata, dict) and metadata.get("parent_id"):
        return str(metadata["parent_id"])
    return str(getattr(chunk, "chunk_id", "") or getattr(chunk, "id", ""))


def _dedupe_chunks_by_parent(chunks: list) -> list:
    seen = set()
    deduped = []
    for chunk in chunks:
        key = _parent_group_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def prepare_generation_context(state: KnowledgeAgentState, config=None) -> Dict[str, Any]:
    """
    构建 LLM 请求上下文（不含实际调用）。供流式路径与 generate_answer 共用。
    返回 dict：messages, model_name, image_map, context_text, reranked_chunks,
    is_multimodal_kb, is_image_mode, query, rag_config
    """
    query = state["rewritten_query"]
    reranked_chunks = state.get("merged_chunks") or []
    rag_config = state["config"]
    conversation_messages = state.get("messages", [])

    if config:
        model_name = config.get("configurable", {}).get("model") or rag_config.model
    else:
        model_name = rag_config.model or settings.default_model

    is_multimodal_kb = getattr(rag_config, "kb_type", "standard") == "multimodal"
    query_image_url = getattr(rag_config, "query_image_url", None)
    if is_multimodal_kb:
        model_name = settings.volces_model

    context_parts = []
    image_map: dict = {}

    collection = rag_config.collection
    is_image_mode = False
    if collection:
        try:
            from app.db import get_kb_repository
            kb = get_kb_repository().get_by_name(collection)
            is_image_mode = bool(kb and kb.get("image_mode"))
        except Exception:
            pass

    if is_image_mode:
        try:
            from app.db import get_chunk_image_repository
            chunk_ids = []
            for chunk in reranked_chunks:
                if isinstance(chunk, dict):
                    cid = chunk.get("chunk_id") or chunk.get("id")
                else:
                    cid = getattr(chunk, "chunk_id", None)
                if cid:
                    chunk_ids.append(cid)
            if chunk_ids:
                img_records = get_chunk_image_repository().get_by_chunk_ids(chunk_ids)
                for r in img_records:
                    ph = r.get("placeholder", "")
                    ok = r.get("oss_key", "")
                    if ok and ph:
                        try:
                            from app.services.oss_service import get_oss_service
                            image_map[ph] = get_oss_service().get_presigned_url(ok, expires=3600)
                        except Exception:
                            from urllib.parse import quote
                            image_map[ph] = f"/api/v1/documents/image-proxy?oss_key={quote(ok, safe='/')}"
        except Exception:
            pass

    # ── 两节上下文：向量检索切片 vs 图谱关联切片 ──────────────────────────────

    # 向量切片（merged_chunks）：按 parent_id 去重，同一父块只注入一次上下文
    unique_vector_chunks = _dedupe_chunks_by_parent(reranked_chunks)
    seen_parent_keys = {_parent_group_key(c) for c in unique_vector_chunks}

    # 图谱切片（kg_graph_chunks）：去重补集（不在向量切片里的）
    kg_chunks_raw: list = state.get("kg_graph_chunks") or []
    graph_only_chunks: list = []
    for c in kg_chunks_raw:
        parent_key = _parent_group_key(c)
        if parent_key and parent_key not in seen_parent_keys:
            seen_parent_keys.add(parent_key)
            graph_only_chunks.append(c)

    # ── 渲染向量上下文 ─────────────────────────────────────────────────────
    vector_ctx_parts = []
    for i, chunk in enumerate(unique_vector_chunks, 1):
        vector_ctx_parts.append(_chunk_to_source_block(chunk, i - 1))

    if vector_ctx_parts:
        vector_ctx_parts.insert(0, "<knowledge_base>")
        vector_ctx_parts.append("</knowledge_base>")
    vector_context_text = "\n".join(vector_ctx_parts)

    # ── 渲染图谱上下文 ─────────────────────────────────────────────────────
    graph_ctx_parts = []
    for i, chunk in enumerate(graph_only_chunks, 1):
        # 图谱切片附带关系类型 hint
        rels = chunk.get("relation_types", []) if isinstance(chunk, dict) else getattr(chunk, "relation_types", []) or []
        hint = "; ".join(rels) if rels else ""
        graph_ctx_parts.append(_chunk_to_source_block(chunk, i - 1, graph_hint=hint))

    if graph_ctx_parts:
        graph_ctx_parts.insert(0, "<knowledge_base>")
        graph_ctx_parts.append("</knowledge_base>")
    graph_context_text = "\n".join(graph_ctx_parts)

    # 空图谱时用占位符（避免 prompt 模板里 {graph_context} 处空荡荡导致模型困惑）
    if not graph_context_text.strip():
        graph_context_text = "<knowledge_base>\n  <source name=\"（暂无图谱关联切片）\">\n    <chunk index=\"0\">\n      <content>（当前查询未在知识图谱中检索到相关切片）</content>\n    </chunk>\n  </source>\n</knowledge_base>"

    # ── 收集图片占位符（两个来源都要） ────────────────────────────────────
    for chunk in unique_vector_chunks:
        phs = _collect_image_placeholders([chunk])
        for ph in phs:
            if ph not in image_map:
                image_map[ph] = ""

    if is_image_mode:
        try:
            from app.db import get_chunk_image_repository
            chunk_ids_for_img = (
                [c.get("chunk_id") for c in unique_vector_chunks if c.get("chunk_id")]
                + [c.get("chunk_id") for c in graph_only_chunks if c.get("chunk_id")]
            )
            if chunk_ids_for_img:
                img_records = get_chunk_image_repository().get_by_chunk_ids(list(set(chunk_ids_for_img)))
                for r in img_records:
                    ph = r.get("placeholder", "")
                    ok = r.get("oss_key", "")
                    if ok and ph and ph not in image_map:
                        try:
                            from app.services.oss_service import get_oss_service
                            image_map[ph] = get_oss_service().get_presigned_url(ok, expires=3600)
                        except Exception:
                            from urllib.parse import quote
                            image_map[ph] = f"/api/v1/documents/image-proxy?oss_key={quote(ok, safe='/')}"
        except Exception:
            pass

    # ── 选择 prompt 模板 ───────────────────────────────────────────────────
    query_intent = state.get("query_intent", "general")
    is_greeting = query_intent == "general" and any(
        g in query.lower() for g in ["你好", "您好", "hello", "hi", "嗨"]
    )

    if is_greeting:
        system_prompt = KNOWLEDGE_GENERATE_SYSTEM_GREETING
    elif graph_only_chunks:
        # 有图谱补集：使用两节 prompt
        if is_multimodal_kb:
            system_prompt = KNOWLEDGE_GENERATE_SYSTEM_VECTOR_GRAPH_MULTIMODAL.format(
                vector_context=vector_context_text, graph_context=graph_context_text
            )
        elif is_image_mode:
            system_prompt = KNOWLEDGE_GENERATE_SYSTEM_VECTOR_GRAPH_IMAGE.format(
                vector_context=vector_context_text, graph_context=graph_context_text
            )
        else:
            system_prompt = KNOWLEDGE_GENERATE_SYSTEM_VECTOR_GRAPH.format(
                vector_context=vector_context_text, graph_context=graph_context_text
            )
    else:
        # 无图谱补集：退回到纯向量模板
        if is_multimodal_kb:
            system_prompt = KNOWLEDGE_GENERATE_SYSTEM_MULTIMODAL.format(context=vector_context_text)
        elif is_image_mode:
            system_prompt = KNOWLEDGE_GENERATE_SYSTEM_IMAGE.format(context=vector_context_text)
        else:
            system_prompt = KNOWLEDGE_GENERATE_SYSTEM.format(context=vector_context_text)

    messages = [{"role": "system", "content": system_prompt}]

    memory_turns = rag_config.memory_turns
    history_dicts = _convert_messages_to_dicts(conversation_messages[-(2 * memory_turns + 1):-1])
    messages.extend(history_dicts)

    if is_multimodal_kb and query_image_url:
        # 公网 URL 直接传；本地路径转为 base64 data URI 供 LLM 访问
        image_input = query_image_url
        if not query_image_url.startswith(("http://", "https://")):
            try:
                from app.services.oss_service import get_oss_service
                import base64
                oss_svc = get_oss_service()
                if "/api/v1/files/" in query_image_url:
                    file_key = query_image_url.split("/api/v1/files/", 1)[1]
                else:
                    file_key = query_image_url.lstrip("/")
                img_bytes = oss_svc.get_object_bytes(file_key)
                image_input = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"
            except Exception:
                pass  # 降级为继续使用原 URL（由模型侧决定能否访问）
        user_content = [
            {"image": image_input},
            {"text": f"用户问题：{query}"},
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": f"用户问题：{query}"})

    # 全部上下文切片（向量 + 图，去重）供 sources / confidence 计算
    all_ctx_chunks = unique_vector_chunks + graph_only_chunks

    return {
        "query": query,
        "reranked_chunks": all_ctx_chunks,
        "context_text": vector_context_text,
        "image_map": image_map,
        "messages": messages,
        "model_name": model_name,
        "is_multimodal_kb": is_multimodal_kb,
        "is_image_mode": is_image_mode,
        "rag_config": rag_config,
    }


def _finalize_generate_outputs(
    state: KnowledgeAgentState,
    answer: str,
    ctx: Dict[str, Any],
    model_name: str,
    tools_used: Optional[list] = None,
) -> Dict[str, Any]:
    """由完整 answer 组装 generate 节点返回值（sources / confidence / metrics / AIMessage）"""
    tools_used = tools_used or []
    reranked_chunks = ctx["reranked_chunks"]
    image_map = ctx["image_map"]

    sources = build_sources_from_reranked(reranked_chunks)

    if reranked_chunks:
        def _s(c):
            return (c.get("score", 0) if isinstance(c, dict) else getattr(c, "score", 0)) or 0
        avg_score = sum(_s(c) for c in reranked_chunks[:3]) / min(3, len(reranked_chunks))
        confidence = min(avg_score, 1.0)
    else:
        confidence = 0.5

    metrics = state["metrics"]
    metrics.llm_calls += 1 + (1 if tools_used else 0)

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "context": ctx["context_text"],
        "tools_used": tools_used,
        "metrics": metrics,
        "image_map": image_map,
        "messages": [AIMessage(content=answer)],
        "processing_log": [{
            "stage": "generate",
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "tools_used": tools_used,
            "confidence": confidence,
        }],
        "precomputed_answer": None,
    }


def generate_answer(state: KnowledgeAgentState, config=None) -> Dict[str, Any]:
    """
    Generate answer based on retrieved context using Volces OpenAI-compatible API

    Features:
        - Uses Volces OpenAI-compatible chat.completions API
        - Supports multimodal input (images) via OpenAI content list format
        - Calculates confidence score
        - Extracts source citations

    若 state 含 precomputed_answer（流式写入），则跳过 LLM 调用。
    """
    ctx = prepare_generation_context(state, config)
    query = ctx["query"]
    reranked_chunks = ctx["reranked_chunks"]
    rag_config = ctx["rag_config"]
    image_map = ctx["image_map"]
    model_name = ctx["model_name"]
    messages = ctx["messages"]
    is_multimodal_kb = ctx["is_multimodal_kb"]
    is_image_mode = ctx["is_image_mode"]

    pre = state.get("precomputed_answer")

    print(f"\n[Node 5: Generate] Generating answer with {len(reranked_chunks)} chunks "
          f"and precomputed={pre is not None} [{_FP}]")

    try:
        if pre is not None:
            answer = pre
            return _finalize_generate_outputs(state, answer, ctx, model_name, [])

        tools_used = []

        print(f"[Generate] model={model_name}, multimodal={is_multimodal_kb}, has_image={bool(getattr(rag_config, 'query_image_url', None))}")

        if is_multimodal_kb:
            answer = get_llm_service().chat_with_images(
                messages=messages,
                model=model_name,
                temperature=0.0,
                max_tokens=settings.max_tokens,
            )
        else:
            answer = get_llm_service().chat(
                messages=messages,
                model=model_name,
                temperature=0.0,
                max_tokens=settings.max_tokens,
            )

        if is_image_mode or is_multimodal_kb:
            answer = _sanitize_image_placeholders(answer, image_map)

        out = _finalize_generate_outputs(state, answer, ctx, model_name, tools_used)
        return out

    except Exception as e:
        import traceback
        print(f"[Generate] Error: {e}\n{traceback.format_exc()}")
        return {
            "answer": f"抱歉，生成答案时出现错误: {e}",
            "confidence": 0.0,
            "sources": [],
            "context": "",
            "tools_used": [],
            "all_errors": [f"Generation failed: {e}"],
            "precomputed_answer": None,
        }
