# -*- coding: utf-8 -*-
"""Query rewrite node with deterministic fast-skip logic."""

import logging
import re
from datetime import datetime
from typing import Iterable

from app.core.config import settings
from app.core.prompts import (
    KNOWLEDGE_QUERY_REWRITE_SYSTEM,
    KNOWLEDGE_QUERY_REWRITE_WITH_HISTORY_SYSTEM,
)
from app.services.llm_service import get_llm_service

from ..state import KnowledgeAgentState

logger = logging.getLogger(__name__)

REWRITE_MAX_TOKENS = 64
REWRITE_TIMEOUT_SECONDS = 8.0
REWRITE_MAX_RETRIES = 0
CONTEXT_DEPENDENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"这个|那个|上述|上面|前面|刚才|之前|这里|那里"),
    re.compile(r"[它他她其]"),
    re.compile(r"这(个|里|边|些|种|样|次|段|部分|项)"),
    re.compile(r"那(个|里|边|些|种|样|次|段|部分|项)"),
    re.compile(r"该(怎么|如何|是否|能否|可以|项|功能|模块|接口|流程)"),
)
KB_SCOPE_FAST_SKIP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(这个|该|当前)(知识库|文档库)"),
    re.compile(r"^(this|the|current)\s+knowledge\s*base\b", re.IGNORECASE),
)
INVALID_REWRITE_MARKERS: tuple[str, ...] = (
    "原始问题",
    "rewritten query",
    "current question",
)


def _normalize_recent_history(conversation_messages: list, memory_turns: int) -> list:
    history = conversation_messages[:-1] if len(conversation_messages) > 1 else []
    return history[-(2 * memory_turns):]


def _contains_context_dependent_term(query: str) -> bool:
    return any(pattern.search(query) for pattern in CONTEXT_DEPENDENT_PATTERNS)


def _is_kb_scope_query(query: str) -> bool:
    normalized = (query or "").strip()
    return any(pattern.search(normalized) for pattern in KB_SCOPE_FAST_SKIP_PATTERNS)


def _should_skip_rewrite(query: str, recent_history: Iterable) -> tuple[bool, str]:
    if list(recent_history):
        return False, "has_recent_history"
    if _is_kb_scope_query(query):
        return True, "kb_scope_query"
    if _contains_context_dependent_term(query):
        return False, "context_dependent_query"
    return True, "standalone_query"


def _is_invalid_rewrite(rewritten_query: str) -> bool:
    text = (rewritten_query or "").strip()
    if len(text) < 2:
        return True
    lowered = text.lower()
    if any(marker in lowered for marker in INVALID_REWRITE_MARKERS):
        return True
    return "改写后的问题：" in text or "改写后问题：" in text


def query_rewrite(state: KnowledgeAgentState) -> dict:
    start_time = datetime.now()

    try:
        original_query = state["original_query"]
        conversation_messages = state.get("messages", [])
        memory_turns = state["config"].memory_turns
        recent_history = _normalize_recent_history(conversation_messages, memory_turns)
        should_skip, reason = _should_skip_rewrite(original_query, recent_history)

        if should_skip:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.info("[QueryRewrite] fast-skip (%s): %s", reason, original_query)
            return {
                "rewritten_query": original_query,
                "processing_log": [{
                    "stage": "query_rewrite",
                    "mode": "fast_skip",
                    "reason": reason,
                    "duration_ms": duration,
                    "original": original_query,
                    "rewritten": original_query,
                    "history_turns": len(recent_history) // 2,
                }],
            }

        if recent_history:
            messages = [{"role": "system", "content": KNOWLEDGE_QUERY_REWRITE_WITH_HISTORY_SYSTEM}]
            for msg in recent_history:
                if hasattr(msg, "type"):
                    if msg.type == "human":
                        messages.append({"role": "user", "content": msg.content})
                    elif msg.type == "ai":
                        messages.append({"role": "assistant", "content": msg.content or ""})
                elif isinstance(msg, dict):
                    messages.append(msg)
            messages.append({"role": "user", "content": f"当前问题：{original_query}\n\n改写后的问题："})
        else:
            messages = [
                {"role": "system", "content": KNOWLEDGE_QUERY_REWRITE_SYSTEM},
                {"role": "user", "content": f"原始问题: {original_query}\n\n改写后的问题:"},
            ]

        logger.info("[QueryRewrite] llm-rewrite (%s): %s", reason, original_query)

        try:
            rewritten_query = get_llm_service().chat(
                messages=messages,
                model=settings.llm_clean_model,
                temperature=0.0,
                max_tokens=REWRITE_MAX_TOKENS,
                timeout=REWRITE_TIMEOUT_SECONDS,
                max_retries=REWRITE_MAX_RETRIES,
            ).strip()
        except Exception as exc:
            logger.warning("[QueryRewrite] LLM 调用失败，回退原 query: %s", exc)
            rewritten_query = ""

        if _is_invalid_rewrite(rewritten_query):
            rewritten_query = original_query

        duration = (datetime.now() - start_time).total_seconds() * 1000
        logger.info("[QueryRewrite] 完成 (%.0fms): %s -> %s", duration, original_query, rewritten_query)

        return {
            "rewritten_query": rewritten_query,
            "processing_log": [{
                "stage": "query_rewrite",
                "mode": "llm_rewrite",
                "reason": reason,
                "duration_ms": duration,
                "original": original_query,
                "rewritten": rewritten_query,
                "history_turns": len(recent_history) // 2,
            }],
        }
    except Exception as exc:
        logger.error("[QueryRewrite] 改写失败: %s", exc, exc_info=True)
        return {
            "query": state["original_query"],
            "rewritten_query": state["original_query"],
            "all_warnings": [f"问题改写失败，使用原始问题: {exc}"],
        }
