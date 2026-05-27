# -*- coding: utf-8 -*-
"""Decide whether graph retrieval needs deep traversal."""

import logging
from datetime import datetime

from app.core.config import settings
from app.core.prompts import KNOWLEDGE_KG_DEEP_ROUTE_SYSTEM
from app.services.llm_service import get_llm_service

from ..state import KnowledgeAgentState

logger = logging.getLogger(__name__)

KG_QUERY_ROUTE_MAX_TOKENS = 8
KG_QUERY_ROUTE_TIMEOUT_SECONDS = 5.0
KG_QUERY_ROUTE_MAX_RETRIES = 0


def kg_query_route(state: KnowledgeAgentState) -> dict:
    start = datetime.now()
    cfg = state.get("config")
    if not cfg or not getattr(cfg, "kg_enabled", False):
        return {
            "kg_deep_traversal": False,
            "processing_log": [{"stage": "kg_query_route", "skipped": True, "reason": "kg_disabled"}],
        }

    query = state.get("rewritten_query") or state.get("query") or ""
    if not query.strip():
        return {
            "kg_deep_traversal": False,
            "processing_log": [{"stage": "kg_query_route", "skipped": True, "reason": "empty_query"}],
        }

    try:
        messages = [
            {"role": "system", "content": KNOWLEDGE_KG_DEEP_ROUTE_SYSTEM},
            {"role": "user", "content": f"用户问题：{query}\n\n只输出 是 或 否："},
        ]
        try:
            text = get_llm_service().chat(
                messages=messages,
                model=settings.llm_clean_model,
                temperature=0.0,
                max_tokens=KG_QUERY_ROUTE_MAX_TOKENS,
                timeout=KG_QUERY_ROUTE_TIMEOUT_SECONDS,
                max_retries=KG_QUERY_ROUTE_MAX_RETRIES,
            )
        except Exception as exc:
            logger.warning("[KgQueryRoute] LLM 调用失败: %s", exc)
            text = ""

        normalized = (text or "").strip().lower()
        deep = normalized.startswith("y") or "yes" in normalized[:8] or "是" in normalized[:4]
        duration_ms = (datetime.now() - start).total_seconds() * 1000
        logger.info("[KgQueryRoute] deep=%s raw=%r (%.0fms)", deep, normalized[:80], duration_ms)
        return {
            "kg_deep_traversal": deep,
            "processing_log": [{
                "stage": "kg_query_route",
                "duration_ms": duration_ms,
                "kg_deep_traversal": deep,
            }],
        }
    except Exception as exc:
        logger.warning("[KgQueryRoute] 失败，默认浅层: %s", exc)
        return {
            "kg_deep_traversal": False,
            "all_warnings": [f"kg_query_route_failed:{exc}"],
        }
