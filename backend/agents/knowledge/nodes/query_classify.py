# -*- coding: utf-8 -*-
"""Query classify node with rule-first and short-timeout LLM fallback."""

import logging
import re
from datetime import datetime

from app.core.config import settings
from app.core.prompts import KNOWLEDGE_QUERY_CLASSIFY_SYSTEM
from app.services.llm_service import get_llm_service

from ..state import KnowledgeAgentState

logger = logging.getLogger(__name__)

QUERY_CLASSIFY_MAX_TOKENS = 8
QUERY_CLASSIFY_TIMEOUT_SECONDS = 5.0
QUERY_CLASSIFY_MAX_RETRIES = 0

STRONG_MULTI_DOC_KEYWORDS = (
    "\u5bf9\u6bd4",
    "\u6bd4\u8f83",
    "\u533a\u522b",
    "\u5dee\u5f02",
    "\u5404\u4e2a",
    "\u5206\u522b",
    "\u591a\u4e2a",
    "compare",
    "comparison",
    "difference",
    "differences",
)
WEAK_MULTI_DOC_KEYWORDS = (
    "\u603b\u7ed3",
    "\u6c47\u603b",
    "\u5f52\u7eb3",
    "\u7efc\u5408",
    "\u54ea\u4e9b",
    "\u6240\u6709",
    "\u6574\u4f53",
    "summarize",
    "summarise",
    "summary",
    "overall",
)
SINGLE_DOC_REFERENCES = (
    "\u8fd9\u4e2a\u6587\u6863",
    "\u8be5\u6587\u4ef6",
    "\u8fd9\u4efd\u6587\u6863",
    "\u8be5\u9644\u4ef6",
    "this document",
    "that document",
    "this file",
    "that file",
    "attached file",
    "attached document",
    "this attachment",
    "that attachment",
)
FILENAME_PATTERN = re.compile(r"\b[\w\-.]+\.(pdf|docx|xlsx|txt|md)\b", re.IGNORECASE)
COLLECTION_SUMMARY_PATTERNS = (
    re.compile(
        r"(\u77e5\u8bc6\u5e93|\u6587\u6863\u5e93).*(\u6709\u4ec0\u4e48|\u6709\u54ea\u4e9b|\u5305\u542b|\u5185\u5bb9|\u6587\u6863|\u8d44\u6599|\u8bb2\u4e86\u4ec0\u4e48|\u8bf4\u660e|\u4ecb\u7ecd)"
    ),
    re.compile(
        r"(what.*(content|documents?|files?).*(knowledge\s*base)|knowledge\s*base.*(what|content|documents?|files?|summary|summarize|summarise))",
        re.IGNORECASE,
    ),
)
LOOKUP_MULTI_DOC_PATTERNS = (
    re.compile(r"(查找|查询|检索|搜索|找一下|帮我找|帮我查).*(制度|政策|流程|规范|文档|资料|办法|指南)"),
    re.compile(r"(相关|有关).*(制度|政策|流程|规范|文档|资料|办法|指南)"),
    re.compile(
        r"(find|search|look\s*up|lookup|show me).*(policy|policies|document|documents|guide|guidelines|procedure|procedures|manual)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(relevant|related).*(policy|policies|document|documents|guide|guidelines|procedure|procedures|manual)",
        re.IGNORECASE,
    ),
)
TOPIC_MULTI_DOC_KEYWORDS = (
    "\u5236\u5ea6",
    "\u653f\u7b56",
    "\u6d41\u7a0b",
    "\u89c4\u8303",
    "\u6307\u5357",
    "\u529e\u6cd5",
    "\u624b\u518c",
    "\u89c4\u5b9a",
    "\u65b9\u6848",
    "\u534f\u8bae",
    "policy",
    "policies",
    "procedure",
    "procedures",
    "process",
    "processes",
    "guide",
    "guidelines",
    "manual",
    "standard",
    "standards",
    "regulation",
    "regulations",
)


def _contains_keyword(query: str, keywords: tuple[str, ...]) -> bool:
    lowered = query.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _extract_filename_mentions(query: str) -> list[str]:
    return [match.group(0).lower() for match in FILENAME_PATTERN.finditer(query)]


def _classify_by_rules(query: str) -> tuple[str, str] | None:
    normalized = (query or "").strip()
    if not normalized:
        return None
    normalized_lower = normalized.lower()

    filenames = list(dict.fromkeys(_extract_filename_mentions(normalized)))
    if len(filenames) >= 2:
        return "multi_doc", "multi_doc_multiple_filenames"

    if _contains_keyword(normalized, STRONG_MULTI_DOC_KEYWORDS):
        return "multi_doc", "multi_doc_keyword"

    if any(reference in normalized_lower for reference in SINGLE_DOC_REFERENCES):
        return "single_doc", "single_doc_reference"

    if len(filenames) == 1:
        return "single_doc", "single_doc_filename"

    if any(pattern.search(normalized) for pattern in COLLECTION_SUMMARY_PATTERNS):
        return "multi_doc", "multi_doc_collection_summary"

    if any(pattern.search(normalized) for pattern in LOOKUP_MULTI_DOC_PATTERNS):
        return "multi_doc", "multi_doc_lookup_query"

    if _contains_keyword(normalized, TOPIC_MULTI_DOC_KEYWORDS):
        return "multi_doc", "multi_doc_topic_query"

    if _contains_keyword(normalized, WEAK_MULTI_DOC_KEYWORDS):
        return "multi_doc", "multi_doc_keyword"

    return None


def _build_processing_log(
    *,
    duration_ms: float,
    query_type: str,
    source: str,
    reason: str | None = None,
    rule_hit: str | None = None,
) -> dict:
    payload = {
        "stage": "query_classify",
        "duration_ms": duration_ms,
        "query_type": query_type,
        "source": source,
    }
    if reason:
        payload["reason"] = reason
    if rule_hit:
        payload["rule_hit"] = rule_hit
    return payload


def query_classify(state: KnowledgeAgentState) -> dict:
    start_time = datetime.now()

    try:
        query = state["rewritten_query"]
        cfg = state.get("config")

        if cfg and cfg.force_multi_doc is True:
            logger.info("[QueryClassify] 用户强制 multi_doc，跳过规则和 LLM 分类")
            return {
                "query_type": "multi_doc",
                "processing_log": [
                    _build_processing_log(
                        duration_ms=0,
                        query_type="multi_doc",
                        source="override",
                        reason="force_multi_doc",
                    )
                ],
            }

        rule_result = _classify_by_rules(query)
        if rule_result is not None:
            query_type, rule_hit = rule_result
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.info("[QueryClassify] 规则命中 (%s): %s -> %s", rule_hit, query, query_type)
            return {
                "query_type": query_type,
                "processing_log": [
                    _build_processing_log(
                        duration_ms=duration,
                        query_type=query_type,
                        source="rule",
                        rule_hit=rule_hit,
                    )
                ],
            }

        logger.info("[QueryClassify] 规则未命中，进入 LLM 分类: %s", query)
        messages = [
            {"role": "system", "content": KNOWLEDGE_QUERY_CLASSIFY_SYSTEM},
            {"role": "user", "content": f"问题: {query}\n\n分类结果:"},
        ]

        try:
            classification = get_llm_service().chat(
                messages=messages,
                model=settings.llm_clean_model,
                temperature=0.0,
                max_tokens=QUERY_CLASSIFY_MAX_TOKENS,
                timeout=QUERY_CLASSIFY_TIMEOUT_SECONDS,
                max_retries=QUERY_CLASSIFY_MAX_RETRIES,
                disable_thinking=True,
            ).strip().lower()
        except Exception as exc:
            logger.warning("[QueryClassify] LLM 调用失败，默认 multi_doc: %s", exc)
            classification = ""

        if "单文档" in classification or "single" in classification:
            query_type = "single_doc"
        elif "多文档" in classification or "multi" in classification:
            query_type = "multi_doc"
        else:
            logger.warning("[QueryClassify] 无法识别分类结果: %r，默认 multi_doc", classification)
            query_type = "multi_doc"

        duration = (datetime.now() - start_time).total_seconds() * 1000
        logger.info("[QueryClassify] 完成 (%.0fms): %s", duration, query_type)

        return {
            "query_type": query_type,
            "processing_log": [
                _build_processing_log(
                    duration_ms=duration,
                    query_type=query_type,
                    source="llm",
                )
            ],
        }
    except Exception as exc:
        logger.error("[QueryClassify] 分类失败: %s", exc, exc_info=True)
        return {
            "query_type": "multi_doc",
            "all_warnings": [f"问题分类失败，默认多文档查询: {exc}"],
        }
