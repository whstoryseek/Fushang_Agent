# -*- coding: utf-8 -*-
import json
import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

INTENT_OPERATION = "A"
INTENT_MISSING_KNOWLEDGE = "B"
INTENT_AMBIGUOUS = "C"
INTENT_NORMAL = "D"

CLASSIFIER_MODEL = getattr(settings, "operation_classifier_model", "doubao-seed-2-0-mini-260428")
CLASSIFIER_MAX_TOKENS = 300
CLASSIFIER_TIMEOUT = min(float(getattr(settings, "operation_classifier_timeout", 1.2)), 1.2)

_VALID_INTENTS = {
    INTENT_OPERATION,
    INTENT_MISSING_KNOWLEDGE,
    INTENT_AMBIGUOUS,
    INTENT_NORMAL,
}


def _fallback_decision(query: str, has_image: bool) -> Dict[str, Any]:
    if has_image or not (query or "").strip():
        return {
            "intent_class": INTENT_AMBIGUOUS,
            "reason": "ambiguous_query",
            "needs_ticket_flow": True,
            "confidence": 0.0,
            "fields": {},
            "missing_fields": ["issue_detail"],
            "question": "请补充说明图片对应的业务场景、操作步骤或报错信息。",
            "rationale_brief": "image_only_or_empty_query_classifier_failed",
            "source": "fallback",
        }
    return {
        "intent_class": INTENT_NORMAL,
        "reason": "knowledge_question",
        "needs_ticket_flow": False,
        "confidence": 0.0,
        "fields": {},
        "missing_fields": [],
        "question": "",
        "rationale_brief": "classifier_failed",
        "source": "fallback",
    }


def _extract_json_payload(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("classifier response must be a JSON object")
    return payload


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rag_result_proves_miss(rag_result: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(rag_result, dict):
        return False
    if not rag_result.get("used_fallback"):
        return False
    fallback_reason = str(rag_result.get("fallback_reason") or "").lower()
    if "no_relevant" in fallback_reason or "no relevant" in fallback_reason:
        return True
    if "knowledge base has no relevant content" in fallback_reason:
        return True
    if "知识库" in fallback_reason:
        return any(
            phrase in fallback_reason
            for phrase in ("无相关", "没有相关", "未找到", "没有找到")
        )
    return False


def _normalize_payload(
    payload: Dict[str, Any],
    query: str,
    has_image: bool,
    rag_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    intent = str(payload.get("intent_class") or "").strip().upper()
    if intent not in _VALID_INTENTS:
        return _fallback_decision(query, has_image)

    needs_ticket_flow = bool(payload.get("needs_ticket_flow"))
    if intent == INTENT_NORMAL:
        needs_ticket_flow = False
    elif intent == INTENT_MISSING_KNOWLEDGE:
        needs_ticket_flow = _rag_result_proves_miss(rag_result)
    elif intent in {INTENT_OPERATION, INTENT_AMBIGUOUS}:
        needs_ticket_flow = True

    fields = payload.get("fields")
    missing_fields = payload.get("missing_fields")

    return {
        "intent_class": intent,
        "reason": str(payload.get("reason") or ""),
        "needs_ticket_flow": needs_ticket_flow,
        "confidence": _as_float(payload.get("confidence")),
        "fields": fields if isinstance(fields, dict) else {},
        "missing_fields": missing_fields if isinstance(missing_fields, list) else [],
        "question": str(payload.get("question") or ""),
        "rationale_brief": str(payload.get("rationale_brief") or ""),
        "source": "llm",
    }


def _classifier_prompt(
    query: str,
    history: Optional[List[Dict[str, Any]]],
    has_image: bool,
    rag_result: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    history_lines = []
    for turn in (history or [])[-4:]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "user")
        content = str(turn.get("content") or "")[:300]
        history_lines.append({"role": role, "content": content})

    rag_summary = None
    if isinstance(rag_result, dict):
        sources = rag_result.get("sources") or []
        rag_summary = {
            "used_fallback": rag_result.get("used_fallback"),
            "fallback_reason": rag_result.get("fallback_reason"),
            "quality_passed": rag_result.get("quality_passed"),
            "quality_level": rag_result.get("quality_level"),
            "source_count": len(sources) if isinstance(sources, list) else 0,
        }

    return [
        {
            "role": "system",
            "content": (
                "你是售后/服务工单意图分类器。只能输出一个 JSON 对象，不要输出 Markdown。"
                "分类：A=用户明确要求系统或人工代办具体动作、开通、修改、处理、排查或提交；"
                "B=结合 RAG 结果确认知识库缺失，需要收集问题形成工单；"
                "C=语义不明、上下文不足、仅图片或无法判断用户到底要什么；"
                "D=普通知识库问答、教程咨询、政策咨询、能力咨询或闲聊。"
                "A/D 边界必须严格：教程、咨询、怎么、如何、能否、有什么用、在哪里看、"
                "为什么等问题不是 A，除非用户明确要求系统或人工代办某个具体动作。"
                "输出字段必须包含：intent_class, reason, needs_ticket_flow, confidence, "
                "fields, missing_fields, question, rationale_brief。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": query or "",
                    "history": history_lines,
                    "has_image": bool(has_image),
                    "rag_result": rag_summary,
                },
                ensure_ascii=False,
            ),
        },
    ]


def classify_ticket_intent_with_llm(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    has_image: bool = False,
    rag_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        text = get_llm_service().responses_text(
            _classifier_prompt(query, history, has_image, rag_result),
            model=CLASSIFIER_MODEL,
            temperature=0.0,
            max_tokens=CLASSIFIER_MAX_TOKENS,
            timeout=CLASSIFIER_TIMEOUT,
            max_retries=0,
        )
        payload = _extract_json_payload(text)
        return _normalize_payload(payload, query, has_image, rag_result)
    except Exception as exc:
        logger.warning("ticket intent classifier failed: %s", exc)
        return _fallback_decision(query, has_image)
