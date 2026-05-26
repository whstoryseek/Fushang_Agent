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


MAX_CLARIFICATION_ROUNDS = {
    INTENT_OPERATION: 20,
    INTENT_MISSING_KNOWLEDGE: 5,
    INTENT_AMBIGUOUS: 3,
}

EXIT_FIELDS_COMPLETE = "fields_complete"
EXIT_NO_STANDARD_WORKFLOW = "no_standard_workflow"
EXIT_MAX_ROUNDS = "max_rounds"
EXIT_SEMANTIC_UNRESOLVED = "semantic_unresolved"
COMPLETION_COMPLETE = "complete"
COMPLETION_INCOMPLETE = "incomplete"


def get_service_ticket_repository():
    from app.db.service_ticket_repository import get_service_ticket_repository as _get_repo

    return _get_repo()


def should_start_missing_knowledge_flow(rag_result: Dict[str, Any]) -> bool:
    return _rag_result_proves_miss(rag_result)


def _manual_ticket_answer(ticket_id: Optional[str], completion_status: str) -> str:
    prefix = "当前信息未完全收集。" if completion_status == COMPLETION_INCOMPLETE else ""
    return f"{prefix}已为您记录问题，工单号 {ticket_id or '后台工单'}，后续将由专人处理。"


def _round_limit(intent_class: str) -> int:
    return MAX_CLARIFICATION_ROUNDS.get(intent_class, 3)


def _round_limit_exit_reason(intent_class: str) -> str:
    if intent_class == INTENT_AMBIGUOUS:
        return EXIT_SEMANTIC_UNRESOLVED
    return EXIT_MAX_ROUNDS


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _collected_missing_fields(required_fields: List[str], collected: Dict[str, Any]) -> List[str]:
    return [field for field in required_fields if not collected.get(field)]


def _workflow_sources(workflow: Dict[str, Any], existing: Dict[str, Any]) -> List[Any]:
    for key in ("workflow_sources", "sources"):
        sources = workflow.get(key)
        if isinstance(sources, list):
            return sources
    return _as_list(existing.get("workflow_sources"))


def _normal_ticket_user_id(user_id: Optional[str]) -> str:
    return user_id or "guest_default"


def process_ticket_clarification_turn(
    *,
    session_id: str,
    user_id: str,
    user_name: Optional[str],
    sender_id: Optional[str],
    requester_name: Optional[str],
    kb_name: Optional[str],
    query: str,
    channel: str,
    decision: Dict[str, Any],
    workflow: Optional[Dict[str, Any]] = None,
    rag_result: Optional[Dict[str, Any]] = None,
    has_image: bool = False,
    query_image_oss_key: Optional[str] = None,
    continue_only: bool = False,
) -> Optional[Dict[str, Any]]:
    repo = get_service_ticket_repository()
    active = repo.find_active_clarification(
        session_id=session_id,
        user_id=_normal_ticket_user_id(user_id),
        kb_name=kb_name,
    )
    if continue_only and not active:
        return None

    existing = active.get("clarification") if active else {}
    existing = existing if isinstance(existing, dict) else {}
    workflow_payload = workflow or {}
    decision_fields = decision.get("fields") if isinstance(decision, dict) else {}
    decision_fields = decision_fields if isinstance(decision_fields, dict) else {}

    intent_class = str(existing.get("intent_class") or decision.get("intent_class") or INTENT_AMBIGUOUS).upper()
    if intent_class not in _VALID_INTENTS:
        intent_class = INTENT_AMBIGUOUS
    reason = str(existing.get("reason") or decision.get("reason") or "")
    original_query = str(existing.get("original_query") or query or "")

    turns = list(existing.get("turns") or [])
    current_round = int(active.get("clarification_round") or 0) if active else 0
    round_no = max(current_round, len(turns)) + 1

    collected = dict(existing.get("collected") or {})
    collected.update(decision_fields)
    if has_image:
        collected["has_image"] = True
        if query_image_oss_key:
            image_keys = list(collected.get("image_keys") or [])
            image_keys.append(query_image_oss_key)
            collected["image_keys"] = image_keys

    workflow_found = bool(
        workflow_payload.get("workflow_found")
        if "workflow_found" in workflow_payload
        else existing.get("workflow_found")
    )
    required_fields = (
        _as_list(workflow_payload.get("required_fields"))
        or _as_list(existing.get("required_fields"))
        or _as_list(decision.get("missing_fields"))
        or ["issue_detail"]
    )
    missing_fields = _collected_missing_fields(required_fields, collected)

    no_standard_workflow = intent_class == INTENT_OPERATION and not workflow_found
    operation_fields_complete = intent_class == INTENT_OPERATION and workflow_found and not missing_fields
    round_limit_reached = round_no >= _round_limit(intent_class)
    should_finalize = no_standard_workflow or operation_fields_complete or round_limit_reached

    if no_standard_workflow:
        exit_reason = EXIT_NO_STANDARD_WORKFLOW
        completion_status = COMPLETION_INCOMPLETE
    elif operation_fields_complete:
        exit_reason = EXIT_FIELDS_COMPLETE
        completion_status = COMPLETION_COMPLETE
    elif round_limit_reached:
        exit_reason = _round_limit_exit_reason(intent_class)
        completion_status = COMPLETION_INCOMPLETE
    else:
        exit_reason = ""
        completion_status = existing.get("completion_status") or COMPLETION_INCOMPLETE

    status = "pending_manual" if should_finalize else "clarifying"
    answer_ticket_id = active.get("id") if active else None
    answer = (
        _manual_ticket_answer(answer_ticket_id, completion_status)
        if should_finalize
        else decision.get("question") or "请补充具体业务场景、操作步骤或报错信息。"
    )

    turns.append(
        {
            "round": round_no,
            "query": query or "",
            "answer": answer,
            "status": status,
            "image_key": query_image_oss_key,
        }
    )

    clarification = {
        "intent_class": intent_class,
        "reason": reason,
        "original_query": original_query,
        "required_fields": required_fields,
        "missing_fields": missing_fields,
        "collected": collected,
        "workflow_found": workflow_found,
        "workflow_summary": workflow_payload.get("workflow_summary") or existing.get("workflow_summary") or "",
        "workflow_sources": _workflow_sources(workflow_payload, existing),
        "kb_result": existing.get("kb_result") or (rag_result if isinstance(rag_result, dict) else {}),
        "image_analysis": existing.get("image_analysis")
        or {"has_image": bool(has_image), "query_image_oss_key": query_image_oss_key},
        "turns": turns,
        "completion_status": completion_status,
        "exit_reason": exit_reason,
        "ready_for_manual": should_finalize,
    }

    common_update = {
        "status": status,
        "answer": answer,
        "clarification_round": round_no,
        "clarification": clarification,
        "sender_id": sender_id,
        "requester_name": requester_name or user_name,
    }
    if active:
        ticket = repo.update_clarification(active["id"], **common_update)
    else:
        ticket = repo.create_with_contexts(
            session_id=session_id,
            user_id=_normal_ticket_user_id(user_id),
            user_name=user_name,
            kb_name=kb_name,
            query=original_query,
            answer=answer,
            status=status,
            confidence=0.0,
            fallback_reason=reason,
            quality_level="clarifying",
            sources=[],
            channel=channel or "web",
            sender_id=sender_id,
            requester_name=requester_name or user_name,
            clarification_round=round_no,
            clarification=clarification,
            contexts=[],
        )

    if ticket and should_finalize and not answer_ticket_id:
        answer = _manual_ticket_answer(ticket.get("id"), completion_status)
        turns[-1]["answer"] = answer
        clarification["turns"] = turns
        ticket = repo.update_clarification(
            ticket["id"],
            status=status,
            answer=answer,
            clarification_round=round_no,
            clarification=clarification,
            sender_id=sender_id,
            requester_name=requester_name or user_name,
        )

    return {
        "ticket_id": ticket.get("id") if ticket else None,
        "status": status,
        "answer": answer,
        "clarification_round": round_no,
        "clarification": clarification,
        "confidence": 0.0,
        "finish_reason": "manual_ticket_created" if should_finalize else "clarification",
    }
