# -*- coding: utf-8 -*-
"""Knowledge API Routes"""
import asyncio
import contextlib
import json
import logging
import re
import uuid
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.api.v1.auth_deps import get_request_identity, get_request_user_id
from app.core.config import settings
from app.models.requests import KnowledgeRequest
from app.models.responses import KnowledgeResponse
from app.services import conversation_service
from app.services.knowledge_service import (
    invoke_knowledge_qa,
    persist_knowledge_result,
    stream_knowledge_qa_sse,
)
from app.services.oss_service import get_oss_service
from app.services.service_ticket_service import (
    INTENT_AMBIGUOUS,
    INTENT_MISSING_KNOWLEDGE,
    INTENT_NORMAL,
    INTENT_OPERATION,
    analyze_clarification_reply_with_llm,
    analyze_operation_workflow_with_llm,
    analyze_ticket_image_fields_with_llm,
    classify_ticket_intent_with_llm,
    find_active_clarification_ticket,
    load_ticket_history_context,
    merge_ticket_image_analysis_into_decision,
    process_ticket_clarification_turn,
    record_ai_resolved_ticket,
    record_unanswered_normal_ticket,
    resolve_active_clarification_as_unanswered_normal,
    resolve_active_clarification_with_rag,
    should_start_missing_knowledge_flow,
)

router = APIRouter(prefix="/knowledge")
logger = logging.getLogger(__name__)

TICKET_CLASSIFIER_INTERCEPT_TIMEOUT = 0.6
STREAM_TICKET_CLASSIFIER_INTERCEPT_TIMEOUT = 1.2

CAPABILITY_QUERY_PATTERNS = {
    "你能做什么",
    "你可以做什么",
    "你能帮我做什么",
    "你会什么",
    "你有什么功能",
    "这个系统能做什么",
    "知识库能做什么",
}


def _coerce_request_identity(user_id: str | None, request_identity: Any) -> dict:
    identity = dict(request_identity) if isinstance(request_identity, dict) else {}
    resolved_user_id = str(identity.get("user_id") or user_id or "guest_default")
    resolved_user_name = identity.get("user_name")
    resolved_channel = str(identity.get("channel") or "web")
    resolved_requester_name = identity.get("requester_name") or resolved_user_name
    return {
        "user_id": resolved_user_id,
        "user_name": resolved_user_name,
        "channel": resolved_channel,
        "sender_id": identity.get("sender_id"),
        "requester_name": resolved_requester_name,
        "entry_user_id": identity.get("entry_user_id"),
        "entry_user_name": identity.get("entry_user_name"),
        "entry_source": identity.get("entry_source"),
    }


def _request_image_payloads(request: KnowledgeRequest) -> list[str]:
    payloads: list[str] = []
    for raw in [request.query_image, *(request.query_images or [])]:
        value = str(raw or "").strip()
        if value and value not in payloads:
            payloads.append(value)
    return payloads


def _normalize_query_for_capability_match(query: str) -> str:
    return re.sub(r"[\s\?？!！,，。.:：、】【（）()]+", "", str(query or "").strip().lower())


def _is_capability_question(query: str, *, has_image: bool = False) -> bool:
    if has_image:
        return False
    normalized = _normalize_query_for_capability_match(query)
    return normalized in CAPABILITY_QUERY_PATTERNS


def _capability_answer_result(*, session_id: str, model_name: str, query: str, kb_name: str | None) -> dict:
    answer = (
        "我主要可以帮您做两类事情："
        "1. 做知识库问答，回答业务流程、资料要求、功能说明和常见报错排查；"
        "2. 如果您明确要代办处理，我会继续追问必要资料并进入工单流程。"
        "您可以直接问具体业务问题，例如“微信子商户号开通需要什么资料？”"
    )
    return {
        "request_id": str(uuid.uuid4()),
        "session_id": session_id,
        "answer": answer,
        "confidence": 0.99,
        "sources": [],
        "model": model_name,
        "thoughts": {
            "query_analysis": {"intent": "capability_intro", "complexity": "simple", "keywords": []},
            "retrieval": {"chunks_retrieved": 0, "chunks_used": 0},
            "conversation_turns": 1,
            "shortcut": "capability_answer",
        },
        "image_map": None,
        "finish_reason": "stop",
        "used_fallback": False,
        "fallback_reason": None,
        "quality_passed": True,
        "quality_level": "high",
        "kb_name": kb_name,
    }


def _upload_query_images(
    *,
    request: KnowledgeRequest,
    user_id: str,
    session_id: str,
) -> tuple[list[str], list[str]]:
    image_payloads = _request_image_payloads(request)
    if not image_payloads:
        return [], []

    query_image_urls: list[str] = []
    query_image_oss_keys: list[str] = []
    try:
        import base64
        from app.services.oss_service import get_oss_service

        oss_service = get_oss_service()
        kb_name = request.collection or "default"
        oss_path = f"query_images/{user_id}/{kb_name}/{session_id}"
        for image_payload in image_payloads:
            img_bytes = base64.b64decode(image_payload)
            img_uuid = uuid.uuid4().hex[:12]
            oss_key = oss_service.upload_file(oss_path, f"{img_uuid}.jpg", img_bytes)
            query_image_oss_keys.append(oss_key)
            query_image_urls.append(oss_service.get_presigned_url(oss_key, expires=600))
    except Exception as e:
        logger.warning("用户查询图片上传失败，降级为纯文字检索: %s", e)
        return [], []

    return query_image_urls, query_image_oss_keys


def _consume_background_task_exception(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    try:
        task.result()
    except Exception as exc:
        logger.debug("background ticket classifier finished with error: %s", exc)


async def _cancel_rag_task(task: asyncio.Task) -> None:
    if task.done():
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.result()
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


def _is_pre_rag_ticket_decision(decision: dict | None) -> bool:
    if not isinstance(decision, dict) or not decision.get("needs_ticket_flow"):
        return False
    return decision.get("intent_class") in {INTENT_OPERATION, INTENT_AMBIGUOUS}


def _is_missing_knowledge_ticket_decision(decision: dict | None) -> bool:
    if not isinstance(decision, dict) or not decision.get("needs_ticket_flow"):
        return False
    return decision.get("intent_class") == INTENT_MISSING_KNOWLEDGE


def _should_record_ai_resolved_ticket(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    if not (result.get("answer") or "").strip():
        return False
    if result.get("used_fallback"):
        return False
    if result.get("quality_passed") is False:
        return False
    return True


def _should_record_unanswered_normal_ticket(result: dict | None, decision: dict | None) -> bool:
    if not isinstance(result, dict) or not isinstance(decision, dict):
        return False
    if not should_start_missing_knowledge_flow(result):
        return False
    if not (result.get("answer") or "").strip():
        return False
    return (
        str(decision.get("intent_class") or "").upper() == INTENT_NORMAL
        and not bool(decision.get("needs_ticket_flow"))
    )


def _load_classifier_history(
    *,
    session_id: str,
    active_ticket: dict | None = None,
) -> list[dict]:
    return load_ticket_history_context(session_id, active_ticket=active_ticket, limit=8)


def _record_ai_resolved_ticket_safely(
    *,
    session_id: str,
    user_id: str,
    user_name: str | None = None,
    kb_name: str | None,
    query: str,
    result: dict,
    channel: str = "web",
    sender_id: str | None = None,
    requester_name: str | None = None,
    entry_user_id: str | None = None,
    entry_user_name: str | None = None,
    entry_source: str | None = None,
    has_image: bool = False,
    query_image_oss_key: str | None = None,
) -> None:
    if not _should_record_ai_resolved_ticket(result):
        return
    try:
        record_ai_resolved_ticket(
            session_id=session_id,
            user_id=user_id,
            user_name=user_name,
            kb_name=kb_name,
            query=query,
            answer=result.get("answer") or "",
            status="resolved_ai",
            confidence=result.get("confidence"),
            rag_result=result,
            channel=channel,
            sender_id=sender_id,
            requester_name=requester_name,
            entry_user_id=entry_user_id,
            entry_user_name=entry_user_name,
            entry_source=entry_source,
            has_image=has_image,
            query_image_oss_key=query_image_oss_key,
        )
    except Exception as exc:
        logger.warning("record resolved-ai service ticket failed: %s", exc)


def _record_unanswered_normal_ticket_safely(
    *,
    session_id: str,
    user_id: str,
    user_name: str | None = None,
    kb_name: str | None,
    query: str,
    result: dict,
    decision: dict | None,
    channel: str = "web",
    sender_id: str | None = None,
    requester_name: str | None = None,
    entry_user_id: str | None = None,
    entry_user_name: str | None = None,
    entry_source: str | None = None,
    has_image: bool = False,
    query_image_oss_key: str | None = None,
    query_image_oss_keys: list[str] | None = None,
) -> None:
    if not _should_record_unanswered_normal_ticket(result, decision):
        return
    try:
        record_unanswered_normal_ticket(
            session_id=session_id,
            user_id=user_id,
            user_name=user_name,
            kb_name=kb_name,
            query=query,
            answer=result.get("answer") or "",
            confidence=result.get("confidence"),
            rag_result=result,
            channel=channel,
            sender_id=sender_id,
            requester_name=requester_name,
            entry_user_id=entry_user_id,
            entry_user_name=entry_user_name,
            entry_source=entry_source,
            has_image=has_image,
            query_image_oss_key=query_image_oss_key,
            query_image_oss_keys=query_image_oss_keys,
        )
    except Exception as exc:
        logger.warning("record unanswered-normal service ticket failed: %s", exc)


def _ticket_response(
    *,
    ticket_result: dict,
    decision: dict,
    session_id: str,
    model_name: str,
    request_id: str | None = None,
) -> KnowledgeResponse:
    return KnowledgeResponse(
        status_code=200,
        request_id=request_id or str(uuid.uuid4()),
        session_id=session_id,
        answer=ticket_result["answer"],
        confidence=float(ticket_result.get("confidence") or 0.0),
        sources=[],
        model=model_name,
        finish_reason=ticket_result.get("finish_reason") or "clarification",
        thoughts={
            "clarification_required": True,
            "clarification": ticket_result.get("clarification"),
            "ticket_intent": decision,
        },
        image_map=None,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _single_done_stream(payload: dict):
    yield _sse("done", payload)


def _ticket_done_payload(
    *,
    ticket_result: dict,
    decision: dict,
    session_id: str,
    model_name: str,
    request_id: str | None = None,
    image_map: dict | None = None,
) -> dict:
    return {
        "request_id": request_id or str(uuid.uuid4()),
        "session_id": session_id,
        "answer": ticket_result["answer"],
        "confidence": float(ticket_result.get("confidence") or 0.0),
        "sources": [],
        "model": model_name,
        "thoughts": {
            "clarification_required": True,
            "clarification": ticket_result.get("clarification"),
            "ticket_intent": decision,
        },
        "image_map": image_map or {},
        "finish_reason": ticket_result.get("finish_reason") or "clarification",
    }


def _normal_response(result: dict) -> KnowledgeResponse:
    return KnowledgeResponse(
        status_code=200,
        request_id=result["request_id"],
        session_id=result["session_id"],
        answer=result["answer"],
        confidence=result["confidence"],
        sources=result["sources"],
        model=result["model"],
        finish_reason=result.get("finish_reason") or "stop",
        thoughts=result["thoughts"],
        image_map=result["image_map"],
    )


def lookup_operation_workflow(
    *,
    query: str,
    kb_name: str | None,
    decision: dict,
    rag_result: dict | None,
) -> dict:
    return analyze_operation_workflow_with_llm(
        query=query,
        decision=decision,
        rag_result=rag_result,
    )


def _find_active_ticket_safely(session_id: str, user_id: str, kb_name: str | None) -> dict | None:
    try:
        return find_active_clarification_ticket(
            session_id=session_id,
            user_id=user_id,
            kb_name=kb_name,
        )
    except Exception as exc:
        logger.debug("find active clarification ticket failed: %s", exc)
        return None


def _active_ticket_image_keys(active_ticket: dict | None) -> list[str]:
    if not isinstance(active_ticket, dict):
        return []
    clarification = active_ticket.get("clarification")
    if not isinstance(clarification, dict):
        return []

    keys: list[str] = []
    for source in (
        clarification.get("collected"),
        clarification.get("image_analysis"),
    ):
        if not isinstance(source, dict):
            continue
        for raw_key in source.get("image_keys") or []:
            key = str(raw_key or "").strip()
            if key and key not in keys:
                keys.append(key)
    return keys


def _resolve_image_urls_from_oss_keys(oss_keys: list[str]) -> list[str]:
    if not oss_keys:
        return []
    urls: list[str] = []
    try:
        oss_service = get_oss_service()
        for raw_key in oss_keys:
            key = str(raw_key or "").strip()
            if not key:
                continue
            urls.append(oss_service.get_presigned_url(key, expires=600))
    except Exception as exc:
        logger.debug("resolve ticket image oss keys failed: %s", exc)
        return []
    return urls


def _create_ticket_result(
    *,
    session_id: str,
    user_id: str,
    user_name: str | None = None,
    kb_name: str | None,
    query: str,
    decision: dict,
    channel: str = "web",
    sender_id: str | None = None,
    requester_name: str | None = None,
    entry_user_id: str | None = None,
    entry_user_name: str | None = None,
    entry_source: str | None = None,
    has_image: bool,
    query_image_oss_key: str | None,
    query_image_oss_keys: list[str] | None = None,
    rag_result: dict | None = None,
    workflow: dict | None = None,
) -> dict | None:
    if workflow is None and decision.get("intent_class") == INTENT_OPERATION:
        workflow = {"workflow_found": False, "required_fields": []}
    if workflow and workflow.get("question"):
        decision = {**decision, "question": workflow.get("question")}
    return process_ticket_clarification_turn(
        session_id=session_id,
        user_id=user_id,
        user_name=user_name,
        sender_id=sender_id,
        requester_name=requester_name,
        kb_name=kb_name,
        query=query,
        channel=channel,
        decision=decision,
        workflow=workflow,
        rag_result=rag_result,
        has_image=has_image,
        query_image_oss_key=query_image_oss_key,
        query_image_oss_keys=query_image_oss_keys,
        entry_user_id=entry_user_id,
        entry_user_name=entry_user_name,
        entry_source=entry_source,
    )


async def _handle_active_clarification_reply(
    *,
    active_ticket: dict,
    query: str,
    session_id: str,
    user_id: str,
    kb_name: str | None,
    model_name: str,
    has_image: bool,
    query_image_url: str | None,
    query_image_urls: list[str] | None = None,
    query_image_oss_key: str | None,
    query_image_oss_keys: list[str] | None = None,
    force_multi_doc: bool | None = None,
    keyword_filter: str | None = None,
    channel: str = "web",
    sender_id: str | None = None,
    requester_name: str | None = None,
    user_name: str | None = None,
    entry_user_id: str | None = None,
    entry_user_name: str | None = None,
    entry_source: str | None = None,
) -> tuple[str, dict, dict]:
    decision = await asyncio.to_thread(
        analyze_clarification_reply_with_llm,
        query=query,
        active_ticket=active_ticket,
        has_image=has_image,
    )
    analysis_image_urls = [url for url in (query_image_urls or []) if str(url or "").strip()]
    if query_image_url and query_image_url not in analysis_image_urls:
        analysis_image_urls.insert(0, query_image_url)
    analysis_image_oss_keys = [key for key in (query_image_oss_keys or []) if str(key or "").strip()]
    if query_image_oss_key and query_image_oss_key not in analysis_image_oss_keys:
        analysis_image_oss_keys.insert(0, query_image_oss_key)

    if not analysis_image_urls and decision.get("needs_ticket_flow"):
        existing_image_oss_keys = _active_ticket_image_keys(active_ticket)
        if existing_image_oss_keys:
            analysis_image_oss_keys = existing_image_oss_keys
            analysis_image_urls = _resolve_image_urls_from_oss_keys(existing_image_oss_keys)

    if analysis_image_urls:
        image_analysis = await asyncio.to_thread(
            analyze_ticket_image_fields_with_llm,
            query=query,
            active_ticket=active_ticket,
            image_url=analysis_image_urls[0],
            image_urls=analysis_image_urls,
            query_image_oss_key=analysis_image_oss_keys[0] if analysis_image_oss_keys else None,
            query_image_oss_keys=analysis_image_oss_keys,
        )
        decision = merge_ticket_image_analysis_into_decision(decision, image_analysis, active_ticket)
    if decision.get("intent_class") == INTENT_NORMAL and not decision.get("needs_ticket_flow"):
        resolved_query = decision.get("resolved_query") or query
        result = await invoke_knowledge_qa(
            query=resolved_query,
            model_name=model_name,
            session_id=session_id,
            collection=kb_name,
            force_multi_doc=force_multi_doc,
            keyword_filter=keyword_filter,
            query_image_url=query_image_url,
            query_image_oss_key=query_image_oss_key,
            persist=False,
        )
        if should_start_missing_knowledge_flow(result):
            missing_decision = await asyncio.to_thread(
                classify_ticket_intent_with_llm,
                query=resolved_query,
                history=_load_classifier_history(session_id=session_id, active_ticket=active_ticket),
                has_image=has_image,
                rag_result=result,
            )
            if _is_missing_knowledge_ticket_decision(missing_decision):
                ticket_result = process_ticket_clarification_turn(
                    session_id=session_id,
                    user_id=user_id,
                    user_name=user_name,
                    sender_id=sender_id,
                    requester_name=requester_name,
                    kb_name=kb_name,
                    query=query,
                    channel=channel or "web",
                    decision=missing_decision,
                    rag_result=result,
                    has_image=has_image,
                    query_image_oss_key=query_image_oss_key,
                    query_image_oss_keys=query_image_oss_keys,
                    continue_only=True,
                    entry_user_id=entry_user_id,
                    entry_user_name=entry_user_name,
                    entry_source=entry_source,
                )
                if ticket_result:
                    return "ticket", ticket_result, missing_decision
            if _should_record_unanswered_normal_ticket(result, missing_decision):
                resolve_active_clarification_as_unanswered_normal(
                    active_ticket=active_ticket,
                    query=query,
                    answer=result.get("answer") or "",
                    rag_result=result,
                    has_image=has_image,
                    query_image_oss_key=query_image_oss_key,
                    query_image_oss_keys=query_image_oss_keys,
                    sender_id=sender_id,
                    requester_name=requester_name,
                    user_name=user_name,
                    entry_user_id=entry_user_id,
                    entry_user_name=entry_user_name,
                    entry_source=entry_source,
                )
        elif result.get("quality_passed") is not False:
            resolve_active_clarification_with_rag(
                active_ticket=active_ticket,
                query=query,
                answer=result.get("answer") or "",
                rag_result=result,
                has_image=has_image,
                query_image_oss_key=query_image_oss_key,
                query_image_oss_keys=query_image_oss_keys,
                sender_id=sender_id,
                requester_name=requester_name,
                user_name=user_name,
                entry_user_id=entry_user_id,
                entry_user_name=entry_user_name,
                entry_source=entry_source,
            )
        return "rag", result, decision

    workflow = None
    rag_result_for_workflow = None
    existing = active_ticket.get("clarification") if isinstance(active_ticket.get("clarification"), dict) else {}
    if decision.get("intent_class") == INTENT_OPERATION and not existing.get("workflow_found") and not existing.get("required_fields"):
        rag_result_for_workflow = await invoke_knowledge_qa(
            query=decision.get("resolved_query") or query,
            model_name=model_name,
            session_id=session_id,
            collection=kb_name,
            force_multi_doc=force_multi_doc,
            keyword_filter=keyword_filter,
            query_image_url=query_image_url,
            query_image_oss_key=query_image_oss_key,
            persist=False,
        )
        workflow = lookup_operation_workflow(
            query=query,
            kb_name=kb_name,
            decision=decision,
            rag_result=rag_result_for_workflow,
        )
    ticket_result = process_ticket_clarification_turn(
        session_id=session_id,
        user_id=user_id,
        user_name=user_name,
        sender_id=sender_id,
        requester_name=requester_name,
        kb_name=kb_name,
        query=query,
        channel=channel or "web",
        decision=decision,
        workflow=workflow,
        rag_result=rag_result_for_workflow,
        has_image=has_image,
        query_image_oss_key=query_image_oss_key,
        query_image_oss_keys=query_image_oss_keys,
        continue_only=True,
        entry_user_id=entry_user_id,
        entry_user_name=entry_user_name,
        entry_source=entry_source,
    )
    if ticket_result:
        return "ticket", ticket_result, decision
    return "none", {}, decision


@router.post("/", response_model=KnowledgeResponse, summary="Knowledge Base Q&A")
async def knowledge_qa(
    request: KnowledgeRequest,
    user_id: str = Depends(get_request_user_id),
    request_identity: dict | None = Depends(get_request_identity),
):
    """RAG 问答，完整流水线：改写→分类→检索→过滤→重排→生成→质量检查"""
    identity = _coerce_request_identity(user_id, request_identity)
    effective_user_id = identity["user_id"]
    model_name = request.model or settings.default_model
    session_id = conversation_service.ensure_knowledge_session(
        collection=request.collection,
        session_id=request.session_id,
        query=request.query,
        user_id=effective_user_id,
    )

    # 多模态：用户图片 base64 → 上传 OSS → 生成预签名 URL
    query_image_urls, query_image_oss_keys = _upload_query_images(
        request=request,
        user_id=effective_user_id,
        session_id=session_id,
    )
    query_image_url = query_image_urls[0] if query_image_urls else None
    query_image_oss_key = query_image_oss_keys[0] if query_image_oss_keys else None

    has_image = bool(_request_image_payloads(request) or query_image_urls or query_image_oss_keys)
    active_ticket = _find_active_ticket_safely(session_id, effective_user_id, request.collection or None)
    if active_ticket:
        mode, payload, decision = await _handle_active_clarification_reply(
            active_ticket=active_ticket,
            query=request.query,
            session_id=session_id,
            user_id=effective_user_id,
            kb_name=request.collection or None,
            model_name=model_name,
            has_image=has_image,
            query_image_url=query_image_url,
            query_image_urls=query_image_urls,
            query_image_oss_key=query_image_oss_key,
            query_image_oss_keys=query_image_oss_keys,
            force_multi_doc=request.force_multi_doc,
            keyword_filter=request.keyword_filter or None,
            channel=identity["channel"],
            sender_id=identity["sender_id"],
            requester_name=identity["requester_name"],
            user_name=identity["user_name"],
            entry_user_id=identity["entry_user_id"],
            entry_user_name=identity["entry_user_name"],
            entry_source=identity["entry_source"],
        )
        if mode == "ticket":
            return _ticket_response(
                ticket_result=payload,
                decision=decision,
                session_id=session_id,
                model_name=model_name,
            )
        if mode == "rag":
            persist_knowledge_result(
                session_id=session_id,
                query=decision.get("resolved_query") or request.query,
                result=payload,
                query_image_oss_key=query_image_oss_key,
                kb_name=payload.get("kb_name") or request.collection,
            )
            return _normal_response(payload)

    if _is_capability_question(request.query, has_image=has_image):
        result = _capability_answer_result(
            session_id=session_id,
            model_name=model_name,
            query=request.query,
            kb_name=request.collection or None,
        )
        persist_knowledge_result(
            session_id=session_id,
            query=request.query,
            result=result,
            query_image_oss_key=query_image_oss_key,
            kb_name=result.get("kb_name") or request.collection,
        )
        return _normal_response(result)

    rag_task = asyncio.create_task(
        invoke_knowledge_qa(
            query=request.query,
            model_name=model_name,
            session_id=session_id,
            collection=request.collection or None,
            force_multi_doc=request.force_multi_doc,
            keyword_filter=request.keyword_filter or None,
            query_image_url=query_image_url,
            query_image_oss_key=query_image_oss_key,
            persist=False,
        )
    )
    classifier_task = asyncio.create_task(
        asyncio.to_thread(
            classify_ticket_intent_with_llm,
            query=request.query,
            history=_load_classifier_history(session_id=session_id),
            has_image=has_image,
        )
    )
    await asyncio.sleep(0)

    decision = None
    done, _pending = await asyncio.wait(
        {classifier_task},
        timeout=TICKET_CLASSIFIER_INTERCEPT_TIMEOUT,
    )
    if classifier_task in done:
        with contextlib.suppress(Exception):
            decision = classifier_task.result()
    else:
        classifier_task.add_done_callback(_consume_background_task_exception)

    if _is_pre_rag_ticket_decision(decision):
        rag_result_for_workflow = None
        workflow = None
        if decision.get("intent_class") == INTENT_OPERATION:
            with contextlib.suppress(Exception):
                rag_result_for_workflow = await rag_task
            workflow = lookup_operation_workflow(
                query=request.query,
                kb_name=request.collection or None,
                decision=decision,
                rag_result=rag_result_for_workflow,
            )
        else:
            await _cancel_rag_task(rag_task)
        ticket_result = _create_ticket_result(
            session_id=session_id,
            user_id=effective_user_id,
            user_name=identity["user_name"],
            kb_name=request.collection or None,
            query=request.query,
            decision=decision,
            channel=identity["channel"],
            sender_id=identity["sender_id"],
            requester_name=identity["requester_name"],
            entry_user_id=identity["entry_user_id"],
            entry_user_name=identity["entry_user_name"],
            entry_source=identity["entry_source"],
            has_image=has_image,
            query_image_oss_key=query_image_oss_key,
            query_image_oss_keys=query_image_oss_keys,
            rag_result=rag_result_for_workflow,
            workflow=workflow,
        )
        if ticket_result:
            return _ticket_response(
                ticket_result=ticket_result,
                decision=decision,
                session_id=session_id,
                model_name=model_name,
            )

    result = await rag_task

    if decision is None and classifier_task.done():
        with contextlib.suppress(Exception):
            decision = classifier_task.result()
    if _is_pre_rag_ticket_decision(decision):
        workflow = None
        if decision.get("intent_class") == INTENT_OPERATION:
            workflow = lookup_operation_workflow(
                query=request.query,
                kb_name=request.collection or None,
                decision=decision,
                rag_result=result,
            )
        ticket_result = _create_ticket_result(
            session_id=session_id,
            user_id=effective_user_id,
            user_name=identity["user_name"],
            kb_name=request.collection or None,
            query=request.query,
            decision=decision,
            channel=identity["channel"],
            sender_id=identity["sender_id"],
            requester_name=identity["requester_name"],
            entry_user_id=identity["entry_user_id"],
            entry_user_name=identity["entry_user_name"],
            entry_source=identity["entry_source"],
            has_image=has_image,
            query_image_oss_key=query_image_oss_key,
            query_image_oss_keys=query_image_oss_keys,
            rag_result=result,
            workflow=workflow,
        )
        if ticket_result:
            return _ticket_response(
                ticket_result=ticket_result,
                decision=decision,
                session_id=session_id,
                model_name=model_name,
                request_id=result.get("request_id"),
            )

    if should_start_missing_knowledge_flow(result):
        missing_decision = await asyncio.to_thread(
            classify_ticket_intent_with_llm,
            query=request.query,
            history=_load_classifier_history(session_id=session_id),
            has_image=has_image,
            rag_result=result,
        )
        if _is_missing_knowledge_ticket_decision(missing_decision):
            ticket_result = _create_ticket_result(
                session_id=session_id,
                user_id=effective_user_id,
                user_name=identity["user_name"],
                kb_name=request.collection or None,
                query=request.query,
                decision=missing_decision,
                channel=identity["channel"],
                sender_id=identity["sender_id"],
                requester_name=identity["requester_name"],
                entry_user_id=identity["entry_user_id"],
                entry_user_name=identity["entry_user_name"],
                entry_source=identity["entry_source"],
                has_image=has_image,
                query_image_oss_key=query_image_oss_key,
                query_image_oss_keys=query_image_oss_keys,
                rag_result=result,
                )
            if ticket_result:
                return _ticket_response(
                    ticket_result=ticket_result,
                    decision=missing_decision,
                    session_id=session_id,
                    model_name=model_name,
                    request_id=result.get("request_id"),
                )
        decision = missing_decision

    persist_knowledge_result(
        session_id=session_id,
        query=request.query,
        result=result,
        query_image_oss_key=query_image_oss_key,
        kb_name=result.get("kb_name") or request.collection,
    )
    _record_ai_resolved_ticket_safely(
        session_id=session_id,
        user_id=effective_user_id,
        user_name=identity["user_name"],
        kb_name=result.get("kb_name") or request.collection,
        query=request.query,
        result=result,
        channel=identity["channel"],
        sender_id=identity["sender_id"],
        requester_name=identity["requester_name"],
        entry_user_id=identity["entry_user_id"],
        entry_user_name=identity["entry_user_name"],
        entry_source=identity["entry_source"],
        has_image=has_image,
        query_image_oss_key=query_image_oss_key,
    )
    _record_unanswered_normal_ticket_safely(
        session_id=session_id,
        user_id=effective_user_id,
        user_name=identity["user_name"],
        kb_name=result.get("kb_name") or request.collection,
        query=request.query,
        result=result,
        decision=decision,
        channel=identity["channel"],
        sender_id=identity["sender_id"],
        requester_name=identity["requester_name"],
        entry_user_id=identity["entry_user_id"],
        entry_user_name=identity["entry_user_name"],
        entry_source=identity["entry_source"],
        has_image=has_image,
        query_image_oss_key=query_image_oss_key,
        query_image_oss_keys=query_image_oss_keys,
    )
    return _normal_response(result)


@router.post("/stream", summary="Knowledge Base Q&A (SSE stream)")
async def knowledge_qa_stream(
    request: KnowledgeRequest,
    user_id: str = Depends(get_request_user_id),
    request_identity: dict | None = Depends(get_request_identity),
):
    """RAG 问答流式输出：event meta / delta / done / error，生成阶段为 OpenAI 兼容 Chat Completions stream。"""
    identity = _coerce_request_identity(user_id, request_identity)
    effective_user_id = identity["user_id"]
    model_name = request.model or settings.default_model
    session_id = conversation_service.ensure_knowledge_session(
        collection=request.collection,
        session_id=request.session_id,
        query=request.query,
        user_id=effective_user_id,
    )

    query_image_urls, query_image_oss_keys = _upload_query_images(
        request=request,
        user_id=effective_user_id,
        session_id=session_id,
    )
    query_image_url = query_image_urls[0] if query_image_urls else None
    query_image_oss_key = query_image_oss_keys[0] if query_image_oss_keys else None

    stream_headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    has_image = bool(_request_image_payloads(request) or query_image_urls or query_image_oss_keys)
    active_ticket = _find_active_ticket_safely(session_id, effective_user_id, request.collection or None)
    if active_ticket:
        mode, payload, decision = await _handle_active_clarification_reply(
            active_ticket=active_ticket,
            query=request.query,
            session_id=session_id,
            user_id=effective_user_id,
            kb_name=request.collection or None,
            model_name=model_name,
            has_image=has_image,
            query_image_url=query_image_url,
            query_image_urls=query_image_urls,
            query_image_oss_key=query_image_oss_key,
            query_image_oss_keys=query_image_oss_keys,
            force_multi_doc=request.force_multi_doc,
            keyword_filter=request.keyword_filter or None,
            channel=identity["channel"],
            sender_id=identity["sender_id"],
            requester_name=identity["requester_name"],
            user_name=identity["user_name"],
            entry_user_id=identity["entry_user_id"],
            entry_user_name=identity["entry_user_name"],
            entry_source=identity["entry_source"],
        )
        if mode == "ticket":
            return StreamingResponse(
                _single_done_stream(
                    _ticket_done_payload(
                        ticket_result=payload,
                        decision=decision,
                        session_id=session_id,
                        model_name=model_name,
                    )
                ),
                media_type="text/event-stream",
                headers=stream_headers,
            )
        if mode == "rag":
            persist_knowledge_result(
                session_id=session_id,
                query=decision.get("resolved_query") or request.query,
                result=payload,
                query_image_oss_key=query_image_oss_key,
                kb_name=payload.get("kb_name") or request.collection,
            )
            return StreamingResponse(
                _single_done_stream(
                    {
                        "request_id": payload.get("request_id") or str(uuid.uuid4()),
                        "session_id": session_id,
                        "answer": payload.get("answer") or "",
                        "confidence": payload.get("confidence"),
                        "sources": payload.get("sources") or [],
                        "model": model_name,
                        "thoughts": payload.get("thoughts") or {},
                        "image_map": payload.get("image_map") or {},
                        "finish_reason": payload.get("finish_reason") or "stop",
                    }
                ),
                media_type="text/event-stream",
                headers=stream_headers,
            )

    if _is_capability_question(request.query, has_image=has_image):
        result = _capability_answer_result(
            session_id=session_id,
            model_name=model_name,
            query=request.query,
            kb_name=request.collection or None,
        )
        persist_knowledge_result(
            session_id=session_id,
            query=request.query,
            result=result,
            query_image_oss_key=query_image_oss_key,
            kb_name=result.get("kb_name") or request.collection,
        )
        return StreamingResponse(
            _single_done_stream(
                {
                    "request_id": result["request_id"],
                    "session_id": session_id,
                    "answer": result["answer"],
                    "confidence": result["confidence"],
                    "sources": result["sources"],
                    "model": model_name,
                    "thoughts": result["thoughts"],
                    "image_map": result["image_map"] or {},
                    "finish_reason": result.get("finish_reason") or "stop",
                }
            ),
            media_type="text/event-stream",
            headers=stream_headers,
        )

    decision = None
    pre_rag_ticket_decision_task = asyncio.create_task(
        asyncio.to_thread(
            classify_ticket_intent_with_llm,
            query=request.query,
            history=_load_classifier_history(session_id=session_id),
            has_image=has_image,
        )
    )
    await asyncio.sleep(0)
    done, _pending = await asyncio.wait(
        {pre_rag_ticket_decision_task},
        timeout=STREAM_TICKET_CLASSIFIER_INTERCEPT_TIMEOUT,
    )
    if pre_rag_ticket_decision_task in done:
        with contextlib.suppress(Exception):
            decision = pre_rag_ticket_decision_task.result()
    if _is_pre_rag_ticket_decision(decision):
        rag_result_for_workflow = None
        workflow = None
        if decision.get("intent_class") == INTENT_OPERATION:
            rag_result_for_workflow = await invoke_knowledge_qa(
                query=request.query,
                model_name=model_name,
                session_id=session_id,
                collection=request.collection or None,
                force_multi_doc=request.force_multi_doc,
                keyword_filter=request.keyword_filter or None,
                query_image_url=query_image_url,
                query_image_oss_key=query_image_oss_key,
                persist=False,
            )
            workflow = lookup_operation_workflow(
                query=request.query,
                kb_name=request.collection or None,
                decision=decision,
                rag_result=rag_result_for_workflow,
            )
        ticket_result = _create_ticket_result(
            session_id=session_id,
            user_id=effective_user_id,
            user_name=identity["user_name"],
            kb_name=request.collection or None,
            query=request.query,
            decision=decision,
            channel=identity["channel"],
            sender_id=identity["sender_id"],
            requester_name=identity["requester_name"],
            entry_user_id=identity["entry_user_id"],
            entry_user_name=identity["entry_user_name"],
            entry_source=identity["entry_source"],
            has_image=has_image,
            query_image_oss_key=query_image_oss_key,
            query_image_oss_keys=query_image_oss_keys,
            rag_result=rag_result_for_workflow,
            workflow=workflow,
        )
        if ticket_result:
            return StreamingResponse(
                _single_done_stream(
                    _ticket_done_payload(
                        ticket_result=ticket_result,
                        decision=decision,
                        session_id=session_id,
                        model_name=model_name,
                    )
                ),
                media_type="text/event-stream",
                headers=stream_headers,
            )

    async def event_gen():
        async for chunk in stream_knowledge_qa_sse(
            query=request.query,
            model_name=model_name,
            session_id=session_id,
            collection=request.collection or None,
            force_multi_doc=request.force_multi_doc,
            keyword_filter=request.keyword_filter or None,
            query_image_url=query_image_url,
            query_image_oss_key=query_image_oss_key,
            query_image_oss_keys=query_image_oss_keys,
            convert_missing_knowledge_to_ticket=True,
            user_id=effective_user_id,
            user_name=identity["user_name"],
            channel=identity["channel"],
            sender_id=identity["sender_id"],
            requester_name=identity["requester_name"],
            entry_user_id=identity["entry_user_id"],
            entry_user_name=identity["entry_user_name"],
            entry_source=identity["entry_source"],
            pre_rag_ticket_decision_task=pre_rag_ticket_decision_task if decision is None else None,
        ):
            yield chunk

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers=stream_headers,
    )
