# -*- coding: utf-8 -*-
"""Knowledge API Routes"""
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import uuid
import asyncio
import logging

from app.api.v1.auth_deps import get_request_identity
from app.core.config import settings
from app.models.requests import KnowledgeRequest
from app.models.responses import KnowledgeResponse
from app.services import conversation_service
from app.services.knowledge_service import (
    invoke_knowledge_qa,
    persist_clarification_message,
    retrieve_operation_workflow,
    stream_clarification_sse,
    stream_knowledge_qa_sse,
)
from app.services.service_ticket_service import (
    classify_operation_query_with_llm,
    extract_operation_workflow_requirements,
    should_clarify_query,
    should_start_missing_knowledge_flow,
    start_missing_knowledge_clarification,
)

router = APIRouter(prefix="/knowledge")
logger = logging.getLogger(__name__)


def _upload_query_image_if_needed(request: KnowledgeRequest, *, user_id: str, kb_name: str, session_id: str):
    if not request.query_image:
        return None, None
    try:
        import base64
        from app.services.oss_service import get_oss_service

        img_bytes = base64.b64decode(request.query_image)
        img_uuid = uuid.uuid4().hex[:12]
        oss_path = f"query_images/{user_id}/{kb_name}/{session_id}"
        query_image_oss_key = get_oss_service().upload_file(
            oss_path, f"{img_uuid}.jpg", img_bytes
        )
        query_image_url = get_oss_service().get_presigned_url(query_image_oss_key, expires=600)
        return query_image_url, query_image_oss_key
    except Exception as e:
        import logging as _log
        _log.getLogger(__name__).warning(f"用户查询图片上传失败，降级为纯文字检索: {e}")
        return None, None


def _fallback_operation_workflow():
    return {
        "workflow_found": False,
        "workflow_summary": "",
        "required_fields": [],
        "field_labels": {},
        "fallback_used": True,
    }


async def _resolve_operation_workflow(
    request: KnowledgeRequest,
    *,
    model_name: str,
    session_id: str,
    query_image_url: str | None,
    user_id: str,
):
    workflow_result = await retrieve_operation_workflow(
        query=request.query,
        model_name=model_name,
        session_id=session_id,
        collection=request.collection or None,
        force_multi_doc=request.force_multi_doc,
        keyword_filter=request.keyword_filter or None,
        query_image_url=query_image_url,
        user_id=user_id,
    )
    workflow_sources = workflow_result.get("sources") or []
    try:
        workflow = await asyncio.to_thread(
            extract_operation_workflow_requirements,
            request.query,
            workflow_sources,
        )
    except Exception as exc:
        logger.warning("Operation workflow extraction failed; falling back to generic clarification: %s", exc)
        workflow = _fallback_operation_workflow()
    return workflow_result, workflow_sources, workflow


def _operation_thoughts(clarification: dict, workflow_result: dict, workflow: dict) -> dict:
    thoughts = {
        "classifier_source": clarification.get("source") or "rule",
        "operation_workflow_found": bool(workflow.get("workflow_found")),
        "operation_required_fields": list(workflow.get("required_fields") or []),
    }
    if workflow_result.get("thoughts"):
        thoughts["operation_workflow_retrieval"] = workflow_result.get("thoughts")
    return thoughts


@router.post("/", response_model=KnowledgeResponse, summary="Knowledge Base Q&A")
async def knowledge_qa(request: KnowledgeRequest, identity: dict = Depends(get_request_identity)):
    """RAG 问答，完整流水线：改写→分类→检索→过滤→重排→生成→质量检查"""
    model_name = request.model or settings.default_model
    user_id = identity.get("user_id") or "guest_default"
    session_id = conversation_service.ensure_knowledge_session(
        collection=request.collection,
        session_id=request.session_id,
        query=request.query,
        user_id=user_id,
    )
    query_image_url, query_image_oss_key = _upload_query_image_if_needed(
        request,
        user_id=user_id,
        kb_name=request.collection or "default",
        session_id=session_id,
    )

    clarification_result = persist_clarification_message(
        session_id=session_id,
        query=request.query,
        answer=None,
        kb_name=request.collection,
        user_id=user_id,
        user_name=identity.get("user_name"),
        channel=identity.get("channel") or "web",
        sender_id=identity.get("sender_id"),
        requester_name=identity.get("requester_name"),
        has_image=bool(query_image_oss_key),
        query_image_oss_key=query_image_oss_key,
        continue_only=True,
    )
    if clarification_result:
        return KnowledgeResponse(
            status_code=200,
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            answer=clarification_result["answer"],
            confidence=0.0,
            sources=[],
            model=model_name,
            finish_reason=clarification_result.get("finish_reason") or "clarification",
            thoughts={"clarification_required": True, "clarification": clarification_result.get("clarification")},
            image_map=None,
        )

    clarification = should_clarify_query(request.query, has_image=bool(query_image_oss_key))
    if not clarification["should_clarify"]:
        clarification = await asyncio.to_thread(
            classify_operation_query_with_llm,
            request.query,
            has_image=bool(query_image_oss_key),
        )
    if clarification["should_clarify"]:
        workflow_result = {}
        workflow_sources = []
        workflow = _fallback_operation_workflow()
        if (clarification.get("reason") or "operation_required") == "operation_required":
            workflow_result, workflow_sources, workflow = await _resolve_operation_workflow(
                request,
                model_name=model_name,
                session_id=session_id,
                query_image_url=query_image_url,
                user_id=user_id,
            )
        clarification_result = persist_clarification_message(
            session_id=session_id,
            query=request.query,
            answer=clarification.get("message"),
            kb_name=request.collection,
            user_id=user_id,
            user_name=identity.get("user_name"),
            channel=identity.get("channel") or "web",
            sender_id=identity.get("sender_id"),
            requester_name=identity.get("requester_name"),
            has_image=bool(query_image_oss_key),
            query_image_oss_key=query_image_oss_key,
            reason=clarification.get("reason") or "operation_required",
            force_start=True,
            workflow=workflow,
            workflow_sources=workflow_sources,
        )
        answer = (clarification_result or {}).get("answer") or clarification.get("message") or ""
        operation_thoughts = _operation_thoughts(clarification, workflow_result, workflow)
        return KnowledgeResponse(
            status_code=200,
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            answer=answer,
            confidence=0.0,
            sources=[],
            model=model_name,
            finish_reason=(clarification_result or {}).get("finish_reason") or "clarification",
            thoughts={
                "clarification_required": True,
                "clarification": (clarification_result or {}).get("clarification"),
                **operation_thoughts,
            },
            image_map=None,
        )

    result = await invoke_knowledge_qa(
        query=request.query,
        model_name=model_name,
        session_id=session_id,
        collection=request.collection or None,
        force_multi_doc=request.force_multi_doc,
        keyword_filter=request.keyword_filter or None,
        query_image_url=query_image_url,
        query_image_oss_key=query_image_oss_key,
        user_id=user_id,
        user_name=identity.get("user_name"),
        channel=identity.get("channel") or "web",
        sender_id=identity.get("sender_id"),
        requester_name=identity.get("requester_name"),
        persist=False,
    )
    if should_start_missing_knowledge_flow(result):
        missing_result = start_missing_knowledge_clarification(
            session_id=session_id,
            user_id=user_id,
            user_name=identity.get("user_name"),
            sender_id=identity.get("sender_id"),
            requester_name=identity.get("requester_name"),
            kb_name=request.collection,
            query=request.query,
            channel=identity.get("channel") or "web",
            rag_result=result,
            has_image=bool(query_image_oss_key),
            query_image_oss_key=query_image_oss_key,
        )
        return KnowledgeResponse(
            status_code=200,
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            answer=missing_result["answer"],
            confidence=0.0,
            sources=[],
            model=model_name,
            finish_reason=missing_result.get("finish_reason") or "clarification",
            thoughts={
                "clarification_required": True,
                "clarification": missing_result.get("clarification"),
                "missing_knowledge_started": True,
            },
            image_map=None,
        )

    from app.services.knowledge_service import _persist_conversation_messages
    _persist_conversation_messages(
        session_id,
        request.query,
        result.get("answer") or "",
        result.get("sources") or [],
        result.get("confidence"),
        query_image_oss_key,
        used_fallback=result.get("used_fallback", False),
        fallback_reason=result.get("fallback_reason"),
        quality_passed=result.get("quality_passed"),
        quality_level=result.get("quality_level"),
        kb_name=request.collection,
        user_id=user_id,
        user_name=identity.get("user_name"),
        channel=identity.get("channel") or "web",
        sender_id=identity.get("sender_id"),
        requester_name=identity.get("requester_name") or identity.get("user_name"),
    )
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


@router.post("/stream", summary="Knowledge Base Q&A (SSE stream)")
async def knowledge_qa_stream(request: KnowledgeRequest, identity: dict = Depends(get_request_identity)):
    """RAG 问答流式输出：event meta / delta / done / error，生成阶段为 OpenAI 兼容 Chat Completions stream。"""
    model_name = request.model or settings.default_model
    user_id = identity.get("user_id") or "guest_default"
    session_id = conversation_service.ensure_knowledge_session(
        collection=request.collection,
        session_id=request.session_id,
        query=request.query,
        user_id=user_id,
    )
    query_image_url, query_image_oss_key = _upload_query_image_if_needed(
        request,
        user_id=user_id,
        kb_name=request.collection or "default",
        session_id=session_id,
    )

    clarification_result = persist_clarification_message(
        session_id=session_id,
        query=request.query,
        answer=None,
        kb_name=request.collection,
        user_id=user_id,
        user_name=identity.get("user_name"),
        channel=identity.get("channel") or "web",
        sender_id=identity.get("sender_id"),
        requester_name=identity.get("requester_name"),
        has_image=bool(query_image_oss_key),
        query_image_oss_key=query_image_oss_key,
        continue_only=True,
    )
    if clarification_result:
        async def clarification_events():
            payload = {
                "request_id": str(uuid.uuid4()),
                "session_id": session_id,
                "model": model_name,
                "thoughts": {"clarification_required": True, "clarification": clarification_result.get("clarification")},
                "sources": [],
                "image_map": {},
            }
            from app.services.knowledge_service import _sse
            yield _sse("meta", payload)
            yield _sse("done", {
                **payload,
                "answer": clarification_result["answer"],
                "confidence": 0.0,
                "finish_reason": clarification_result.get("finish_reason") or "clarification",
            })

        return StreamingResponse(
            clarification_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    clarification = should_clarify_query(request.query, has_image=bool(query_image_oss_key))
    if not clarification["should_clarify"]:
        clarification = await asyncio.to_thread(
            classify_operation_query_with_llm,
            request.query,
            has_image=bool(query_image_oss_key),
        )
    if clarification["should_clarify"]:
        workflow_result = {}
        workflow_sources = []
        workflow = _fallback_operation_workflow()
        if (clarification.get("reason") or "operation_required") == "operation_required":
            workflow_result, workflow_sources, workflow = await _resolve_operation_workflow(
                request,
                model_name=model_name,
                session_id=session_id,
                query_image_url=query_image_url,
                user_id=user_id,
            )
        return StreamingResponse(
            stream_clarification_sse(
                query=request.query,
                answer=clarification.get("message") or "",
                model_name=model_name,
                session_id=session_id,
                kb_name=request.collection,
                user_id=user_id,
                user_name=identity.get("user_name"),
                channel=identity.get("channel") or "web",
                sender_id=identity.get("sender_id"),
                requester_name=identity.get("requester_name"),
                has_image=bool(query_image_oss_key),
                query_image_oss_key=query_image_oss_key,
                reason=clarification.get("reason") or "operation_required",
                force_start=True,
                workflow=workflow,
                workflow_sources=workflow_sources,
                thoughts=_operation_thoughts(clarification, workflow_result, workflow),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
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
            user_id=user_id,
            user_name=identity.get("user_name"),
            channel=identity.get("channel") or "web",
            sender_id=identity.get("sender_id"),
            requester_name=identity.get("requester_name"),
        ):
            yield chunk

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
