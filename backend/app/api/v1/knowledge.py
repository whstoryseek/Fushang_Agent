# -*- coding: utf-8 -*-
"""Knowledge API Routes"""
import asyncio
import contextlib
import logging
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.api.v1.auth_deps import get_request_user_id
from app.core.config import settings
from app.models.requests import KnowledgeRequest
from app.models.responses import KnowledgeResponse
from app.services import conversation_service
from app.services.knowledge_service import (
    invoke_knowledge_qa,
    persist_knowledge_result,
    stream_knowledge_qa_sse,
)
from app.services.service_ticket_service import (
    INTENT_AMBIGUOUS,
    INTENT_MISSING_KNOWLEDGE,
    INTENT_OPERATION,
    classify_ticket_intent_with_llm,
    process_ticket_clarification_turn,
    should_start_missing_knowledge_flow,
)

router = APIRouter(prefix="/knowledge")
logger = logging.getLogger(__name__)

TICKET_CLASSIFIER_INTERCEPT_TIMEOUT = 0.6


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


def _create_ticket_result(
    *,
    session_id: str,
    user_id: str,
    kb_name: str | None,
    query: str,
    decision: dict,
    has_image: bool,
    query_image_oss_key: str | None,
    rag_result: dict | None = None,
) -> dict | None:
    workflow = None
    if decision.get("intent_class") == INTENT_OPERATION:
        workflow = {"workflow_found": False, "required_fields": []}
    return process_ticket_clarification_turn(
        session_id=session_id,
        user_id=user_id,
        user_name=None,
        sender_id=None,
        requester_name=None,
        kb_name=kb_name,
        query=query,
        channel="web",
        decision=decision,
        workflow=workflow,
        rag_result=rag_result,
        has_image=has_image,
        query_image_oss_key=query_image_oss_key,
    )


@router.post("/", response_model=KnowledgeResponse, summary="Knowledge Base Q&A")
async def knowledge_qa(request: KnowledgeRequest, user_id: str = Depends(get_request_user_id)):
    """RAG 问答，完整流水线：改写→分类→检索→过滤→重排→生成→质量检查"""
    model_name = request.model or settings.default_model
    session_id = conversation_service.ensure_knowledge_session(
        collection=request.collection,
        session_id=request.session_id,
        query=request.query,
        user_id=user_id,
    )

    # 多模态：用户图片 base64 → 上传 OSS → 生成预签名 URL
    query_image_url = None
    query_image_oss_key = None
    if request.query_image:
        try:
            import base64, uuid
            from app.services.oss_service import get_oss_service
            img_bytes = base64.b64decode(request.query_image)
            img_uuid = uuid.uuid4().hex[:12]
            # 路径：query_images/{user_id}/{kb_name}/{session_id}/{uuid}.jpg
            kb_name = request.collection or "default"
            oss_path = f"query_images/{user_id}/{kb_name}/{session_id}"
            query_image_oss_key = get_oss_service().upload_file(
                oss_path, f"{img_uuid}.jpg", img_bytes
            )
            query_image_url = get_oss_service().get_presigned_url(query_image_oss_key, expires=600)
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).warning(f"用户查询图片上传失败，降级为纯文字检索: {e}")

    has_image = bool(request.query_image or query_image_url or query_image_oss_key)
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
            history=[],
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
        await _cancel_rag_task(rag_task)
        ticket_result = _create_ticket_result(
            session_id=session_id,
            user_id=user_id,
            kb_name=request.collection or None,
            query=request.query,
            decision=decision,
            has_image=has_image,
            query_image_oss_key=query_image_oss_key,
        )
        if ticket_result:
            return _ticket_response(
                ticket_result=ticket_result,
                decision=decision,
                session_id=session_id,
                model_name=model_name,
            )

    result = await rag_task

    if should_start_missing_knowledge_flow(result):
        missing_decision = await asyncio.to_thread(
            classify_ticket_intent_with_llm,
            query=request.query,
            history=[],
            has_image=has_image,
            rag_result=result,
        )
        if _is_missing_knowledge_ticket_decision(missing_decision):
            ticket_result = _create_ticket_result(
                session_id=session_id,
                user_id=user_id,
                kb_name=request.collection or None,
                query=request.query,
                decision=missing_decision,
                has_image=has_image,
                query_image_oss_key=query_image_oss_key,
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

    persist_knowledge_result(
        session_id=session_id,
        query=request.query,
        result=result,
        query_image_oss_key=query_image_oss_key,
        kb_name=result.get("kb_name") or request.collection,
    )
    return _normal_response(result)


@router.post("/stream", summary="Knowledge Base Q&A (SSE stream)")
async def knowledge_qa_stream(request: KnowledgeRequest, user_id: str = Depends(get_request_user_id)):
    """RAG 问答流式输出：event meta / delta / done / error，生成阶段为 OpenAI 兼容 Chat Completions stream。"""
    model_name = request.model or settings.default_model
    session_id = conversation_service.ensure_knowledge_session(
        collection=request.collection,
        session_id=request.session_id,
        query=request.query,
        user_id=user_id,
    )

    query_image_url = None
    query_image_oss_key = None
    if request.query_image:
        try:
            import base64
            import uuid as _uuid
            from app.services.oss_service import get_oss_service
            img_bytes = base64.b64decode(request.query_image)
            img_uuid = _uuid.uuid4().hex[:12]
            kb_name = request.collection or "default"
            oss_path = f"query_images/{user_id}/{kb_name}/{session_id}"
            query_image_oss_key = get_oss_service().upload_file(
                oss_path, f"{img_uuid}.jpg", img_bytes
            )
            query_image_url = get_oss_service().get_presigned_url(query_image_oss_key, expires=600)
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).warning(f"用户查询图片上传失败，降级为纯文字检索: {e}")

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
