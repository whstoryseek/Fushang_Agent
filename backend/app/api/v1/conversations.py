# -*- coding: utf-8 -*-
"""对话会话 API"""
from typing import Optional
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.v1.auth_deps import get_request_user_id
from app.services import conversation_service

router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateSessionRequest(BaseModel):
    kb_name: str
    title: str = "新会话"
    user_id: str = "default"


@router.get("")
async def list_sessions(
    kb_name: Optional[str] = Query(default=None),
    user_id: str = Depends(get_request_user_id),
):
    result = conversation_service.list_sessions(kb_name=kb_name, user_id=user_id)
    return JSONResponse(content={"success": True, "data": result})


@router.post("")
async def create_session(body: CreateSessionRequest, user_id: str = Depends(get_request_user_id)):
    session = conversation_service.create_session(
        kb_name=body.kb_name, user_id=user_id, title=body.title
    )
    return JSONResponse(content={"success": True, "data": session})


@router.get("/{session_id}/messages")
async def get_messages(
    session_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    user_id: str = Depends(get_request_user_id),
):
    result = conversation_service.get_session_messages(session_id, limit=limit, user_id=user_id)
    return JSONResponse(content={"success": True, "data": result})


@router.delete("/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(get_request_user_id)):
    conversation_service.delete_session(session_id, user_id=user_id)
    return JSONResponse(content={"success": True, "message": "会话已删除"})
