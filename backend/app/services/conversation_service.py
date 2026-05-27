# -*- coding: utf-8 -*-
"""
对话会话业务逻辑
"""
import logging
from typing import Optional

from app.core.exceptions import ForbiddenError, NotFoundError
from app.db import get_conversation_repository

logger = logging.getLogger(__name__)
DEFAULT_SESSION_ID = "default"


def _assert_session_owner(session: dict, user_id: Optional[str]) -> None:
    if user_id and session.get("user_id") != user_id:
        raise ForbiddenError("无权访问该会话")


def create_session(kb_name: str, user_id: str = "default", title: str = "新会话") -> dict:
    return get_conversation_repository().create_session(
        kb_name=kb_name, user_id=user_id, title=title
    )


def list_sessions(kb_name: Optional[str] = None, user_id: str = "default") -> dict:
    sessions = get_conversation_repository().list_sessions(kb_name=kb_name, user_id=user_id)
    return {"sessions": sessions, "total": len(sessions)}


def build_session_title(query: str, max_length: int = 24) -> str:
    text = " ".join((query or "").strip().split())
    if not text:
        return "新会话"
    if len(text) <= max_length:
        return text
    return f"{text[:max_length].rstrip()}..."


def ensure_knowledge_session(
    *,
    collection: Optional[str],
    session_id: Optional[str],
    query: str,
    user_id: str = "default",
) -> str:
    normalized_session_id = (session_id or "").strip()
    if normalized_session_id and normalized_session_id != DEFAULT_SESSION_ID:
        try:
            session = get_conversation_repository().get_session(normalized_session_id)
        except Exception as exc:
            logger.warning("会话 ID 无法查询，按新会话处理: %s", exc)
            session = None
            normalized_session_id = ""
        if session:
            _assert_session_owner(session, user_id)
            return normalized_session_id
    if not collection:
        return normalized_session_id or DEFAULT_SESSION_ID

    session = create_session(
        kb_name=collection,
        user_id=user_id,
        title=build_session_title(query),
    )
    return session.get("id") or DEFAULT_SESSION_ID


def get_session_messages(session_id: str, limit: int = 100, user_id: Optional[str] = None) -> dict:
    repo = get_conversation_repository()
    session = repo.get_session(session_id)
    if not session:
        raise NotFoundError(f"会话不存在: {session_id}")
    _assert_session_owner(session, user_id)
    messages = repo.list_messages(session_id, limit=limit)
    return {"session": session, "messages": messages, "total": len(messages)}


def delete_session(session_id: str, user_id: Optional[str] = None) -> None:
    repo = get_conversation_repository()
    session = repo.get_session(session_id)
    if not session:
        raise NotFoundError(f"会话不存在: {session_id}")
    _assert_session_owner(session, user_id)

    # 清理用户查询图片（query_images/ 路径，生命周期归属对话）
    try:
        messages = repo.list_messages(session_id, limit=1000)
        oss_keys = [m["query_image_oss_key"] for m in messages if m.get("query_image_oss_key")]
        if oss_keys:
            from app.services.oss_service import get_oss_service
            get_oss_service().delete_objects(oss_keys)
            logger.info(f"已清理会话 {session_id} 的 {len(oss_keys)} 张查询图片")
    except Exception as e:
        logger.warning(f"查询图片 OSS 清理失败（继续删除会话）: {e}")

    # 删除会话（CASCADE 删 conversation_message）
    # 切片图片（conversation_assets/）生命周期归属知识库文件，不随对话删除
    repo.delete_session(session_id)
    logger.info(f"会话已删除: {session_id}")
