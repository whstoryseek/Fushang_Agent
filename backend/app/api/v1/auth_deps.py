# -*- coding: utf-8 -*-
"""认证与匿名用户依赖。"""
import re
from urllib.parse import unquote
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.services.auth_service import decode_access_token

bearer = HTTPBearer(auto_error=False)
GUEST_ID_RE = re.compile(r"^guest_[A-Za-z0-9_-]{8,80}$")
ENTRY_ID_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,120}$")


def _decode_header_text(value: str) -> str:
    try:
        return unquote(value or "").strip()
    except Exception:
        return (value or "").strip()


def require_authenticated_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> Dict[str, Any]:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先登录管理员账号",
        )
    try:
        return decode_access_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录凭证无效")


def require_admin(user: Dict[str, Any] = Depends(require_authenticated_user)) -> Dict[str, Any]:
    if user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="权限不足")
    return user


def get_request_identity(request: Request) -> Dict[str, Optional[str]]:
    query_user_id = (request.query_params.get("user_id") or "").strip()
    header_user_id = (
        request.headers.get("X-Wecom-User-Id")
        or request.headers.get("X-User-Id")
        or ""
    ).strip()
    guest_id = (request.headers.get("X-Guest-Id") or "").strip()
    nickname = (
        request.query_params.get("nickname")
        or _decode_header_text(request.headers.get("X-Wecom-Nickname") or "")
        or _decode_header_text(request.headers.get("X-User-Name") or "")
        or ""
    ).strip()
    sender_id = (
        request.query_params.get("sender_id")
        or request.headers.get("X-Wecom-Sender-Id")
        or request.headers.get("X-Sender-Id")
        or ""
    ).strip()
    requester_name = (
        request.query_params.get("requester_name")
        or _decode_header_text(request.headers.get("X-Requester-Name") or "")
        or nickname
        or ""
    ).strip()
    channel = (
        request.query_params.get("channel")
        or request.headers.get("X-Channel")
        or ""
    ).strip().lower()

    sender_id = sender_id if sender_id and ENTRY_ID_RE.match(sender_id) else None
    requester_name = requester_name or None

    if query_user_id and ENTRY_ID_RE.match(query_user_id):
        return {
            "user_id": query_user_id,
            "user_name": nickname or None,
            "channel": channel or "h5",
            "sender_id": sender_id,
            "requester_name": requester_name,
        }
    if header_user_id and ENTRY_ID_RE.match(header_user_id):
        return {
            "user_id": header_user_id,
            "user_name": nickname or None,
            "channel": channel or "h5",
            "sender_id": sender_id,
            "requester_name": requester_name,
        }
    if GUEST_ID_RE.match(guest_id):
        return {
            "user_id": guest_id,
            "user_name": nickname or None,
            "channel": channel or "web",
            "sender_id": sender_id,
            "requester_name": requester_name,
        }
    return {
        "user_id": "guest_default",
        "user_name": nickname or None,
        "channel": channel or "web",
        "sender_id": sender_id,
        "requester_name": requester_name,
    }


def get_request_user_id(request: Request) -> str:
    return get_request_identity(request)["user_id"] or "guest_default"


def _legacy_get_request_user_id(request: Request) -> str:
    guest_id = (request.headers.get("X-Guest-Id") or "").strip()
    if GUEST_ID_RE.match(guest_id):
        return guest_id
    return "guest_default"
