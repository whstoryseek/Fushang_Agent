# -*- coding: utf-8 -*-
"""Authentication dependencies and request identity helpers."""

import re
from typing import Any, Dict, Optional
from urllib.parse import unquote

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.services.auth_service import decode_access_token

bearer = HTTPBearer(auto_error=False)
GUEST_ID_RE = re.compile(r"^guest_[A-Za-z0-9_-]{8,80}$")
ENTRY_ID_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,120}$")
ADMIN_ROLES = {"super_admin", "sub_admin"}


def _decode_header_text(value: str) -> str:
    try:
        return unquote(value or "").strip()
    except Exception:
        return (value or "").strip()


def _pick_first(*values: Optional[str], decode: bool = False) -> str:
    for value in values:
        current = _decode_header_text(value) if decode else str(value or "").strip()
        if current:
            return current
    return ""


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
    effective_role = user.get("admin_role") or user.get("role")
    if effective_role not in ADMIN_ROLES:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="权限不足")
    return user


def require_super_admin(user: Dict[str, Any] = Depends(require_authenticated_user)) -> Dict[str, Any]:
    effective_role = user.get("admin_role") or user.get("role")
    if effective_role != "super_admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="仅主管理员可操作")
    return user


def get_request_identity(request: Request) -> Dict[str, Optional[str]]:
    query_user_id = _pick_first(
        request.query_params.get("userId"),
        request.query_params.get("entry_user_id"),
        request.query_params.get("user_id"),
    )
    header_user_id = _pick_first(
        request.headers.get("X-Entry-User-Id"),
        request.headers.get("X-Wecom-User-Id"),
        request.headers.get("X-User-Id"),
    )
    guest_id = _pick_first(request.headers.get("X-Guest-Id"))
    nickname = _pick_first(
        request.query_params.get("nickName"),
        request.query_params.get("entry_user_name"),
        request.query_params.get("nickname"),
        request.headers.get("X-Entry-User-Name"),
        request.headers.get("X-Wecom-Nickname"),
        request.headers.get("X-User-Name"),
        decode=True,
    )
    entry_source = _pick_first(
        request.query_params.get("entry_source"),
        request.headers.get("X-Entry-Source"),
    )
    sender_id = _pick_first(
        request.query_params.get("sender_id"),
        request.headers.get("X-Wecom-Sender-Id"),
        request.headers.get("X-Sender-Id"),
    )
    requester_name = _pick_first(
        request.query_params.get("requester_name"),
        request.headers.get("X-Requester-Name"),
        nickname,
        decode=True,
    )
    channel = _pick_first(
        request.query_params.get("channel"),
        request.headers.get("X-Channel"),
    ).lower()

    sender_id = sender_id if sender_id and ENTRY_ID_RE.match(sender_id) else None
    requester_name = requester_name or None

    entry_user_id = query_user_id if query_user_id and ENTRY_ID_RE.match(query_user_id) else None
    if not entry_user_id and header_user_id and ENTRY_ID_RE.match(header_user_id):
        entry_user_id = header_user_id

    entry_user_name = nickname or None
    has_platform_style_entry = bool(
        request.query_params.get("userId")
        or request.query_params.get("nickName")
        or request.headers.get("X-Entry-User-Id")
    )
    if not entry_source and entry_user_id:
        entry_source = "renruikeji_sso" if has_platform_style_entry else "legacy_query"

    if entry_user_id:
        return {
            "user_id": entry_user_id,
            "user_name": entry_user_name,
            "channel": channel or "h5",
            "sender_id": sender_id,
            "requester_name": requester_name,
            "entry_user_id": entry_user_id,
            "entry_user_name": entry_user_name,
            "entry_source": entry_source or None,
        }

    if GUEST_ID_RE.match(guest_id):
        return {
            "user_id": guest_id,
            "user_name": entry_user_name,
            "channel": channel or "web",
            "sender_id": sender_id,
            "requester_name": requester_name,
            "entry_user_id": None,
            "entry_user_name": entry_user_name,
            "entry_source": "anonymous",
        }

    return {
        "user_id": "guest_default",
        "user_name": entry_user_name,
        "channel": channel or "web",
        "sender_id": sender_id,
        "requester_name": requester_name,
        "entry_user_id": None,
        "entry_user_name": entry_user_name,
        "entry_source": "anonymous",
    }


def get_request_user_id(request: Request) -> str:
    return get_request_identity(request)["user_id"] or "guest_default"


def _legacy_get_request_user_id(request: Request) -> str:
    guest_id = (request.headers.get("X-Guest-Id") or "").strip()
    if GUEST_ID_RE.match(guest_id):
        return guest_id
    return "guest_default"
