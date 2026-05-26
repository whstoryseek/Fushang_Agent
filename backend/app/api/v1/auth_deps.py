# -*- coding: utf-8 -*-
"""认证与匿名用户依赖。"""
import re
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.services.auth_service import decode_access_token

bearer = HTTPBearer(auto_error=False)
GUEST_ID_RE = re.compile(r"^guest_[A-Za-z0-9_-]{8,80}$")


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


def get_request_user_id(request: Request) -> str:
    guest_id = (request.headers.get("X-Guest-Id") or "").strip()
    if GUEST_ID_RE.match(guest_id):
        return guest_id
    return "guest_default"
