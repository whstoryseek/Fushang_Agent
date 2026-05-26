# -*- coding: utf-8 -*-
"""本地管理员认证服务。"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
import jwt

from app.core.config import settings
from app.db import get_auth_user_repository

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    if not password or not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(
    *,
    subject: str,
    username: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    expires = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.auth_token_expire_minutes)
    )
    payload = {
        "sub": subject,
        "username": username,
        "role": role,
        "exp": expires,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.auth_secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, settings.auth_secret_key, algorithms=[ALGORITHM])


def authenticate_admin(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = get_auth_user_repository().get_by_username(username)
    if not user or not user.get("is_active"):
        return None
    if user.get("role") != "admin":
        return None
    if not verify_password(password, user.get("password_hash", "")):
        return None
    return user


def seed_admin_user(username: str, password: str, repo=None) -> Dict[str, Any]:
    if not password:
        raise EnvironmentError("ADMIN_PASSWORD must be configured before starting the API")
    repo = repo or get_auth_user_repository()
    password_hash = hash_password(password)
    existing = repo.get_by_username(username)
    if existing:
        return repo.update_password(user_id=existing["id"], password_hash=password_hash) or existing
    return repo.create_user(username=username, password_hash=password_hash, role="admin")


def seed_admin_from_settings() -> Dict[str, Any]:
    return seed_admin_user(settings.admin_username, settings.admin_password)
