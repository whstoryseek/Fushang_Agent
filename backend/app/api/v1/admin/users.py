# -*- coding: utf-8 -*-
"""管理员账号管理 API。"""
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.v1.auth_deps import require_super_admin
from app.db import get_auth_user_repository
from app.services.auth_service import hash_password

router = APIRouter(prefix="/users", tags=["admin-users"])

DEFAULT_SUB_ADMIN_PASSWORD = "88888888"


class CreateSubAdminRequest(BaseModel):
    username: str


def _serialize_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "is_active": user.get("is_active", True),
        "created_at": user.get("created_at"),
        "updated_at": user.get("updated_at"),
    }


@router.get("")
async def list_admin_users(_super_admin=Depends(require_super_admin)):
    repo = get_auth_user_repository()
    users = [_serialize_user(user) for user in repo.list_users()]
    return JSONResponse(content={"success": True, "data": {"users": users}})


@router.post("/sub-admin")
async def create_sub_admin(
    body: CreateSubAdminRequest,
    _super_admin=Depends(require_super_admin),
):
    username = body.username.strip()
    if not username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="账号不能为空")

    repo = get_auth_user_repository()
    if repo.get_by_username(username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="账号已存在")

    user = repo.create_user(
        username=username,
        password_hash=hash_password(DEFAULT_SUB_ADMIN_PASSWORD),
        role="sub_admin",
    )
    return JSONResponse(
        content={
            "success": True,
            "data": {
                "user": _serialize_user(user),
                "default_password": DEFAULT_SUB_ADMIN_PASSWORD,
            },
        }
    )
