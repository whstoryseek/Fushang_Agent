# -*- coding: utf-8 -*-
"""管理员登录 API。"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.config import settings
from app.api.v1.auth_deps import require_admin
from app.services.auth_service import authenticate_admin, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
async def login(body: LoginRequest):
    user = authenticate_admin(body.username, body.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    token = create_access_token(subject=user["id"], username=user["username"], role=user["role"])
    return JSONResponse(content={
        "success": True,
        "data": {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": settings.auth_token_expire_minutes * 60,
            "user": {"id": user["id"], "username": user["username"], "role": user["role"]},
        },
    })


@router.get("/me")
async def me(admin=Depends(require_admin)):
    return JSONResponse(content={
        "success": True,
        "data": {"username": admin["username"], "role": admin["role"]},
    })


@router.post("/logout")
async def logout():
    return JSONResponse(content={"success": True, "message": "已退出登录"})
