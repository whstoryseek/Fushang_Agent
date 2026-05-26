# -*- coding: utf-8 -*-
"""普通用户可访问的知识库列表。"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.db import get_kb_repository

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


@router.get("")
async def list_public_knowledge_bases():
    kbs = get_kb_repository().list_all()
    collections = [
        {
            "name": kb["name"],
            "display_name": kb.get("display_name"),
            "description": kb.get("description"),
            "kb_type": kb.get("kb_type", "standard"),
            "image_mode": bool(kb.get("image_mode", False)),
        }
        for kb in kbs
    ]
    return JSONResponse(content={"success": True, "data": {"collections": collections, "total": len(collections)}})
