# -*- coding: utf-8 -*-
"""未回答问题与对话统计 API（独立表 unanswered_question）"""
from typing import Optional
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.db import get_unanswered_repository

router = APIRouter(prefix="/unanswered", tags=["admin-unanswered"])


class UnansweredListResponse(BaseModel):
    total: int
    items: list


@router.get("/messages")
async def list_unanswered_messages(
    kb_name: Optional[str] = Query(None, description="按知识库筛选"),
    start_date: Optional[str] = Query(None, description="开始日期，格式 YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="结束日期，格式 YYYY-MM-DD"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """查询 AI 未回答的消息列表（独立表，长期保留）"""
    repo = get_unanswered_repository()

    total = repo.count(
        kb_name=kb_name,
        start_date=start_date,
        end_date=end_date,
    )
    items = repo.list(
        kb_name=kb_name,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )

    return JSONResponse(content={"success": True, "data": {"total": total, "items": items}})


@router.get("/stats")
async def conversation_stats(
    kb_name: Optional[str] = Query(None, description="按知识库筛选"),
    start_date: Optional[str] = Query(None, description="开始日期，格式 YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="结束日期，格式 YYYY-MM-DD"),
):
    """未回答问题统计：总数 + 按日期分布"""
    repo = get_unanswered_repository()

    stats_data = repo.stats(
        kb_name=kb_name,
        start_date=start_date,
        end_date=end_date,
    )

    total = stats_data["total"]
    # unanswered_question 表中所有记录均为未回答问题（触发 fallback 时写入）
    unanswered = total
    unanswered_rate = round((unanswered / total) * 100, 1) if total > 0 else 0.0

    return JSONResponse(content={
        "success": True,
        "data": {
            "total": total,
            "unanswered": unanswered,
            "unanswered_rate": unanswered_rate,
            "daily": [
                {"date": d["date"], "total": d["count"], "unanswered": d["count"]}
                for d in stats_data["daily"]
            ],
        }
    })
