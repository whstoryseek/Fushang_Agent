# -*- coding: utf-8 -*-
"""服务记录 / 工单后台 API。"""
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services import service_ticket_service

router = APIRouter(prefix="/service-tickets", tags=["admin-service-tickets"])


class UpdateTicketRequest(BaseModel):
    status: Optional[str] = None
    answer: Optional[str] = None
    note: Optional[str] = None


class UpdateTicketChunkRequest(BaseModel):
    content: str


@router.get("")
async def list_service_tickets(
    status: Optional[str] = Query(None),
    kb_name: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    data = service_ticket_service.list_tickets(
        status=status,
        kb_name=kb_name,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )
    return JSONResponse(content={"success": True, "data": data})


@router.get("/stats")
async def service_ticket_stats(
    kb_name: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    data = service_ticket_service.ticket_stats(
        kb_name=kb_name,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )
    return JSONResponse(content={"success": True, "data": data})


@router.get("/{ticket_id}")
async def get_service_ticket(ticket_id: str):
    data = service_ticket_service.get_ticket(ticket_id)
    return JSONResponse(content={"success": True, "data": data})


@router.patch("/{ticket_id}")
async def update_service_ticket(ticket_id: str, body: UpdateTicketRequest):
    data = service_ticket_service.update_ticket(
        ticket_id,
        status=body.status,
        answer=body.answer,
        note=body.note,
    )
    return JSONResponse(content={"success": True, "data": data})


@router.delete("/{ticket_id}")
async def delete_service_ticket(ticket_id: str):
    data = service_ticket_service.delete_ticket(ticket_id)
    return JSONResponse(content={"success": True, "data": data})


@router.put("/{ticket_id}/chunks/{chunk_id}")
async def update_ticket_chunk(ticket_id: str, chunk_id: str, body: UpdateTicketChunkRequest):
    data = service_ticket_service.update_original_chunk_from_ticket(
        ticket_id,
        chunk_id,
        body.content,
    )
    return JSONResponse(content={"success": True, "data": data})


@router.post("/{ticket_id}/revectorize")
async def revectorize_ticket(ticket_id: str):
    data = await service_ticket_service.revectorize_ticket_context(ticket_id)
    return JSONResponse(content={"success": True, "data": data})
