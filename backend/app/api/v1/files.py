# -*- coding: utf-8 -*-
"""本地文件存储代理路由（替代 OSS 预签名 URL）"""
import logging
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from app.api.v1.auth_deps import require_admin
from app.db import get_file_storage_repository
from app.services import file_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/files", tags=["files"])
admin_only = [Depends(require_admin)]


class DeleteFileRequest(BaseModel):
    file_id: str


class BatchDeleteFilesRequest(BaseModel):
    file_ids: list[str]
    kb_name: str


def _guess_mime(file_key: str) -> str:
    ext = file_key.lower().rsplit(".", 1)[-1] if "." in file_key else ""
    mime_map = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc": "application/msword",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel",
        "txt": "text/plain",
        "md": "text/markdown",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    return mime_map.get(ext, "application/octet-stream")


@router.get("", dependencies=admin_only)
async def list_files(
    kb_name: str = Query(..., description="知识库名称"),
    limit: int = Query(200, ge=1, le=2000),
):
    result = file_service.list_files(kb_name, limit=limit)
    return {"success": True, "data": result}


@router.delete("", dependencies=admin_only)
async def delete_file(body: DeleteFileRequest):
    file_name = file_service.delete_file(body.file_id)
    return {"success": True, "message": f"文件「{file_name}」已删除"}


@router.post("/batch-delete", dependencies=admin_only)
async def batch_delete_files(body: BatchDeleteFilesRequest):
    result = file_service.batch_delete_files(body.file_ids, body.kb_name)
    return {
        "success": True,
        "message": f"已删除 {len(result['deleted'])} 个文件",
        "data": result,
    }


@router.get("/{file_key:path}")
async def serve_file(file_key: str):
    """通过 file_key 读取本地存储的文件内容并返回"""
    try:
        repo = get_file_storage_repository()
        record = repo.get_by_key(file_key)
        if not record:
            return Response(status_code=404, content="文件不存在")

        # 优先使用数据库记录的 mime_type，否则根据扩展名猜测
        mime_type = record.get("mime_type") or _guess_mime(file_key)
        content = repo.get_bytes(file_key)
        if content is None:
            return Response(status_code=404, content="文件内容为空")

        return Response(
            content=content,
            media_type=mime_type,
            headers={
                "Cache-Control": "public, max-age=86400",
                "Content-Length": str(len(content)),
            },
        )
    except Exception as e:
        logger.error(f"本地文件读取失败: {file_key}, error={e}")
        return Response(status_code=500, content=str(e))
