# -*- coding: utf-8 -*-
"""
未回答问题解决服务
将管理员补充答案作为 synthetic chunk 写入知识库并归档
"""
import logging
from datetime import datetime
from typing import Optional

from app.core.exceptions import NotFoundError, ConflictError, ValidationError, ExternalServiceError
from app.db import (
    get_unanswered_repository,
    get_kb_repository,
    get_file_repository,
    get_job_repository,
    get_chunk_repository,
)
from app.services.oss_service import get_oss_service
from app.services import chunk_service, job_service

logger = logging.getLogger(__name__)


async def resolve_unanswered(
    unanswered_id: str,
    kb_name: str,
    manual_answer: str,
    image_file_content: Optional[bytes] = None,
    image_filename: Optional[str] = None,
) -> dict:
    """
    将管理员补充答案写入目标知识库，并归档未回答问题。

    流程：
      1. 验证 unanswered 记录存在且为 pending
      2. 查询目标知识库
      3. 创建 synthetic knowledge_file（文本存入 local storage）
      4. 创建 knowledge_job
      5. 写入 chunk（原始问题 + 管理员答案）
      6. 如有图片，注入占位符
      7. 向量化到 Milvus
      8. 更新 unanswered 状态为 resolved
    """
    # ── 1. 验证 unanswered 记录 ─────────────────────────────────────────────
    unanswered_repo = get_unanswered_repository()
    unanswered = unanswered_repo.get_by_id(unanswered_id)
    if not unanswered:
        raise NotFoundError(f"未回答问题不存在: {unanswered_id}")
    if unanswered.get("status") != "pending":
        raise ConflictError(f"该记录已处理，当前状态: {unanswered.get('status')}")

    if not manual_answer or len(manual_answer.strip()) < 10:
        raise ValidationError("补充答案不能为空，且长度至少 10 个字符")

    # ── 2. 查询目标知识库 ───────────────────────────────────────────────────
    kb = get_kb_repository().get_by_name(kb_name)
    if not kb:
        raise NotFoundError(f"知识库不存在: {kb_name}")
    kb_id = kb["id"]

    # ── 3. 创建 synthetic knowledge_file ────────────────────────────────────
    # 将管理员答案文本存入 file_storage，oss_key 作为文件标识
    safe_id = unanswered_id.replace("-", "_")
    oss_key = f"manual_answers/{kb_name}/unanswered_{safe_id}.txt"
    file_name = f"unanswered_{safe_id}.txt"

    text_content = manual_answer.strip().encode("utf-8")
    get_oss_service().upload_bytes(oss_key, text_content)

    file_record = get_file_repository().create(
        kb_id=kb_id,
        file_name=file_name,
        oss_key=oss_key,
        file_size=len(text_content),
        mime_type="text/plain",
        status="pending",
        sync_graph=False,
    )
    file_id = file_record["id"]

    # ── 4. 创建 knowledge_job ───────────────────────────────────────────────
    job_record = get_job_repository().create(file_id=file_id, kb_id=kb_id)
    job_id = job_record["id"]

    # 手动设置 job 状态为 chunked（跳过解析流程）
    get_job_repository().update_status(
        job_id, "chunked", stage="手动补充答案，等待向量化", progress=50, chunk_count=1
    )

    # ── 5. 写入 chunk（原始问题 + 管理员答案）───────────────────────────────
    query_text = unanswered.get("query", "")
    combined_content = f"Q: {query_text}\nA: {manual_answer.strip()}"

    chunks = [
        {
            "page_content": combined_content,
            "metadata": {
                "source": "manual_answer",
                "unanswered_id": unanswered_id,
                "original_query": query_text,
            },
        }
    ]
    get_chunk_repository().bulk_insert(job_id, file_name, chunks)

    # ── 6. 可选：注入图片占位符 ─────────────────────────────────────────────
    if image_file_content and image_filename:
        try:
            chunk_service.add_chunk_image(
                job_id=job_id,
                chunk_index=0,
                file_content=image_file_content,
                filename=image_filename,
                insert_position=0,
            )
            logger.info(f"[resolve] 图片注入成功: unanswered_id={unanswered_id}")
        except Exception as e:
            logger.warning(f"[resolve] 图片注入失败（继续向量化）: {e}")

    # ── 7. 向量化到 Milvus ──────────────────────────────────────────────────
    try:
        result = await job_service.upsert_job_to_milvus(job_id)
        upsert_count = result.get("upsert_count", 0)
        failed_batches = result.get("failed_batches", [])
        logger.info(f"[resolve] 向量化完成: unanswered_id={unanswered_id}, upsert={upsert_count}")
    except Exception as e:
        logger.error(f"[resolve] 向量化失败: unanswered_id={unanswered_id}, error={e}")
        # 向量化失败时，job 保持 chunked 状态，管理员可重试
        raise ExternalServiceError(f"向量化失败: {e}") from e

    # ── 8. 归档未回答问题 ───────────────────────────────────────────────────
    unanswered_repo.update_status(
        unanswered_id,
        status="resolved",
        resolved_at=datetime.now().isoformat(),
        resolved_kb_name=kb_name,
        resolved_job_id=job_id,
    )

    return {
        "unanswered_id": unanswered_id,
        "kb_name": kb_name,
        "job_id": job_id,
        "file_id": file_id,
        "upsert_count": upsert_count,
        "status": "resolved",
    }
