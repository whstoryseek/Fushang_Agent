# -*- coding: utf-8 -*-
"""Build a smart-mix shadow KB and benchmark it against live `fushang`.

This is intentionally a manual benchmark script. It touches the live local
Postgres, Milvus, storage bytes, embeddings, and configured LLM provider.
"""
import asyncio
import json
import re
import time
import uuid
from collections import Counter
from io import BytesIO
from pathlib import Path

import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.db import (
    get_chunk_repository,
    get_file_repository,
    get_job_repository,
    get_kb_repository,
)
from app.services.document_service import search_documents
from app.services.job_service import run_job_pipeline, upsert_job_to_milvus
from app.services.embedding_service import get_embedding_service
from app.services.knowledge_service import invoke_knowledge_qa
from app.services.milvus_service import get_milvus_service
from app.services.oss_service import get_oss_service

LIVE_KB = "fushang"
SHADOW_PREFIX = "fushang_mix_eval"

CHUNK_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 80,
    "image_dpi": 150,
    "parent_chunk_size": 1800,
    "child_chunk_size": 500,
    "chunk_strategy": "parent_child",
    "chunk_profile": "smart_mix",
    "excel_rows_per_chunk": 1,
}

MANUAL_QUERIES = [
    {"query": "开通微信子商户号需要准备哪些资料？", "expected_file": "开通微信子商户号.docx"},
    {"query": "直播订单无法退款怎么处理？", "expected_file": "近期常见问题.docx"},
    {"query": "顾客提现失败或不到账怎么办？", "expected_file": "常见问题及对应答案汇总.docx"},
    {"query": "店长怎么开通收款账户？", "expected_file": "简版店长开通收款账户教程.docx"},
    {"query": "直播间订单店长端怎么确认提货和退款？", "expected_file": "直播间订单店长操作手册.docx"},
    {"query": "人康课堂 app 怎么下载？", "expected_file": "人康课堂app下载及常见问题.docx"},
    {"query": "直播间订单称重退差怎么操作？", "expected_file": "直播间订单称重退差操作手册.docx"},
    {"query": "人康直播电脑端和手机端怎么使用？", "expected_file": "人康直播操作使用文档.docx"},
    {"query": "加盟商取消白名单后账户怎么迁移？", "expected_file": "加盟商要货取消白名单培训.docx"},
    {"query": "招商流程梳理和审批软件操作看哪份文档？", "expected_file": "2026招扶商版常见审批与软件操作问题汇总.docx"},
]

FULL_ROUND_QUERIES = [
    "店长端无法打开",
    "开通微信子商户号需要准备哪些资料？",
]

_IMAGE_PH_RE = re.compile(r"<<IMAGE:[0-9a-f]+>>")


def _load_live_context():
    kb_repo = get_kb_repository()
    file_repo = get_file_repository()
    kb = kb_repo.get_by_name(LIVE_KB)
    if not kb:
        raise RuntimeError(f"KB not found: {LIVE_KB}")
    files = file_repo.list_by_kb(kb["id"])
    return kb, files


def _load_faq_queries(live_files: list[dict]) -> list[dict]:
    file_map = {f["file_name"]: f for f in live_files}
    excel_file = file_map["tickets_kb_qa_table_merged.xlsx"]
    excel_bytes = get_oss_service().get_object_bytes(excel_file["oss_key"])
    faq_df = pd.read_excel(BytesIO(excel_bytes), sheet_name=0)
    return [
        {"query": str(row["问题"]).strip(), "expected_file": "tickets_kb_qa_table_merged.xlsx"}
        for _, row in faq_df.head(10).iterrows()
        if str(row["问题"]).strip()
    ]


async def _build_shadow_kb(live_kb: dict, live_files: list[dict]) -> tuple[dict, list[dict], float]:
    kb_repo = get_kb_repository()
    file_repo = get_file_repository()
    job_repo = get_job_repository()

    shadow_name = f"{SHADOW_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}"
    shadow_kb = kb_repo.create(
        name=shadow_name,
        display_name=shadow_name,
        description="smart_mix benchmark shadow kb",
        image_mode=live_kb["image_mode"],
        kb_type=live_kb["kb_type"],
        embedding_model=live_kb["embedding_model"],
        vector_dim=live_kb["vector_dim"],
        metadata_fields=live_kb["metadata_fields"],
        retrieval_config=live_kb["retrieval_config"],
    )

    build_rows = []
    started = time.perf_counter()
    for src in reversed(live_files):
        file_record = file_repo.create(
            kb_id=shadow_kb["id"],
            file_name=src["file_name"],
            oss_key=src["oss_key"],
            category_file_id=src.get("category_file_id"),
            file_size=src.get("file_size"),
            mime_type=src.get("mime_type"),
            status="pending",
            sync_graph=False,
        )
        job = job_repo.create(file_id=file_record["id"], kb_id=shadow_kb["id"])

        t0 = time.perf_counter()
        await run_job_pipeline(
            job_id=job["id"],
            file_id=file_record["id"],
            kb_id=shadow_kb["id"],
            kb_name=shadow_kb["name"],
            file_name=src["file_name"],
            oss_key=src["oss_key"],
            image_mode=shadow_kb["image_mode"] and not src["file_name"].lower().endswith((".xlsx", ".xls")),
            **CHUNK_CONFIG,
        )
        t1 = time.perf_counter()
        fallback_used = False
        try:
            upsert_result = await upsert_job_to_milvus(job["id"])
        except Exception:
            upsert_result = _fallback_text_upsert(shadow_kb, job["id"], src["file_name"])
            fallback_used = True
        else:
            if upsert_result.get("upsert_count", 0) == 0:
                upsert_result = _fallback_text_upsert(shadow_kb, job["id"], src["file_name"])
                fallback_used = True
        t2 = time.perf_counter()

        build_rows.append(
            {
                "file_name": src["file_name"],
                "job_id": job["id"],
                "chunk_ms": round((t1 - t0) * 1000, 2),
                "upsert_ms": round((t2 - t1) * 1000, 2),
                "upsert_count": upsert_result.get("upsert_count", 0),
                "fallback_used": fallback_used,
            }
        )

    total_ms = round((time.perf_counter() - started) * 1000, 2)
    return shadow_kb, build_rows, total_ms


def _fallback_text_upsert(shadow_kb: dict, job_id: str, file_name: str) -> dict:
    chunk_repo = get_chunk_repository()
    milvus_svc = get_milvus_service()
    embedding_svc = get_embedding_service()
    rc = shadow_kb.get("retrieval_config") or {}
    image_dim = shadow_kb.get("vector_dim", 1024)

    milvus_svc.get_or_create_collection(
        shadow_kb["name"],
        dim=image_dim,
        kb_type=shadow_kb.get("kb_type", "standard"),
        image_vector_dim=rc.get("image_vector_dim", image_dim),
    )

    pg_chunks = chunk_repo.get_by_job(job_id)
    if not pg_chunks:
        return {"upsert_count": 0}

    texts = []
    rows = []
    for chunk in pg_chunks:
        clean = _IMAGE_PH_RE.sub("", chunk.get("current_content") or "").strip()
        if not clean:
            continue
        texts.append(clean)
        rows.append(chunk)

    if not texts:
        return {"upsert_count": 0}

    vectors = embedding_svc.embed_texts(texts, dimension=image_dim)
    data = []
    for chunk, vec, clean in zip(rows, vectors, texts):
        row = {
            "chunk_id": chunk["chunk_id"],
            "job_id": job_id,
            "file_name": file_name,
            "chunk_index": int(chunk.get("chunk_index", 0)),
            "content": clean,
            "dense": vec,
            "image_dense": [0.0] * image_dim,
        }
        for key, value in (chunk.get("metadata") or {}).items():
            if key not in row:
                row[key] = value
        data.append(row)

    total = 0
    for start in range(0, len(data), 50):
        batch = data[start:start + 50]
        res = milvus_svc.client.upsert(collection_name=shadow_kb["name"], data=batch)
        total += res.get("upsert_count", len(batch))
    return {"upsert_count": total, "fallback_mode": "text_only_multimodal"}


def _summarize_chunks(kb_id: str) -> dict:
    chunk_repo = get_chunk_repository()
    job_repo = get_job_repository()

    jobs = job_repo.list_by_kb(kb_id, limit=500)
    chunks = []
    for job in jobs:
        chunks.extend(chunk_repo.get_by_job(job["id"]))

    strategy_counter = Counter()
    bucket_counter = Counter()
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        strategy_counter[meta.get("chunk_strategy", "unknown")] += 1
        bucket_counter[meta.get("retrieval_bucket", "unknown")] += 1

    return {
        "job_count": len(jobs),
        "chunk_count": len(chunks),
        "chunk_strategy": dict(strategy_counter),
        "retrieval_bucket": dict(bucket_counter),
    }


def _run_search_benchmark(live_kb: dict, kb_name: str, queries: list[dict]) -> tuple[list[dict], dict]:
    rc = live_kb.get("retrieval_config") or {}
    rows = []
    for item in queries:
        started = time.perf_counter()
        hits = search_documents(
            query=item["query"],
            kb_name=kb_name,
            top_k=10,
            ranker=rc.get("ranker", "RRF"),
            hybrid_alpha=rc.get("hybrid_alpha", 0.5),
            keyword_filter=None,
            rerank=False,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        hit_files = [hit.get("file_name") for hit in hits]
        rows.append(
            {
                "query": item["query"],
                "expected_file": item["expected_file"],
                "top1_hit": bool(hit_files[:1] and hit_files[0] == item["expected_file"]),
                "top3_hit": item["expected_file"] in hit_files[:3],
                "selected_hits": len(hits),
                "context_chars": sum(len(hit.get("content") or "") for hit in hits),
                "search_ms": elapsed_ms,
                "top3_files": hit_files[:3],
            }
        )

    summary = {
        "top1": round(sum(row["top1_hit"] for row in rows) / len(rows), 4),
        "top3": round(sum(row["top3_hit"] for row in rows) / len(rows), 4),
        "avg_selected_hits": round(sum(row["selected_hits"] for row in rows) / len(rows), 2),
        "avg_context_chars": round(sum(row["context_chars"] for row in rows) / len(rows), 2),
        "avg_search_ms": round(sum(row["search_ms"] for row in rows) / len(rows), 2),
    }
    return rows, summary


async def _run_full_round_latency(kb_name: str) -> list[dict]:
    rows = []
    for query in FULL_ROUND_QUERIES:
        session_id = f"bench-{kb_name}-{uuid.uuid4().hex[:8]}"
        started = time.perf_counter()
        result = await invoke_knowledge_qa(
            query=query,
            model_name="doubao-seed-2-0-pro-260215",
            session_id=session_id,
            collection=kb_name,
            persist=False,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        rows.append(
            {
                "query": query,
                "latency_ms": elapsed_ms,
                "sources": [src.get("file_name") for src in (result.get("sources") or [])[:3]],
                "answer_preview": (result.get("answer") or "")[:120],
            }
        )
    return rows


async def main() -> int:
    live_kb, live_files = _load_live_context()
    faq_queries = _load_faq_queries(live_files)
    shadow_kb, build_rows, build_total_ms = await _build_shadow_kb(live_kb, live_files)

    live_doc_rows, live_doc_summary = _run_search_benchmark(live_kb, LIVE_KB, MANUAL_QUERIES)
    shadow_doc_rows, shadow_doc_summary = _run_search_benchmark(live_kb, shadow_kb["name"], MANUAL_QUERIES)
    live_faq_rows, live_faq_summary = _run_search_benchmark(live_kb, LIVE_KB, faq_queries)
    shadow_faq_rows, shadow_faq_summary = _run_search_benchmark(live_kb, shadow_kb["name"], faq_queries)

    live_round = await _run_full_round_latency(LIVE_KB)
    shadow_round = await _run_full_round_latency(shadow_kb["name"])

    output = {
        "shadow_kb": shadow_kb["name"],
        "chunk_config": CHUNK_CONFIG,
        "build_total_ms": build_total_ms,
        "build_rows": build_rows,
        "live_chunk_stats": _summarize_chunks(live_kb["id"]),
        "shadow_chunk_stats": _summarize_chunks(shadow_kb["id"]),
        "doc_query_benchmark": {
            "live": live_doc_summary,
            "shadow": shadow_doc_summary,
            "live_rows": live_doc_rows,
            "shadow_rows": shadow_doc_rows,
        },
        "faq_query_benchmark": {
            "live": live_faq_summary,
            "shadow": shadow_faq_summary,
            "live_rows": live_faq_rows,
            "shadow_rows": shadow_faq_rows,
        },
        "full_round_latency": {
            "live": live_round,
            "shadow": shadow_round,
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
