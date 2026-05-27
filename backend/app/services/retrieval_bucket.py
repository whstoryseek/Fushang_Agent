from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Tuple

RetrievalBucket = Literal["faq", "manual"]

_FAQ_FILE_HINTS = (
    "faq",
    "qa",
    "q&a",
    "问答",
    "问题",
    "常见问题",
    "故障",
    "排障",
)

_MANUAL_FILE_HINTS = (
    "手册",
    "指南",
    "教程",
    "操作",
    "流程",
    "培训",
    "知识",
    "说明",
    "指引",
    "资料",
    "入驻",
    "开店",
)

_MANUAL_QUERY_HINTS = (
    "怎么",
    "如何",
    "步骤",
    "流程",
    "操作",
    "教程",
    "手册",
    "指南",
    "资料",
    "准备",
    "申请",
    "开通",
    "配置",
    "设置",
    "提交",
    "审核",
    "需要什么",
    "需要哪些",
)

_FAQ_QUERY_HINTS = (
    "无法",
    "打不开",
    "报错",
    "异常",
    "失败",
    "闪退",
    "黑屏",
    "卡住",
    "没反应",
    "为什么",
    "啥原因",
    "怎么回事",
)


def infer_retrieval_bucket(file_name: str) -> RetrievalBucket:
    normalized = (file_name or "").strip().lower()
    ext = normalized.rsplit(".", 1)[-1] if "." in normalized else ""

    if ext in {"xlsx", "xls"}:
        return "faq"

    if any(token in normalized for token in _FAQ_FILE_HINTS):
        return "faq"

    if any(token in normalized for token in _MANUAL_FILE_HINTS):
        return "manual"

    return "manual"


def preferred_retrieval_bucket(query: str) -> Optional[RetrievalBucket]:
    normalized = (query or "").strip().lower()
    if not normalized:
        return None

    if any(token in normalized for token in _MANUAL_QUERY_HINTS):
        return "manual"

    if any(token in normalized for token in _FAQ_QUERY_HINTS):
        return "faq"

    if len(normalized) <= 12:
        return None

    return None


def resolve_chunking_strategy(
    file_name: str,
    chunk_profile: Optional[str],
    requested_chunk_strategy: Optional[str],
) -> str:
    profile = (chunk_profile or "smart_mix").strip().lower()
    requested = (requested_chunk_strategy or "parent_child").strip().lower()

    if profile == "smart_mix":
        return "flat" if infer_retrieval_bucket(file_name) == "faq" else "parent_child"

    if profile in {"flat", "uniform_flat"}:
        return "flat"

    if profile in {"parent_child", "uniform_parent_child"}:
        return "parent_child"

    return requested or "parent_child"


def build_bucket_filter(bucket: str, base_filter_expr: Optional[str] = None) -> str:
    bucket_filter = f'retrieval_bucket == "{bucket}"'
    if base_filter_expr:
        return f"({base_filter_expr}) and {bucket_filter}"
    return bucket_filter


def _chunk_key(chunk: dict) -> str:
    return str(chunk.get("chunk_id") or chunk.get("id") or "")


def merge_bucket_hits(bucket_hits: List[dict], fallback_hits: List[dict], top_k: int) -> List[dict]:
    merged: List[dict] = []
    seen = set()

    for chunk in bucket_hits + fallback_hits:
        key = _chunk_key(chunk)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(chunk)
        if len(merged) >= top_k:
            break

    return merged


def search_with_bucket_fallback(
    *,
    query: str,
    top_k: int,
    search_fn: Callable[[Optional[str]], List[dict]],
    base_filter_expr: Optional[str] = None,
    min_bucket_hits: int = 3,
) -> Tuple[List[dict], Dict[str, object]]:
    preferred_bucket = preferred_retrieval_bucket(query)
    routing_log: Dict[str, object] = {
        "preferred_bucket": preferred_bucket,
        "bucket_fallback": False,
        "fallback_reason": None,
        "bucket_hits": 0,
        "bucket_filter_expr": None,
    }

    if not preferred_bucket:
        hits = search_fn(base_filter_expr)
        routing_log["merged_hits"] = len(hits)
        return hits, routing_log

    bucket_filter_expr = build_bucket_filter(preferred_bucket, base_filter_expr)
    routing_log["bucket_filter_expr"] = bucket_filter_expr

    try:
        bucket_hits = search_fn(bucket_filter_expr)
    except Exception as exc:
        full_hits = search_fn(base_filter_expr)
        routing_log["bucket_fallback"] = True
        routing_log["fallback_reason"] = "bucket_filter_error"
        routing_log["bucket_error"] = str(exc)
        routing_log["merged_hits"] = len(full_hits)
        return full_hits, routing_log

    routing_log["bucket_hits"] = len(bucket_hits)
    required_hits = min(max(min_bucket_hits, 1), max(top_k, 1))
    if len(bucket_hits) >= required_hits or len(bucket_hits) >= top_k:
        selected_hits = bucket_hits[:top_k]
        routing_log["merged_hits"] = len(selected_hits)
        return selected_hits, routing_log

    full_hits = search_fn(base_filter_expr)
    merged_hits = merge_bucket_hits(bucket_hits, full_hits, top_k)
    routing_log["bucket_fallback"] = True
    routing_log["fallback_reason"] = "insufficient_bucket_hits"
    routing_log["merged_hits"] = len(merged_hits)
    return merged_hits, routing_log
