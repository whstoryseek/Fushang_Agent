from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Tuple

from app.core.config import settings
from app.services.llm_service import get_llm_service

RetrievalBucket = Literal["faq", "manual"]
_ALLOWED_BUCKETS = {"faq", "manual", "none"}
_ROUTER_MODEL = getattr(settings, "operation_classifier_model", "doubao-seed-2-0-mini-260428")
_ROUTER_MAX_TOKENS = 80
_ROUTER_TIMEOUT = 0.45

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


def _bucket_router_prompt(query: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": """
你是知识库检索路由判断器，只输出一个 JSON 对象，不要输出 Markdown。

你的任务是判断用户问题更应该优先检索哪一类知识：
- faq：故障现象、异常、报错、状态问题、结果不符合预期、原因排查、怎么办
- manual：操作步骤、配置方法、开通流程、准备资料、申请/提交流程
- none：无法判断或两边都可能，直接走全量检索

判断原则：
1. 这是检索路由，不是工单意图分类。
2. 只要问题核心是在描述“出了什么现象/什么结果不对/为什么没成功/怎么办”，优先 faq。
3. 只有问题核心明显是在问“怎么操作/怎么配置/需要什么资料/开通流程/步骤”，才判 manual。
4. 不要因为句子里出现“怎么”“怎么办”“如何”就机械判 manual，要看它是在问症状处理还是操作流程。
5. 拿不准就输出 none。

示例：
- “顾客能进直播，但是直播画面显示没有画面怎么办？” => faq
- “顾客在人康课堂已经提现但没有到微信怎么办？” => faq
- “店长端无法打开” => faq
- “开通微信子商户号需要准备哪些资料？” => manual
- “如何配置税率自动分摊？” => manual

输出固定字段：
bucket, reason

其中 bucket 只能是 faq、manual、none 之一。
""",
        },
        {"role": "user", "content": query or ""},
    ]


def _normalize_bucket(value: object) -> Optional[RetrievalBucket]:
    normalized = str(value or "").strip().lower()
    if normalized in {"faq", "manual"}:
        return normalized  # type: ignore[return-value]
    return None


def preferred_retrieval_bucket(query: str) -> Optional[RetrievalBucket]:
    normalized = (query or "").strip().lower()
    if not normalized:
        return None
    if len(normalized) <= 2:
        return None
    try:
        text = get_llm_service().responses_text(
            _bucket_router_prompt(query),
            model=_ROUTER_MODEL,
            temperature=0.0,
            max_tokens=_ROUTER_MAX_TOKENS,
            timeout=_ROUTER_TIMEOUT,
            max_retries=0,
        )
        import json

        payload = json.loads((text or "").strip())
        if not isinstance(payload, dict):
            return None
        bucket = str(payload.get("bucket") or "").strip().lower()
        if bucket == "none" or bucket not in _ALLOWED_BUCKETS:
            return None
        return _normalize_bucket(bucket)
    except Exception:
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
