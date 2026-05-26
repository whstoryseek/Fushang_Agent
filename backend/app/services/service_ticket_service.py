# -*- coding: utf-8 -*-
"""服务记录 / 工单业务逻辑。"""
from __future__ import annotations

import re
import json
import logging
from typing import Any, Dict, List, Optional

from app.core.exceptions import NotFoundError, ValidationError
from app.core.config import settings
from app.db import (
    get_chunk_repository,
    get_job_repository,
    get_service_ticket_repository,
)
from app.services import chunk_service
from app.services.llm_service import get_llm_service


logger = logging.getLogger(__name__)
MANUAL_STATUSES = {"pending_manual", "in_progress"}
FINAL_STATUSES = {"resolved_ai", "resolved_manual", "ignored"}
ALL_TICKET_STATUSES = {
    "clarifying",
    "resolved_ai",
    "pending_manual",
    "in_progress",
    "resolved_manual",
    "ignored",
}

INTENT_OPERATION = "A"
INTENT_MISSING_KNOWLEDGE = "B"
INTENT_AMBIGUOUS = "C"

MAX_OPERATION_CLARIFICATION_ROUNDS = 20
MAX_MISSING_KNOWLEDGE_ROUNDS = 5
MAX_AMBIGUOUS_ROUNDS = 3

EXIT_FIELDS_COMPLETE = "fields_complete"
EXIT_NO_STANDARD_WORKFLOW = "no_standard_workflow"
EXIT_MAX_ROUNDS = "max_rounds"
EXIT_SEMANTIC_UNRESOLVED = "semantic_unresolved"
EXIT_USER_REFUSED = "user_refused_or_unknown"
EXIT_RAG_HIT = "rag_hit_after_clarification"

COMPLETION_COMPLETE = "complete"
COMPLETION_INCOMPLETE = "incomplete"

_AMBIGUOUS_SHORT_PATTERNS = (
    re.compile(r"^(怎么|咋|如何).{0,4}(处理|弄|办|解决)?[？?]?$"),
    re.compile(r"^(这个|那个|这|那|它|他|她).{0,8}(怎么|如何|啥|什么)"),
    re.compile(r"^(可以吗|能不能|怎么办|咋办)[？?]?$"),
)

_OPERATION_PATTERNS = (
    re.compile(r"(帮我|给我|麻烦|请|需要|我要|帮忙|协助).{0,16}(开通|绑定|解绑|处理|办理|修改|重置|冻结|解封|提现|实名|认证|操作|查一下|弄一下|弄下|搞一下)"),
    re.compile(r"(开通|关闭|绑定|解绑|重置|冻结|解封|提现).{0,8}(权限|账户|账号|直播|富友|手机号|密码|收款)"),
    re.compile(r"^(操作|处理).{0,8}(账户|账号|权限|直播|富友|收款)"),
    re.compile(r"(申请|处理|办理).{0,8}(权限|账户|账号|直播|认证|富友|收款)"),
    re.compile(r"(账户|账号|后台|权限|直播|富友|收款).{0,12}(报错|异常|失败|登不上|不能用|打不开|处理一下|弄一下)"),
)

OPERATION_CLARIFICATION_MESSAGE = "这类操作需要先补齐工单信息。请说明要处理的账号/门店、具体权限或页面，并尽量补充截图、手机号和实名/主体信息。"
AMBIGUOUS_CLARIFICATION_MESSAGE = "请补充一下具体对象或场景，我再帮你查询。例如：是哪项政策、哪个系统、哪一步操作遇到问题？"

_DELEGATE_WORD_RE = re.compile(
    r"(帮我|给我|替我|麻烦(你|帮我|给我)?|协助|帮忙|安排一下|弄一下|弄下|处理一下|办一下|查一下|搞一下|"
    r"请帮我|请给我|请处理|请开通|请关闭|我要|我想要|我需要|需要你)"
)
_BARE_OPERATION_COMMAND_RE = re.compile(
    r"^(开通|关闭|绑定|解绑|重置|冻结|解封|提现|实名|认证|申请).{0,12}(权限|账号|账户|收款|直播|富友|微信)"
)
_KNOWLEDGE_QUESTION_RE = re.compile(
    r"(怎么办|怎么处理|如何处理|如何申诉|怎么申诉|什么原因|为什么|怎么回事|处理办法|解决办法|要怎么办|"
    r"怎么|如何|咋办|能否|能不能|可不可以|是否可以|是否能|有没有办法|可以.{0,16}吗|能.{0,16}吗|"
    r"要不要|需要.{0,16}吗|有什么条件|什么条件|有什么要求|什么要求|需要什么|准备什么|流程|步骤|教程)"
)

PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
MIN_CLARIFICATION_ROUNDS = 3
MAX_CLARIFICATION_ROUNDS = 5


def _has_question_form(text: str) -> bool:
    if not text:
        return False
    return bool(_KNOWLEDGE_QUESTION_RE.search(text) or re.search(r"(吗|么|呢|？|\?)$", text))


def _has_delegate_operation_intent(text: str) -> bool:
    if not text:
        return False
    if _DELEGATE_WORD_RE.search(text):
        return True
    return bool(_BARE_OPERATION_COMMAND_RE.search(text) and not _has_question_form(text))


def _is_knowledge_help_question(text: str) -> bool:
    if not text:
        return False
    return _has_question_form(text) and not _has_delegate_operation_intent(text)


_REFUSAL_RE = re.compile(
    r"(不想回答|不用问|别问|不知道|不清楚|无法提供|没法提供|没有更多|没有其他|直接处理|直接提交|你们处理|人工处理)"
)
_REFUSAL_STOP_RE = re.compile(r"(不想回答|不用问|别问|直接处理|直接提交|你们处理|人工处理)")
_USEFUL_INFO_RE = re.compile(
    r"(手机号|手机|电话|门店|店铺|账号|账户|截图|图片|已传|上传|身份证|证件|138\d{8}|1[3-9]\d{9})"
)


def _is_user_refusal(query: str) -> bool:
    text = " ".join((query or "").strip().split())
    if not text or not _REFUSAL_RE.search(text):
        return False
    if _REFUSAL_STOP_RE.search(text):
        return True
    if _USEFUL_INFO_RE.search(text):
        return False
    return True


def _intent_class_for_reason(reason: str) -> str:
    if reason == "knowledge_missing":
        return INTENT_MISSING_KNOWLEDGE
    if reason == "ambiguous_query":
        return INTENT_AMBIGUOUS
    return INTENT_OPERATION


def _max_rounds_for_intent(intent_class: str) -> int:
    if intent_class == INTENT_AMBIGUOUS:
        return MAX_AMBIGUOUS_ROUNDS
    if intent_class == INTENT_MISSING_KNOWLEDGE:
        return MAX_MISSING_KNOWLEDGE_ROUNDS
    return MAX_OPERATION_CLARIFICATION_ROUNDS


def _is_complete(intent_class: str, missing: List[str]) -> bool:
    return intent_class == INTENT_OPERATION and not missing


def _exit_reason_for_round_cap(intent_class: str) -> str:
    if intent_class == INTENT_AMBIGUOUS:
        return EXIT_SEMANTIC_UNRESOLVED
    return EXIT_MAX_ROUNDS


def should_start_missing_knowledge_flow(rag_result: Dict[str, Any]) -> bool:
    if not isinstance(rag_result, dict):
        return False
    if not rag_result.get("used_fallback"):
        return False
    fallback_reason = str(rag_result.get("fallback_reason") or "").lower()
    sources = rag_result.get("sources") or []
    confidence = rag_result.get("confidence")
    no_relevant_reason = any(
        marker in fallback_reason
        for marker in [
            "no_relevant",
            "no relevant",
            "knowledge base",
            "未找到",
            "未检索",
            "没有找到",
            "无相关",
        ]
    )
    if no_relevant_reason or rag_result.get("quality_passed") is False:
        return True
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return False
    if confidence_value < 0.35 and not sources:
        return True
    return False


def _is_clarification_followup(
    *,
    query: str,
    has_image: bool = False,
    query_image_oss_key: Optional[str] = None,
) -> bool:
    text = " ".join((query or "").strip().split())
    if has_image or query_image_oss_key:
        return True
    if PHONE_RE.search(text) or ID_CARD_RE.search(text):
        return True
    if _is_knowledge_help_question(text):
        return False
    return bool(
        re.search(r"(补充|手机号|电话|身份证|证件|主体|门店|账号|账户|截图|图片|报错|页面|权限|富友|收款码)", text)
        or (len(text) <= 18 and not re.search(r"(怎么|如何|什么|为什么|怎么办|吗|？|\?)", text))
    )


def should_clarify_query(query: str, *, has_image: bool = False) -> Dict[str, Any]:
    text = " ".join((query or "").strip().split())
    if not text:
        return {"should_clarify": False, "message": "", "reason": ""}
    if len(text) <= 4 or any(p.search(text) for p in _AMBIGUOUS_SHORT_PATTERNS):
        return {
            "should_clarify": True,
            "reason": "ambiguous_query",
            "message": AMBIGUOUS_CLARIFICATION_MESSAGE,
        }
    if _is_knowledge_help_question(text):
        return {"should_clarify": False, "message": "", "reason": "knowledge_question"}
    if any(p.search(text) for p in _OPERATION_PATTERNS):
        return {
            "should_clarify": True,
            "reason": "operation_required",
            "message": OPERATION_CLARIFICATION_MESSAGE,
        }
    return {"should_clarify": False, "message": "", "reason": ""}


def _extract_json_object(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    parsed = json.loads(text)
    return parsed if isinstance(parsed, dict) else {}


def classify_operation_query_with_llm(query: str, *, has_image: bool = False) -> Dict[str, Any]:
    text = " ".join((query or "").strip().split())
    if not text:
        return {"should_clarify": False, "reason": "", "message": "", "source": "llm_skipped"}

    system_prompt = (
        "你是扶商店长助手的客服工单分流器。判断店长原话是否必须进入澄清/人工工单，而不是知识库问答。"
        "只输出 JSON，不要输出解释。"
        "硬性规则：只有用户明确要求我们代办、执行、处理某个具体账号/员工/门店/权限时，才返回 should_clarify=true。"
        "true 示例：“帮我绑定一下富友账户”“帮我把富友收款弄一下”“给我开通直播权限”“帮我关闭这个员工的直播权限”“帮我解封这个微信号”“这个账号实名处理一下”。"
        "硬性规则：凡是用户在问能不能、可不可以、是否可以、怎么做、如何操作、有什么条件、需要什么材料、为什么、怎么办、在哪里查看，必须返回 should_clarify=false，先走知识库问答。"
        "false 示例：“店面之前开通的员工的直播权限可以关闭吗”“直播权限可以关闭吗”“怎么关闭直播权限”“富友账户怎么绑定”“开通直播权限有什么条件”“在哪里看收款记录”“微信出现封禁的情况要怎么办”“微信封禁怎么申诉”。"
        "如果同时有问法和代办词，以是否明确让我们替他执行为准；不确定时返回 false。"
    )
    user_prompt = (
        "请分类这条店长问题：\n"
        f"{text}\n\n"
        "输出格式："
        '{"should_clarify": true/false, "reason": "operation_required|knowledge_question|uncertain", '
        '"message": "给用户的简短追问信息", "confidence": 0.0}'
    )

    try:
        raw = get_llm_service().responses_text(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{system_prompt}\n\n{user_prompt}",
                        }
                    ],
                }
            ],
            model=settings.operation_classifier_model,
            temperature=0.0,
            max_tokens=300,
            timeout=settings.operation_classifier_timeout,
            max_retries=0,
        )
        parsed = _extract_json_object(raw)
        should_clarify = bool(parsed.get("should_clarify"))
        reason = str(parsed.get("reason") or ("operation_required" if should_clarify else "knowledge_question")).strip()
        try:
            confidence = float(parsed.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0
        if should_clarify and _is_knowledge_help_question(text):
            return {
                "should_clarify": False,
                "reason": "knowledge_question",
                "message": "",
                "confidence": confidence,
                "source": "llm_guard",
            }
        if should_clarify and reason not in {"operation_required", "ambiguous_query"}:
            reason = "operation_required"
        message = str(parsed.get("message") or "").strip()
        if should_clarify and not message:
            message = OPERATION_CLARIFICATION_MESSAGE
        raw_confidence = parsed.get("confidence", 0.0)
        if isinstance(raw_confidence, str) and raw_confidence.strip() in {"高", "high"}:
            confidence = 0.9
        elif isinstance(raw_confidence, str) and raw_confidence.strip() in {"中", "medium"}:
            confidence = 0.6
        elif isinstance(raw_confidence, str) and raw_confidence.strip() in {"低", "low"}:
            confidence = 0.3
        else:
            try:
                confidence = float(raw_confidence or 0.0)
            except Exception:
                confidence = 0.0
        return {
            "should_clarify": should_clarify,
            "reason": reason,
            "message": message,
            "confidence": confidence,
            "source": "llm",
        }
    except Exception as exc:
        logger.warning("操作类问题 LLM 判断失败，按普通知识问答继续: %s", exc)
        return {
            "should_clarify": False,
            "reason": "llm_unavailable",
            "message": "",
            "confidence": 0.0,
            "source": "llm_error",
        }


def _normalize_required_field(field: str, label: str = "") -> Optional[str]:
    text = f"{field} {label}".lower()
    if any(key in text for key in ["phone", "mobile", "手机号", "电话", "联系方式"]):
        return "phone"
    if any(key in text for key in ["image", "screenshot", "截图", "图片", "照片"]):
        return "image"
    if any(key in text for key in ["id_card", "identity", "身份证", "证件", "实名", "主体"]):
        return "id_card"
    if any(key in text for key in ["issue", "account", "store", "shop", "门店", "账号", "账户", "富友", "问题", "页面", "权限", "现象"]):
        return "issue_detail"
    return None


def extract_operation_workflow_requirements(query: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not sources:
        return {
            "workflow_found": False,
            "workflow_summary": "",
            "required_fields": [],
            "field_labels": {},
            "fallback_used": True,
        }

    context_lines = []
    for idx, src in enumerate(sources[:8], start=1):
        content = " ".join((src.get("content") or "").split())
        if not content:
            continue
        file_name = src.get("file_name") or src.get("title") or (src.get("metadata") or {}).get("file_name") or "unknown"
        context_lines.append(f"[{idx}] {file_name}: {content[:900]}")
    if not context_lines:
        return {
            "workflow_found": False,
            "workflow_summary": "",
            "required_fields": [],
            "field_labels": {},
            "fallback_used": True,
        }

    prompt = (
        "你是扶商店长助手的工单流程抽取器。根据知识库召回内容判断是否有与用户操作请求相关的现成流程，"
        "并抽取后台处理前必须向用户收集的信息。只输出 JSON。\n"
        f"用户问题：{query}\n\n"
        "知识库召回：\n"
        + "\n".join(context_lines)
        + "\n\n输出 JSON 字段："
        '{"workflow_found": true/false, "workflow_summary": "一句话流程摘要", '
        '"required_fields": ["issue_detail|phone|image|id_card"], '
        '"field_labels": {"phone": "可联系手机号"}, "fallback_used": false}'
        "\n如果召回内容只是泛泛无关内容或没有明确流程，workflow_found=false。"
    )

    try:
        raw = get_llm_service().responses_text(
            [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            model=settings.operation_classifier_model,
            temperature=0.0,
            max_tokens=500,
            timeout=settings.operation_classifier_timeout,
            max_retries=0,
        )
        parsed = _extract_json_object(raw)
    except Exception as exc:
        logger.warning("操作流程抽取失败，使用兜底澄清: %s", exc)
        return {
            "workflow_found": False,
            "workflow_summary": "",
            "required_fields": [],
            "field_labels": {},
            "fallback_used": True,
        }

    raw_labels = parsed.get("field_labels") if isinstance(parsed.get("field_labels"), dict) else {}
    required_fields: List[str] = []
    field_labels: Dict[str, str] = {}
    for raw_field in parsed.get("required_fields") or []:
        raw_field = str(raw_field or "").strip()
        raw_label = str(raw_labels.get(raw_field) or "").strip()
        field = _normalize_required_field(raw_field, raw_label)
        if not field or field in required_fields:
            continue
        required_fields.append(field)
        if raw_label:
            field_labels[field] = raw_label

    workflow_found = bool(parsed.get("workflow_found")) and bool(required_fields)
    return {
        "workflow_found": workflow_found,
        "workflow_summary": str(parsed.get("workflow_summary") or "").strip()[:240],
        "required_fields": required_fields if workflow_found else [],
        "field_labels": field_labels if workflow_found else {},
        "fallback_used": not workflow_found,
    }


def _is_generic_short_text(text: str) -> bool:
    return len(text) <= 4 or any(p.search(text) for p in _AMBIGUOUS_SHORT_PATTERNS)


def _required_clarification_fields(reason: str, original_query: str) -> List[str]:
    text = original_query or ""
    if reason == "operation_required":
        fields = ["issue_detail", "phone", "image"]
        if re.search(r"(账户|账号|实名|认证|直播|权限|身份证|主体)", text):
            fields.append("id_card")
        return fields
    if reason == "knowledge_missing":
        return ["issue_detail", "phone"]
    return ["issue_detail"]


def _normalize_workflow_payload(workflow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    workflow = dict(workflow or {})
    workflow_found = bool(workflow.get("workflow_found"))
    workflow_summary = str(workflow.get("workflow_summary") or "").strip()
    if workflow_found and not workflow_summary:
        workflow_summary = "知识库中有相关处理流程/要求"
    required_fields = []
    for field in workflow.get("required_fields") or []:
        field = str(field or "").strip()
        if field and field not in required_fields:
            required_fields.append(field)
    field_labels = workflow.get("field_labels") or {}
    if not isinstance(field_labels, dict):
        field_labels = {}
    return {
        "workflow_found": workflow_found,
        "workflow_summary": workflow_summary,
        "required_fields": required_fields,
        "field_labels": {str(k): str(v) for k, v in field_labels.items() if k and v},
        "fallback_used": bool(workflow.get("fallback_used", not workflow.get("workflow_found"))),
    }


def _workflow_sources_summary(sources: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    result = []
    for src in sources or []:
        result.append(
            {
                "chunk_id": src.get("chunk_id") or src.get("id"),
                "job_id": src.get("job_id") or (src.get("metadata") or {}).get("job_id"),
                "file_name": src.get("file_name") or src.get("title") or (src.get("metadata") or {}).get("file_name"),
                "chunk_index": src.get("chunk_index") if src.get("chunk_index") is not None else (src.get("metadata") or {}).get("chunk_index"),
                "score": src.get("score"),
            }
        )
    return result


def _merge_collected_info(
    existing: Dict[str, Any],
    *,
    query: str,
    has_image: bool = False,
    query_image_oss_key: Optional[str] = None,
    requester_name: Optional[str] = None,
    sender_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    collected = dict(existing or {})
    text = " ".join((query or "").strip().split())
    phone = PHONE_RE.search(text)
    if phone:
        collected["phone"] = phone.group(0)
    id_card = ID_CARD_RE.search(text)
    if id_card:
        collected["id_card"] = id_card.group(0).upper()
    if text and not _is_generic_short_text(text):
        details = list(collected.get("issue_details") or [])
        if text not in details:
            details.append(text)
        collected["issue_details"] = details[-5:]
        collected["issue_detail"] = details[-1]
    if has_image or query_image_oss_key:
        image_keys = list(collected.get("image_keys") or [])
        if query_image_oss_key and query_image_oss_key not in image_keys:
            image_keys.append(query_image_oss_key)
        collected["image_keys"] = image_keys
        collected["has_image"] = True
    if requester_name:
        collected["requester_name"] = requester_name
    if sender_id:
        collected["sender_id"] = sender_id
    if user_id:
        collected["user_id"] = user_id
    return collected


def _missing_fields(required: List[str], collected: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for field in required:
        if field == "image":
            if not collected.get("has_image") and not collected.get("image_keys"):
                missing.append(field)
        elif not collected.get(field):
            missing.append(field)
    return missing


def _field_label(field: str, field_labels: Optional[Dict[str, str]] = None) -> str:
    if field_labels and field_labels.get(field):
        return field_labels[field]
    return {
        "issue_detail": "要处理的账号/门店、页面、权限名称和遇到的问题",
        "phone": "可用于后台查询的手机号",
        "id_card": "涉及实名/主体/权限时的身份证号或主体证件信息",
        "image": "相关页面、报错或操作入口截图",
    }.get(field, field)


def _next_clarification_question(
    round_no: int,
    reason: str,
    missing: List[str],
    *,
    workflow_summary: str = "",
    field_labels: Optional[Dict[str, str]] = None,
) -> str:
    if workflow_summary:
        labels = (
            "、".join(_field_label(f, field_labels) for f in missing)
            if missing
            else "请确认以上信息是否准确，或继续补充手机号、截图等后台排查材料"
        )
        return f"我查到相关流程：{workflow_summary} 请补充：{labels}。"
    if round_no <= 1:
        return (
            "我先帮你把这个问题整理成后台工单。请补充：要处理的账号/门店、具体页面或权限名称、遇到的现象；"
            "如果方便，请上传相关截图。"
        )
    if round_no == 2:
        return (
            "还需要用于后台定位的信息：手机号；如果涉及实名、主体认证、直播权限或账号权限，请一并补充身份证号或主体证件信息。"
        )
    labels = "、".join(_field_label(f, field_labels) for f in missing) if missing else "请确认以上信息是否准确"
    return f"我还差一点信息才能提交后台：{labels}。补充后我会转成待人工工单。"


def _manual_ticket_answer(ticket_id: Optional[str], missing: List[str], *, completion_status: str) -> str:
    ticket_label = ticket_id or "后台工单"
    base = f"已为您记录问题，工单号 {ticket_label}，后续将由专人处理。"
    if completion_status == COMPLETION_INCOMPLETE:
        if missing:
            labels = "、".join(_field_label(f) for f in missing)
            return f"当前信息未完全收集，仍缺少：{labels}。{base}"
        return f"当前信息未完全收集。{base}"
    return base


def process_clarification_turn(
    *,
    session_id: str,
    user_id: str,
    user_name: Optional[str],
    sender_id: Optional[str],
    requester_name: Optional[str],
    kb_name: Optional[str],
    query: str,
    channel: str = "web",
    has_image: bool = False,
    query_image_oss_key: Optional[str] = None,
    reason: Optional[str] = None,
    force_start: bool = False,
    workflow: Optional[Dict[str, Any]] = None,
    workflow_sources: Optional[List[Dict[str, Any]]] = None,
    continue_only: bool = False,
) -> Optional[Dict[str, Any]]:
    repo = get_service_ticket_repository()
    active = repo.find_active_clarification(
        session_id=session_id,
        user_id=user_id or "guest_default",
        kb_name=kb_name,
    )
    if continue_only and not active:
        return None
    if continue_only and active and not _is_clarification_followup(
        query=query,
        has_image=has_image,
        query_image_oss_key=query_image_oss_key,
    ):
        return None
    decision = should_clarify_query(query, has_image=has_image)
    trigger_reason = reason or decision.get("reason") or "ambiguous_query"
    if not active and not force_start and not decision.get("should_clarify"):
        return None

    existing = active.get("clarification") if active else {}
    workflow_info = _normalize_workflow_payload(workflow)
    if not workflow_info["workflow_found"] and existing.get("workflow_found"):
        workflow_info = _normalize_workflow_payload(existing)

    original_query = existing.get("original_query") or query
    trigger_reason = existing.get("reason") or trigger_reason
    intent_class = existing.get("intent_class") or _intent_class_for_reason(trigger_reason)
    turns = list(existing.get("turns") or [])
    round_no = max(int(active.get("clarification_round") or 0) if active else 0, len(turns)) + 1

    collected = _merge_collected_info(
        existing.get("collected") or {},
        query=query,
        has_image=has_image,
        query_image_oss_key=query_image_oss_key,
        requester_name=requester_name or user_name,
        sender_id=sender_id,
        user_id=user_id,
    )
    required = (
        workflow_info.get("required_fields")
        or existing.get("required_fields")
        or _required_clarification_fields(trigger_reason, original_query)
    )
    field_labels = workflow_info.get("field_labels") or existing.get("field_labels") or {}
    workflow_summary = workflow_info.get("workflow_summary") or existing.get("workflow_summary") or ""
    workflow_found = bool(workflow_info.get("workflow_found") or existing.get("workflow_found"))
    fallback_used = bool(workflow_info.get("fallback_used", existing.get("fallback_used", False)))
    missing = _missing_fields(required, collected)
    user_refused = _is_user_refusal(query)
    no_standard_workflow = (
        intent_class == INTENT_OPERATION
        and force_start
        and workflow is not None
        and not workflow_found
    )
    round_cap_reached = round_no >= _max_rounds_for_intent(intent_class)
    fields_complete = _is_complete(intent_class, missing)

    should_finalize = bool(
        user_refused
        or no_standard_workflow
        or fields_complete
        or round_cap_reached
    )
    status = "pending_manual" if should_finalize else "clarifying"

    if user_refused:
        exit_reason = EXIT_USER_REFUSED
        completion_status = COMPLETION_INCOMPLETE
    elif no_standard_workflow:
        exit_reason = EXIT_NO_STANDARD_WORKFLOW
        completion_status = COMPLETION_INCOMPLETE
    elif fields_complete:
        exit_reason = EXIT_FIELDS_COMPLETE
        completion_status = COMPLETION_COMPLETE
    elif round_cap_reached:
        exit_reason = _exit_reason_for_round_cap(intent_class)
        completion_status = COMPLETION_INCOMPLETE
    else:
        exit_reason = ""
        completion_status = existing.get("completion_status") or COMPLETION_INCOMPLETE

    if should_finalize:
        answer = _manual_ticket_answer(
            active.get("id") if active else None,
            missing,
            completion_status=completion_status,
        )
    else:
        answer = _next_clarification_question(
            round_no,
            trigger_reason,
            missing,
            workflow_summary=workflow_summary,
            field_labels=field_labels,
        )

    turns.append(
        {
            "round": round_no,
            "query": query,
            "answer": answer,
            "image_key": query_image_oss_key,
            "status": status,
        }
    )
    clarification = {
        "intent_class": intent_class,
        "reason": trigger_reason,
        "original_query": original_query,
        "required_fields": required,
        "missing_fields": missing,
        "workflow_found": workflow_found,
        "workflow_summary": workflow_summary,
        "field_labels": field_labels,
        "workflow_sources": existing.get("workflow_sources") or _workflow_sources_summary(workflow_sources),
        "fallback_used": fallback_used,
        "kb_result": existing.get("kb_result") or {},
        "image_analysis": existing.get("image_analysis") or {
            "has_image": bool(collected.get("has_image") or collected.get("image_keys")),
            "categories": [],
        },
        "collected": collected,
        "turns": turns,
        "completion_status": completion_status,
        "exit_reason": exit_reason,
        "ready_for_manual": should_finalize,
    }

    if active:
        ticket = repo.update_clarification(
            active["id"],
            status=status,
            answer=answer,
            clarification_round=round_no,
            clarification=clarification,
            sender_id=sender_id,
            requester_name=requester_name or user_name,
        )
    else:
        ticket = repo.create_with_contexts(
            session_id=session_id,
            user_id=user_id or "guest_default",
            user_name=user_name,
            kb_name=kb_name,
            query=original_query,
            answer=answer,
            status=status,
            confidence=0.0,
            fallback_reason=trigger_reason,
            quality_level="clarifying",
            sources=[],
            channel=channel or "web",
            sender_id=sender_id,
            requester_name=requester_name or user_name,
            clarification_round=round_no,
            clarification=clarification,
            contexts=build_context_snapshots(workflow_sources),
        )
    if ticket and status == "pending_manual" and "工单号 后台工单" in answer:
        answer = _manual_ticket_answer(
            ticket.get("id"),
            missing,
            completion_status=completion_status,
        )
        turns[-1]["answer"] = answer
        clarification["turns"] = turns
        ticket = repo.update_clarification(
            ticket["id"],
            status=status,
            answer=answer,
            clarification_round=round_no,
            clarification=clarification,
            sender_id=sender_id,
            requester_name=requester_name or user_name,
        )
    return {
        "ticket_id": ticket["id"] if ticket else active.get("id") if active else None,
        "status": status,
        "answer": answer,
        "clarification_round": round_no,
        "clarification": clarification,
        "confidence": 0.0,
        "finish_reason": "manual_ticket_created" if should_finalize else "clarification",
    }


def classify_ticket_status(
    *,
    used_fallback: bool,
    quality_passed: Optional[bool],
    confidence: Optional[float],
    fallback_reason: Optional[str],
    min_confidence: float = 0.6,
) -> str:
    if used_fallback or fallback_reason:
        return "pending_manual"
    if quality_passed is False:
        return "pending_manual"
    if confidence is not None and confidence < min_confidence:
        return "pending_manual"
    return "resolved_ai"


def build_context_snapshots(sources: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for idx, src in enumerate(sources or []):
        metadata = src.get("metadata") or {}
        chunk_id = src.get("chunk_id") or src.get("id")
        job_id = src.get("job_id") or metadata.get("job_id")
        chunk_index = src.get("chunk_index")
        if chunk_index is None:
            chunk_index = metadata.get("chunk_index")
        contexts.append(
            {
                "chunk_id": chunk_id,
                "job_id": job_id,
                "file_name": src.get("file_name") or src.get("title") or metadata.get("file_name"),
                "chunk_index": chunk_index,
                "score": src.get("score"),
                "content": src.get("content") or "",
                "metadata": metadata,
                "sort_order": idx,
            }
        )
    return contexts


def start_missing_knowledge_clarification(
    *,
    session_id: str,
    user_id: str,
    user_name: Optional[str],
    sender_id: Optional[str],
    requester_name: Optional[str],
    kb_name: Optional[str],
    query: str,
    channel: str,
    rag_result: Dict[str, Any],
    has_image: bool = False,
    query_image_oss_key: Optional[str] = None,
) -> Dict[str, Any]:
    repo = get_service_ticket_repository()
    collected = _merge_collected_info(
        {},
        query=query,
        has_image=has_image,
        query_image_oss_key=query_image_oss_key,
        requester_name=requester_name or user_name,
        sender_id=sender_id,
        user_id=user_id,
    )
    required = _required_clarification_fields("knowledge_missing", query)
    missing = _missing_fields(required, collected)
    answer = _next_clarification_question(
        1,
        "knowledge_missing",
        missing,
        workflow_summary="",
        field_labels={},
    )
    clarification = {
        "intent_class": INTENT_MISSING_KNOWLEDGE,
        "reason": "knowledge_missing",
        "original_query": query,
        "required_fields": required,
        "missing_fields": missing,
        "workflow_found": False,
        "workflow_summary": "",
        "field_labels": {},
        "workflow_sources": [],
        "fallback_used": True,
        "kb_result": {
            "hit": False,
            "fallback_reason": rag_result.get("fallback_reason"),
            "confidence": rag_result.get("confidence"),
            "answer": rag_result.get("answer"),
        },
        "image_analysis": {
            "has_image": bool(has_image or query_image_oss_key),
            "categories": [],
        },
        "collected": collected,
        "turns": [
            {
                "round": 1,
                "query": query,
                "answer": answer,
                "image_key": query_image_oss_key,
                "status": "clarifying",
            }
        ],
        "completion_status": COMPLETION_INCOMPLETE,
        "exit_reason": "",
        "ready_for_manual": False,
    }
    ticket = repo.create_with_contexts(
        session_id=session_id,
        user_id=user_id or "guest_default",
        user_name=user_name,
        kb_name=kb_name,
        query=query,
        answer=answer,
        status="clarifying",
        confidence=rag_result.get("confidence"),
        fallback_reason="knowledge_missing",
        quality_level="clarifying",
        sources=rag_result.get("sources") or [],
        channel=channel or "web",
        sender_id=sender_id,
        requester_name=requester_name or user_name,
        clarification_round=1,
        clarification=clarification,
        contexts=build_context_snapshots(rag_result.get("sources") or []),
    )
    return {
        "ticket_id": ticket["id"] if ticket else None,
        "status": "clarifying",
        "answer": answer,
        "clarification_round": 1,
        "clarification": clarification,
        "confidence": 0.0,
        "finish_reason": "clarification",
    }


def record_ticket(
    *,
    user_id: str,
    user_name: Optional[str],
    kb_name: Optional[str],
    query: str,
    answer: Optional[str],
    status: str,
    confidence: Optional[float] = None,
    fallback_reason: Optional[str] = None,
    quality_level: Optional[str] = None,
    sources: Optional[list] = None,
    channel: str = "web",
    processing_ms: Optional[float] = None,
    session_id: Optional[str] = None,
    sender_id: Optional[str] = None,
    requester_name: Optional[str] = None,
) -> Dict[str, Any]:
    return get_service_ticket_repository().create_with_contexts(
        session_id=session_id,
        user_id=user_id or "guest_default",
        user_name=user_name,
        kb_name=kb_name,
        query=query,
        answer=answer,
        status=status,
        confidence=confidence,
        fallback_reason=fallback_reason,
        quality_level=quality_level,
        sources=sources or [],
        channel=channel or "web",
        processing_ms=processing_ms,
        sender_id=sender_id,
        requester_name=requester_name or user_name,
        contexts=build_context_snapshots(sources),
    )


def record_qa_ticket(
    *,
    user_id: str,
    user_name: Optional[str],
    kb_name: Optional[str],
    query: str,
    answer: str,
    sources: list,
    confidence: Optional[float],
    used_fallback: bool,
    fallback_reason: Optional[str],
    quality_passed: Optional[bool],
    quality_level: Optional[str],
    session_id: Optional[str],
    channel: str,
    processing_ms: Optional[float] = None,
    sender_id: Optional[str] = None,
    requester_name: Optional[str] = None,
) -> Dict[str, Any]:
    status = classify_ticket_status(
        used_fallback=used_fallback,
        quality_passed=quality_passed,
        confidence=confidence,
        fallback_reason=fallback_reason,
    )
    return record_ticket(
        user_id=user_id,
        user_name=user_name,
        kb_name=kb_name,
        query=query,
        answer=answer,
        status=status,
        confidence=confidence,
        fallback_reason=fallback_reason,
        quality_level=quality_level,
        sources=sources,
        channel=channel,
        processing_ms=processing_ms,
        session_id=session_id,
        sender_id=sender_id,
        requester_name=requester_name,
    )


def list_tickets(**filters) -> Dict[str, Any]:
    repo = get_service_ticket_repository()
    total = repo.count(
        status=filters.get("status"),
        kb_name=filters.get("kb_name"),
        user_id=filters.get("user_id"),
        start_date=filters.get("start_date"),
        end_date=filters.get("end_date"),
    )
    items = repo.list(**filters)
    return {"total": total, "items": items}


def ticket_stats(**filters) -> Dict[str, Any]:
    return get_service_ticket_repository().stats(
        kb_name=filters.get("kb_name"),
        user_id=filters.get("user_id"),
        start_date=filters.get("start_date"),
        end_date=filters.get("end_date"),
    )


def get_ticket(ticket_id: str) -> Dict[str, Any]:
    ticket = get_service_ticket_repository().get_with_contexts(ticket_id)
    if not ticket:
        raise NotFoundError(f"服务记录不存在: {ticket_id}")
    return ticket


def update_ticket(ticket_id: str, *, status: Optional[str] = None, answer: Optional[str] = None, note: Optional[str] = None) -> Dict[str, Any]:
    if status and status not in ALL_TICKET_STATUSES:
        raise ValidationError(f"无效状态: {status}")
    updated = get_service_ticket_repository().update(
        ticket_id,
        status=status,
        answer=answer,
        note=note,
    )
    if not updated:
        raise NotFoundError(f"服务记录不存在: {ticket_id}")
    return updated


def update_original_chunk_from_ticket(ticket_id: str, chunk_id: str, content: str) -> Dict[str, Any]:
    if not content or len(content.strip()) < 2:
        raise ValidationError("切片内容不能为空")
    ticket = get_ticket(ticket_id)
    context = next((c for c in ticket.get("contexts", []) if c.get("chunk_id") == chunk_id), None)
    if not context:
        raise NotFoundError("该服务记录没有关联此切片")

    chunk = get_chunk_repository().get_by_id(chunk_id)
    if not chunk:
        raise NotFoundError("原始切片不存在")
    job_id = chunk["job_id"]
    get_chunk_repository().update_content(chunk_id, content.strip(), status="edited")
    get_job_repository().mark_needs_vectorization(
        job_id,
        stage="服务记录回修后待重新向量化",
    )
    return {"ticket_id": ticket_id, "chunk_id": chunk_id, "job_id": job_id, "status": "chunked"}


async def revectorize_ticket_context(ticket_id: str) -> Dict[str, Any]:
    ticket = get_ticket(ticket_id)
    job_ids = []
    for ctx in ticket.get("contexts", []):
        if ctx.get("job_id") and ctx["job_id"] not in job_ids:
            job_ids.append(ctx["job_id"])
    if not job_ids:
        raise ValidationError("该服务记录没有可重新向量化的上下文")

    succeeded, failed = [], []
    for job_id in job_ids:
        try:
            result = await chunk_service.upsert_job_chunks(job_id)
            succeeded.append({"job_id": job_id, "result": result})
        except Exception as exc:
            failed.append({"job_id": job_id, "error": str(exc)})
    return {"ticket_id": ticket_id, "succeeded": succeeded, "failed": failed}
