# -*- coding: utf-8 -*-
import json
import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.exceptions import NotFoundError, ValidationError
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

INTENT_OPERATION = "A"
INTENT_MISSING_KNOWLEDGE = "B"
INTENT_AMBIGUOUS = "C"
INTENT_NORMAL = "D"
STATUS_UNANSWERED_NORMAL = "unanswered_normal"

CLASSIFIER_MODEL = getattr(settings, "operation_classifier_model", "doubao-seed-2-0-mini-260428")
IMAGE_FIELD_ANALYSIS_MODEL = getattr(
    settings,
    "operation_image_analysis_model",
    getattr(settings, "default_model", "doubao-seed-2-0-pro-260215"),
)
CLASSIFIER_MAX_TOKENS = 300
CLASSIFIER_TIMEOUT = min(max(float(getattr(settings, "operation_classifier_timeout", 4.0)), 2.5), 4.0)
CLASSIFIER_REVIEW_MAX_TOKENS = 320
CLASSIFIER_REVIEW_TIMEOUT = min(CLASSIFIER_TIMEOUT, 2.0)
WORKFLOW_ANALYSIS_MAX_TOKENS = 600
WORKFLOW_ANALYSIS_TIMEOUT = min(
    max(float(getattr(settings, "operation_workflow_timeout", getattr(settings, "operation_classifier_timeout", 5.5))), 4.0),
    6.0,
)
CLARIFICATION_REPLY_MAX_TOKENS = 500
IMAGE_FIELD_ANALYSIS_MAX_TOKENS = 700
IMAGE_FIELD_ANALYSIS_TIMEOUT = min(CLASSIFIER_TIMEOUT, 4.0)
CHAT_HISTORY_LIMIT = 50

_VALID_INTENTS = {
    INTENT_OPERATION,
    INTENT_MISSING_KNOWLEDGE,
    INTENT_AMBIGUOUS,
    INTENT_NORMAL,
}


def _fallback_decision(query: str, has_image: bool) -> Dict[str, Any]:
    if has_image or not (query or "").strip():
        return {
            "intent_class": INTENT_AMBIGUOUS,
            "reason": "ambiguous_query",
            "needs_ticket_flow": True,
            "confidence": 0.0,
            "fields": {},
            "missing_fields": ["issue_detail"],
            "question": "请补充说明图片对应的业务场景、操作步骤或报错信息。",
            "rationale_brief": "image_only_or_empty_query_classifier_failed",
            "source": "fallback",
        }
    return {
        "intent_class": INTENT_NORMAL,
        "reason": "knowledge_question",
        "needs_ticket_flow": False,
        "confidence": 0.0,
        "fields": {},
        "missing_fields": [],
        "question": "",
        "rationale_brief": "classifier_failed",
        "source": "fallback",
    }


def _extract_json_payload(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("classifier response must be a JSON object")
    return payload


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _default_clarification_question(
    *,
    intent_class: str,
    missing_fields: Optional[List[Any]] = None,
    round_no: int = 1,
    has_image: bool = False,
) -> str:
    fields = [str(field) for field in (missing_fields or []) if str(field or "").strip()]
    generic_detail_fields = {
        "issue_detail",
        "problem_detail",
        "detail",
        "用户的具体需求内容",
        "用户具体问题",
        "具体问题",
        "问题细节",
    }
    specific_fields = [field for field in fields if field not in generic_detail_fields]
    if specific_fields:
        return f"请补充{ '、'.join(specific_fields) }。"
    if fields and intent_class not in {INTENT_MISSING_KNOWLEDGE, INTENT_AMBIGUOUS}:
        return f"请补充{ '、'.join(fields) }。"
    if intent_class == INTENT_MISSING_KNOWLEDGE:
        if round_no <= 2:
            return "请补充业务场景、发生入口、完整报错或截图，以及影响范围（单个账号还是多名用户）。"
        if round_no <= 4:
            return "请再确认业务背景和影响范围，例如涉及的门店/账号、订单或红包类型，以及出现频率。"
        return "请确认这个问题更接近账号权限/风控、支付限制，还是红包发放配置异常。"
    if intent_class == INTENT_AMBIGUOUS:
        if has_image:
            return "请补充截图对应的业务场景、操作步骤或完整报错信息。"
        return "请补充发生入口、账号/门店、操作步骤、完整截图，以及影响范围。"
    if intent_class == INTENT_OPERATION:
        return "请补充办理对象、账号或门店、联系方式，以及当前操作入口或报错信息。"
    return "请补充具体业务场景、操作步骤或报错信息。"


def _rag_result_proves_miss(rag_result: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(rag_result, dict):
        return False
    if not rag_result.get("used_fallback"):
        return False
    fallback_reason = str(rag_result.get("fallback_reason") or "").lower()
    if "no_relevant" in fallback_reason or "no relevant" in fallback_reason:
        return True
    if "knowledge base has no relevant content" in fallback_reason:
        return True
    if "知识库" in fallback_reason:
        return any(
            phrase in fallback_reason
            for phrase in ("无相关", "没有相关", "未找到", "没有找到")
        )
    return False


def _normalize_payload(
    payload: Dict[str, Any],
    query: str,
    has_image: bool,
    rag_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    intent = str(payload.get("intent_class") or "").strip().upper()
    if intent not in _VALID_INTENTS:
        return _fallback_decision(query, has_image)

    needs_ticket_flow = bool(payload.get("needs_ticket_flow"))
    if intent == INTENT_NORMAL:
        needs_ticket_flow = False
    elif intent == INTENT_MISSING_KNOWLEDGE:
        needs_ticket_flow = _rag_result_proves_miss(rag_result)
    elif intent in {INTENT_OPERATION, INTENT_AMBIGUOUS}:
        needs_ticket_flow = True

    fields = payload.get("fields")
    missing_fields = payload.get("missing_fields")
    normalized_missing_fields = missing_fields if isinstance(missing_fields, list) else []

    return {
        "intent_class": intent,
        "reason": str(payload.get("reason") or ""),
        "needs_ticket_flow": needs_ticket_flow,
        "confidence": _as_float(payload.get("confidence")),
        "fields": fields if isinstance(fields, dict) else {},
        "missing_fields": normalized_missing_fields,
        "question": "" if intent == INTENT_NORMAL else str(payload.get("question") or ""),
        "rationale_brief": str(payload.get("rationale_brief") or ""),
        "source": "llm",
    }


def _classifier_prompt(
    query: str,
    history: Optional[List[Dict[str, Any]]],
    has_image: bool,
    rag_result: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    history_lines = []
    for turn in (history or [])[-4:]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "user")
        content = str(turn.get("content") or "")[:300]
        history_lines.append({"role": role, "content": content})

    rag_summary = None
    if isinstance(rag_result, dict):
        sources = rag_result.get("sources") or []
        rag_summary = {
            "used_fallback": rag_result.get("used_fallback"),
            "fallback_reason": rag_result.get("fallback_reason"),
            "quality_passed": rag_result.get("quality_passed"),
            "quality_level": rag_result.get("quality_level"),
            "answer_preview": str(rag_result.get("answer") or "")[:500],
            "source_count": len(sources) if isinstance(sources, list) else 0,
        }

    return [
        {
            "role": "system",
            "content": """
你是售后/服务工单意图分类器。只输出一个 JSON 对象，不要输出 Markdown。

核心目标：
1. 优先保护 RAG。只要用户已经把“想问什么”说清楚，就先判 D，让 RAG 正常检索。
2. 只在确需人工介入时才进入工单，避免把普通咨询、教程问题、已有完整报错的问题误送工单。
3. RAG 后也不能仅因为 rag_result.used_fallback=true 就转 B；只有答案确实无效、且必须继续向用户收集现场信息时，才允许转 B。

判定顺序：
步骤1：先判断是否“语义清晰”。
- 只要 query 已经包含以下任意两类信息，就视为语义清晰，不得判 C：
  1) 业务对象/系统名/功能名；
  2) 现象/报错/提示语；
  3) 目标动作或咨询点，例如“怎么开通”“为什么失败”“怎么办”“在哪里看”“是否支持”。
- 完整问句但缺少账号、门店、订单号、手机号、截图，不算 C。这些是后续工单补充字段，不是前门分类条件。
- 前门分类只看“用户是否已经把问题说清楚”，不看“线下办理字段是否收齐”。

步骤2：判断 A。
- A=操作代办类。只有用户明确要求系统或人工替他执行具体业务动作，才是 A，例如“帮我开通/关闭/修改/绑定/解绑/处理/提交/恢复/配置权限”。
- A 必须同时满足：有具体动作对象；用户是在请求代办/处理。
- 问教程、步骤、原因、用途、位置、条件，不是 A，优先判 D。

步骤3：判断 B。
- B=知识库未覆盖。仅当本轮输入提供 rag_result，且 rag_result 明确显示 used_fallback=true，同时 answer_preview 没有给出可用解释/排查方向，并且必须继续向用户收集现场信息时，才可判 B。
- 不能仅因为 rag_result.used_fallback=true、quality 低、source 少，就直接判 B。

步骤4：判断 D。
- D=常规知识问答。用户问原因、怎么办、如何解决、如何操作、教程、步骤、功能用途、在哪里看、为什么、能不能，或已经给出完整报错希望解释/解决，均优先判 D。
- 只要 query 自身已经足够让 RAG 去检索，就判 D，不要先追问。

步骤5：判断 C。
- C=语义不明。仅限“这个怎么弄”“不行”“见图”“帮看看”“报错了”这类真正残缺表达：没有系统/业务对象、没有完整报错、没有场景，或者仅图片/截图。
- 在输出 C 前必须先自检：query 是否已经包含业务对象 + 现象/报错，或业务对象 + 目标动作/咨询点；如果是，就不能判 C。

强制边界：
- 最高优先级：下列“保护型常规问答”优先级高于 rag_result.used_fallback、fallback_reason、answer_preview。即使 answer_preview 是“当前知识库暂未找到相关信息，无法解答”，也必须判 D，needs_ticket_flow=false，question=""，不得前台工单追问。
- “企业微信发不了红包，提示该单已被其他账号发起支付，你无权再发起”
  - 如果 rag_result 为 null：必须判 D，needs_ticket_flow=false，question=""。
  - 如果 rag_result 证明知识库未覆盖：仍判 D，不在前台追问。
- 以下问题及其类似问题在 rag_result 为 null 时都必须判 D，needs_ticket_flow=false，question=""，先交给 RAG：
  - “企业微信无法发视频是什么原因？”
  - “企业微信登录时提示之前加入企业后退出，还在企业通讯录内，需联系管理员删除后再加入确定”
  - “客户积分一直在扣是什么原因？”
  - “新活动政策刷新后没有是什么原因？”
  - “富友账户有钱，但下单时显示余额不足怎么办？”
- 上面这些问题即使 rag_result.used_fallback=true，也不能直接进入前台工单追问；除非用户明确要求人工处理，否则仍判 D。
- 问“是什么原因”“怎么办”“如何解决”不是 A，也不是 C；只要有明确业务对象和现象，默认判 D。
- “这个报错怎么处理”=>C，因为缺少系统/业务对象和报错原文。
- “微信子商户号怎么开通”=>D，因为用户只是问教程。
- “帮我开通微信子商户号”=>A，因为用户要求代办开通。
- “如何开通 XX 权限”=>D，除非用户明确说“帮我开通”。

输出前自检：
1. 在判 A 前先问自己：用户是在让我替他做，还是在问怎么做？如果是问怎么做，判 D。
2. 在判 B 前先问自己：是否真的已经看到 rag_result 证明知识库未覆盖且答案没有有效方向？如果没有，不能判 B。
3. 在判 C 前先问自己：用户是否已经把问题说清楚？如果 query 已有业务对象 + 现象/报错/咨询点，就不能判 C。

question 输出自检：
1. D 的 question 必须为空字符串。
2. A/B/C 的 question 必须是下一轮要问用户的中文问题。
3. question 严禁复制、改写、复述 query。
4. question 严禁使用空泛问法：“您具体遇到了什么问题”“请描述操作需求”“请补充需求信息”“请补充遇到的问题”。
5. 如果 query 已有报错/提示语且确实需要进入 A/B/C，question 必须索要真实缺失信息，优先追问发生入口、涉及账号/门店、截图、影响范围、订单/红包类型等具体字段。

输出固定字段（字段名不要翻译）必须包含：
intent_class, reason, needs_ticket_flow, confidence, fields, missing_fields, question, rationale_brief

字段要求：
fields 必须是对象；missing_fields 必须是字符串数组；confidence 是 0 到 1。
A/C 的 needs_ticket_flow=true；D 的 needs_ticket_flow=false；B 是否 true 取决于 rag_result 是否证明未覆盖且确实需要前台追问收集信息。
""",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": query or "",
                    "history": history_lines,
                    "has_image": bool(has_image),
                    "rag_result": rag_summary,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _classifier_review_prompt(
    query: str,
    history: Optional[List[Dict[str, Any]]],
    has_image: bool,
    rag_result: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": """
你是 RAG/工单流转二审模型。只输出 JSON，不要输出 Markdown。

你只判断一个问题：候选分类是否应该在 RAG 之前拦截，或在 RAG 之后把用户问题转入前台工单追问。

默认动作是“放行RAG”。只有以下情况才“进入工单流”：
1. 用户明确请求人工/系统代办具体动作，例如“帮我开通/处理/修改/绑定/解绑/配置/恢复”。
2. 用户表达真正残缺，例如“这个怎么弄”“不行”“见图”“帮看看”“报错了”，缺少系统名、业务对象或报错原文。
3. 仅图片/截图且文本无法判断。
4. RAG 后答案明确无法回答，并且必须向用户补充收集现场信息才能创建工单。

必须“放行RAG”的情况：
0. 最高优先级：下列示例及语义相同的问题，必须“放行RAG”，这条优先级高于 candidate_decision、rag_result.used_fallback、fallback_reason 和 answer_preview。即使知识库暂未找到相关信息，也不得让用户进入前台工单追问。
1. query 已包含明确业务对象/系统名 + 现象/报错/提示语。
2. query 已包含明确业务对象/系统名 + 目标动作或咨询点，例如“怎么开通”“怎么办”“为什么失败”“在哪里看”“是否支持”。
3. query 是询问原因、怎么办、如何解决、如何操作、教程、步骤、为什么、能不能。
4. query 虽然缺少账号、门店、订单号、手机号、截图，但这不影响它已经是一个完整问句。缺少这些字段，不得作为拦成 C/B 的理由。
5. 只要 query 自身已经足够让 RAG 检索，就放行RAG，不要因为字段未收齐而拦截。
6. query 示例：“企业微信发不了红包，提示该单已被其他账号发起支付，你无权再发起”。
7. query 示例：“企业微信无法发视频是什么原因？”“客户积分一直在扣是什么原因？”“富友账户有钱，但下单时显示余额不足怎么办？”“新活动政策刷新后没有是什么原因？”。
8. query 示例：“企业微信登录时提示之前加入企业后退出，还在企业通讯录内，需联系管理员删除后再加入确定”。
9. 即使 rag_result.used_fallback=true，上述清楚的原因/怎么办/报错排查类问题也必须放行，不要前台追问；除非用户明确要求人工处理。

输出前自检：
1. 如果 candidate_decision 是 C，请先确认 query 是否真的缺少业务对象、现象/报错、咨询点；如果没有缺，就改为“放行RAG”。
2. 如果 candidate_decision 是 B，请先确认 rag_result 是否真的证明“知识库无覆盖且答案无有效方向”；如果没有证据，就改为“放行RAG”。
3. 如果 query 的核心是在问“怎么做/为什么/怎么办”，而不是“帮我做”，就改为“放行RAG”。

输出固定字段，字段名保持 action、reason、question：
action：填“放行RAG”或“进入工单流”
reason：简短中文原因
question：如果 action 为“进入工单流”，给出面向用户的下一轮追问；如果 action 为“放行RAG”，必须为空字符串。
""",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": query or "",
                    "history": history or [],
                    "has_image": bool(has_image),
                    "rag_result": rag_result,
                    "candidate_decision": candidate,
                },
                ensure_ascii=False,
                default=str,
            ),
        },
    ]


def _review_fail_open_decision(candidate: Dict[str, Any], reason: str = "llm_review_failed_allow_rag") -> Dict[str, Any]:
    return {
        "intent_class": INTENT_NORMAL,
        "reason": reason,
        "needs_ticket_flow": False,
        "confidence": candidate.get("confidence") or 0.0,
        "fields": {},
        "missing_fields": [],
        "question": "",
        "rationale_brief": "二审模型失败，文本问题默认放行 RAG，避免误拦截",
        "source": "llm_review_fallback",
    }


def _review_ticket_decision_with_llm(
    *,
    query: str,
    history: Optional[List[Dict[str, Any]]],
    has_image: bool,
    rag_result: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        text = get_llm_service().responses_text(
            _classifier_review_prompt(query, history, has_image, rag_result, candidate),
            model=CLASSIFIER_MODEL,
            temperature=0.0,
            max_tokens=CLASSIFIER_REVIEW_MAX_TOKENS,
            timeout=CLASSIFIER_REVIEW_TIMEOUT,
            max_retries=0,
        )
        payload = _extract_json_payload(text)
        action = str(payload.get("action") or payload.get("动作") or "").strip()
        normalized_action = action.lower().replace(" ", "").replace("_", "")
        if normalized_action in {"allowrag", "放行rag", "放行", "允许rag", "继续rag", "正常rag"}:
            return {
                "intent_class": INTENT_NORMAL,
                "reason": str(payload.get("reason") or "llm_review_allow_rag"),
                "needs_ticket_flow": False,
                "confidence": _as_float(payload.get("confidence")) or candidate.get("confidence") or 0.0,
                "fields": {},
                "missing_fields": [],
                "question": "",
                "rationale_brief": str(payload.get("reason") or "二审模型判定应先放行 RAG"),
                "source": "llm_review",
            }
        if normalized_action in {"ticketflow", "进入工单流", "工单流", "进入工单", "拦截进入工单", "工单追问"}:
            reviewed = dict(candidate)
            question = str(payload.get("question") or "").strip()
            if question:
                reviewed["question"] = question
            reviewed["source"] = "llm_review"
            return reviewed
        if str(candidate.get("intent_class") or "").upper() in {INTENT_AMBIGUOUS, INTENT_MISSING_KNOWLEDGE} and not has_image:
            logger.warning("ticket intent classifier review returned unknown action: %s", action)
            return _review_fail_open_decision(candidate, reason="llm_review_invalid_action_allow_rag")
        return candidate
    except Exception as exc:
        logger.warning("ticket intent classifier review failed: %s", exc)
        if str(candidate.get("intent_class") or "").upper() in {INTENT_AMBIGUOUS, INTENT_MISSING_KNOWLEDGE} and not has_image:
            return _review_fail_open_decision(candidate)
        return candidate


def classify_ticket_intent_with_llm(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    has_image: bool = False,
    rag_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        text = get_llm_service().responses_text(
            _classifier_prompt(query, history, has_image, rag_result),
            model=CLASSIFIER_MODEL,
            temperature=0.0,
            max_tokens=CLASSIFIER_MAX_TOKENS,
            timeout=CLASSIFIER_TIMEOUT,
            max_retries=0,
        )
        payload = _extract_json_payload(text)
        decision = _normalize_payload(payload, query, has_image, rag_result)
        if (
            decision.get("source") == "llm"
            and decision.get("intent_class") in {INTENT_AMBIGUOUS, INTENT_MISSING_KNOWLEDGE}
            and decision.get("needs_ticket_flow")
            and not has_image
        ):
            return _review_ticket_decision_with_llm(
                query=query,
                history=history,
                has_image=has_image,
                rag_result=rag_result,
                candidate=decision,
            )
        return decision
    except Exception as exc:
        logger.warning("ticket intent classifier failed: %s", exc)
        return _fallback_decision(query, has_image)


def _workflow_default(reason: str = "workflow_analysis_failed") -> Dict[str, Any]:
    return {
        "workflow_found": False,
        "confidence": 0.0,
        "workflow_summary": "",
        "required_fields": [],
        "required_field_details": [],
        "question": "",
        "workflow_sources": [],
        "rationale_brief": reason,
        "source": "fallback",
    }


def _workflow_candidate_payload(rag_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(rag_result, dict):
        return {"answer": "", "sources": []}
    sources = []
    evidence_parts = []
    answer = str(rag_result.get("answer") or "")[:800]
    if answer:
        evidence_parts.append(f"RAG答案摘要：\n{answer}")
    for source in (rag_result.get("sources") or [])[:3]:
        if not isinstance(source, dict):
            continue
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        parent_content = source.get("parent_content") or metadata.get("parent_content")
        content = str(source.get("content") or "")[:500]
        item = {
            "file_name": source.get("file_name") or source.get("title"),
            "chunk_index": source.get("chunk_index"),
            "content": content,
            "score": source.get("score"),
        }
        if parent_content:
            item["parent_content"] = str(parent_content)[:1200]
        sources.append(item)
        evidence_parts.append(
            "\n".join(
                part
                for part in (
                    f"来源：{item.get('file_name') or '未知'} / chunk_index={item.get('chunk_index')}",
                    f"子块内容：{content}" if content else "",
                    f"父块上下文：{item.get('parent_content')}" if item.get("parent_content") else "",
                )
                if part
            )
        )
    return {
        "answer": answer,
        "candidate_text": "\n\n".join(evidence_parts)[:3200],
        "used_fallback": rag_result.get("used_fallback"),
        "fallback_reason": rag_result.get("fallback_reason"),
        "sources": sources,
    }


def _workflow_analysis_prompt(
    query: str,
    decision: Dict[str, Any],
    rag_result: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "你是工单系统的 SOP 判定与字段抽取器，只能输出一个合法 JSON 对象，不要 Markdown。"
                "只依据固定输入 knowledge_candidates 判断，不能根据关键词、常识或用户意图自行猜测，严禁仅凭文件名或标题命中。"
                "必须先读候选证据文本（candidate_text）；answer 为空不代表没有 SOP。"
                "将固定字段 workflow_found 设为 true 的条件：在候选证据文本或候选正文（content、parent_content）中，能看到与用户代办动作一致的办理资料、入口、步骤、提交、审核、签约或注意事项，至少两类证据。"
                "将固定字段 workflow_found 设为 false 的条件：只有概念、用途、费率、同名词，或正文没有办理资料、入口、流程、提交规则。"
                "正例：用户要代办开通微信子商户号，候选含准备资料(营业执照、法人身份证、手机号邮箱、基本存款账户)和开户流程(店长端-门店收款码管理-微信子商户号提交资料/审核/签约)，必须判 true。"
                "反例：候选只说明子商户号用途或手续费，判 false。"
                "若 workflow_found 为 true，抽取还需向用户确认/收集的字段，字段 key 使用英文蛇形命名，例如 issue_detail, store, account, phone, business_license, legal_person_id_card, legal_person_phone, legal_person_email, bank_account, screenshot。"
                "question 用中文逐项追问缺失字段；workflow_sources 只能引用输入候选 file_name 和 chunk_index。"
                "JSON 字段：workflow_found, confidence, workflow_summary, required_fields, required_field_details, question, workflow_sources, rationale_brief。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_query": query or "",
                    "intent_decision": {
                        "intent_class": decision.get("intent_class"),
                        "fields": decision.get("fields") if isinstance(decision.get("fields"), dict) else {},
                        "missing_fields": decision.get("missing_fields")
                        if isinstance(decision.get("missing_fields"), list)
                        else [],
                        "question": decision.get("question") or "",
                    },
                    "knowledge_candidates": _workflow_candidate_payload(rag_result),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _normalize_workflow_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    workflow_found = bool(payload.get("workflow_found"))
    if not workflow_found:
        result = _workflow_default("llm_no_workflow")
        result["confidence"] = _as_float(payload.get("confidence"))
        result["rationale_brief"] = str(payload.get("rationale_brief") or "llm_no_workflow")
        result["source"] = "llm"
        return result

    raw_details = payload.get("required_field_details")
    details = raw_details if isinstance(raw_details, list) else []
    normalized_details = []
    field_keys = []
    for item in details:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        if key not in field_keys:
            field_keys.append(key)
        normalized_details.append(
            {
                "key": key,
                "label": str(item.get("label") or key),
                "reason": str(item.get("reason") or ""),
            }
        )
    raw_fields = payload.get("required_fields")
    if isinstance(raw_fields, list):
        for field in raw_fields:
            key = str(field or "").strip()
            if key and key not in field_keys:
                field_keys.append(key)
    if "issue_detail" not in field_keys:
        field_keys.insert(0, "issue_detail")

    sources = []
    for source in payload.get("workflow_sources") or []:
        if isinstance(source, dict):
            sources.append(
                {
                    "file_name": source.get("file_name"),
                    "chunk_index": source.get("chunk_index"),
                }
            )

    return {
        "workflow_found": True,
        "confidence": _as_float(payload.get("confidence")),
        "workflow_summary": str(payload.get("workflow_summary") or ""),
        "required_fields": field_keys,
        "required_field_details": normalized_details,
        "question": str(payload.get("question") or ""),
        "workflow_sources": sources,
        "rationale_brief": str(payload.get("rationale_brief") or ""),
        "source": "llm",
    }


def analyze_operation_workflow_with_llm(
    *,
    query: str,
    decision: Dict[str, Any],
    rag_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = _workflow_candidate_payload(rag_result)
    if not candidates.get("answer") and not candidates.get("sources"):
        return _workflow_default("no_knowledge_candidates")
    try:
        text = get_llm_service().responses_text(
            _workflow_analysis_prompt(query, decision, rag_result),
            model=CLASSIFIER_MODEL,
            temperature=0.0,
            max_tokens=WORKFLOW_ANALYSIS_MAX_TOKENS,
            timeout=WORKFLOW_ANALYSIS_TIMEOUT,
            max_retries=0,
        )
        payload = _extract_json_payload(text)
        return _normalize_workflow_payload(payload)
    except Exception as exc:
        logger.warning("operation workflow analyzer failed: %s", exc)
        return _workflow_default()


def _active_ticket_clarification(active_ticket: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(active_ticket, dict):
        return {}
    clarification = active_ticket.get("clarification")
    return clarification if isinstance(clarification, dict) else {}


def _clarification_reply_default(
    query: str,
    active_ticket: Optional[Dict[str, Any]],
    has_image: bool,
    reason: str = "clarification_reply_analysis_failed",
) -> Dict[str, Any]:
    clarification = _active_ticket_clarification(active_ticket)
    intent = str(clarification.get("intent_class") or INTENT_AMBIGUOUS).upper()
    if intent not in _VALID_INTENTS or intent == INTENT_NORMAL:
        intent = INTENT_AMBIGUOUS
    fields: Dict[str, Any] = {}
    if (query or "").strip():
        fields["issue_detail"] = query.strip()
    if has_image:
        fields["has_image"] = True
    return {
        "intent_class": intent,
        "reason": reason,
        "needs_ticket_flow": True,
        "confidence": 0.0,
        "fields": fields,
        "missing_fields": _as_list(clarification.get("missing_fields")) or ["issue_detail"],
        "question": "请继续补充具体业务场景、操作步骤或报错信息。",
        "ready_for_manual": False,
        "user_refused": False,
        "resolved_query": query or clarification.get("original_query") or "",
        "rationale_brief": reason,
        "source": "fallback",
    }


def _clarification_reply_prompt(
    query: str,
    active_ticket: Dict[str, Any],
    has_image: bool,
) -> List[Dict[str, Any]]:
    clarification = _active_ticket_clarification(active_ticket)
    chat_history = load_ticket_history_context(
        str(active_ticket.get("session_id") or ""),
        active_ticket=active_ticket,
        limit=8,
    )
    payload = {
        "current_user_reply": query or "",
        "has_image": bool(has_image),
        "active_ticket": {
            "intent_class": clarification.get("intent_class"),
            "original_query": clarification.get("original_query"),
            "required_fields": _as_list(clarification.get("required_fields")),
            "missing_fields": _as_list(clarification.get("missing_fields")),
            "collected": clarification.get("collected") if isinstance(clarification.get("collected"), dict) else {},
            "workflow_found": clarification.get("workflow_found"),
            "workflow_summary": clarification.get("workflow_summary") or "",
            "round": active_ticket.get("clarification_round"),
            "max_rounds": _round_limit(str(clarification.get("intent_class") or INTENT_AMBIGUOUS).upper()),
        },
        "recent_chat_history": chat_history[-8:],
    }
    return [
        {
            "role": "system",
            "content": (
                "你是工单澄清续接分析器。只能输出一个 JSON 对象，禁止 Markdown，禁止解释过程。"
                "你的任务是合并上下文：读取当前澄清中工单、已收集字段、缺失字段、最近对话和用户最新回复，"
                "判断用户是在补充工单信息、拒绝继续回答，还是已经把问题澄清成可由知识库直接回答的普通问题。"
                "如果仍需工单流程，将固定字段 needs_ticket_flow 设为 true，并抽取本轮新增 fields，更新 missing_fields，生成下一句中文 question。"
                "如果用户明确表示不知道、不想回答、无法提供，将固定字段 user_refused 和 ready_for_manual 都设为 true。"
                "如果信息已经足够交给人工处理，将固定字段 ready_for_manual 设为 true。"
                "如果澄清后已经是普通知识库咨询，将固定字段 intent_class 设为 D、needs_ticket_flow 设为 false，并给出 resolved_query。"
                "不要因为仍然缺少账号、门店、订单号、截图，就阻止转成 D；如果最新回复 + 原问题 + 历史对话已经能组成完整问句，就应转成 D。"
                "完整问句的标准是：已经能看出业务对象/系统名，以及现象、报错、目标动作或咨询点中的至少一项。"
                "A/D 边界继续严格：教程、怎么、如何、有什么用、在哪里看、为什么等咨询不是 A，除非用户要求代办具体动作。"
                "输出固定字段（字段名不要翻译）必须包含 intent_class, needs_ticket_flow, confidence, fields, missing_fields, question, "
                "ready_for_manual, user_refused, resolved_query, rationale_brief。"
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _normalize_clarification_reply_payload(
    payload: Dict[str, Any],
    query: str,
    active_ticket: Dict[str, Any],
    has_image: bool,
) -> Dict[str, Any]:
    intent = str(payload.get("intent_class") or "").strip().upper()
    if intent not in _VALID_INTENTS:
        return _clarification_reply_default(query, active_ticket, has_image, "invalid_clarification_reply_intent")

    fields = payload.get("fields")
    missing_fields = payload.get("missing_fields")
    needs_ticket_flow = bool(payload.get("needs_ticket_flow"))
    if intent == INTENT_NORMAL:
        needs_ticket_flow = False

    return {
        "intent_class": intent,
        "reason": str(payload.get("reason") or "clarification_reply"),
        "needs_ticket_flow": needs_ticket_flow,
        "confidence": _as_float(payload.get("confidence")),
        "fields": fields if isinstance(fields, dict) else {},
        "missing_fields": missing_fields if isinstance(missing_fields, list) else [],
        "question": str(payload.get("question") or ""),
        "ready_for_manual": bool(payload.get("ready_for_manual")),
        "user_refused": bool(payload.get("user_refused")),
        "resolved_query": str(payload.get("resolved_query") or query or ""),
        "rationale_brief": str(payload.get("rationale_brief") or ""),
        "source": "llm",
    }


def analyze_clarification_reply_with_llm(
    *,
    query: str,
    active_ticket: Dict[str, Any],
    has_image: bool = False,
) -> Dict[str, Any]:
    try:
        text = get_llm_service().responses_text(
            _clarification_reply_prompt(query, active_ticket, has_image),
            model=CLASSIFIER_MODEL,
            temperature=0.0,
            max_tokens=CLARIFICATION_REPLY_MAX_TOKENS,
            timeout=CLASSIFIER_TIMEOUT,
            max_retries=0,
        )
        payload = _extract_json_payload(text)
        return _normalize_clarification_reply_payload(payload, query, active_ticket, has_image)
    except Exception as exc:
        logger.warning("clarification reply analyzer failed: %s", exc)
        return _clarification_reply_default(query, active_ticket, has_image)


def _image_field_default(reason: str = "image_field_analysis_failed") -> Dict[str, Any]:
    return {
        "fields": {},
        "document_types": [],
        "image_summary": "",
        "rationale_brief": reason,
        "source": "fallback",
    }


def _ticket_image_field_prompt(
    *,
    query: str,
    active_ticket: Dict[str, Any],
    image_url: Optional[str] = None,
    image_urls: Optional[List[str]] = None,
    query_image_oss_key: Optional[str],
    query_image_oss_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    clarification = _active_ticket_clarification(active_ticket)
    normalized_image_urls = [url for url in (image_urls or []) if str(url or "").strip()]
    if image_url and image_url not in normalized_image_urls:
        normalized_image_urls.insert(0, image_url)
    normalized_oss_keys = [key for key in (query_image_oss_keys or []) if str(key or "").strip()]
    if query_image_oss_key and query_image_oss_key not in normalized_oss_keys:
        normalized_oss_keys.insert(0, query_image_oss_key)
    payload = {
        "current_user_reply": query or "",
        "image_oss_key": query_image_oss_key,
        "image_oss_keys": normalized_oss_keys,
        "image_count": len(normalized_image_urls),
        "active_ticket": {
            "intent_class": clarification.get("intent_class"),
            "original_query": clarification.get("original_query"),
            "required_fields": _as_list(clarification.get("required_fields")),
            "required_field_details": _as_list(clarification.get("required_field_details")),
            "missing_fields": _as_list(clarification.get("missing_fields")),
            "collected": clarification.get("collected") if isinstance(clarification.get("collected"), dict) else {},
            "workflow_summary": clarification.get("workflow_summary") or "",
        },
    }
    user_content: List[Dict[str, Any]] = [{"text": json.dumps(payload, ensure_ascii=False)}]
    user_content.extend({"image": url} for url in normalized_image_urls)
    return [
        {
            "role": "system",
            "content": """
你是工单图片资料识别助手。请读取用户上传的图片，判断它是否能补齐当前工单缺失字段。
只输出一个 JSON 对象，不要输出 Markdown，不要解释过程。字段名不要翻译。

识别原则：
1. 只填图片中能明确看出或读出的字段，禁止根据缺失字段名称猜测。
2. 对营业执照、法人身份证、法人授权书/授权函/授权委托书、银行卡/基本存款账户这类资料图片，如果能明确判断图片类型，即使部分号码看不清，也可以把对应资料字段标记为已上传图片。
3. 对手机号、邮箱、账号、开户行、银行卡号等文本字段，只有图片中可读时才抽取具体值。
4. 如果图片与缺失字段无关，fields 返回空对象。
5. active_ticket.required_field_details 会给出字段中文含义和收集原因，你需要优先根据这些中文说明判断图片对应哪个缺失字段，而不是只看英文 key。
6. 如果你识别出图片是“法人授权材料”等资料，并且 required_field_details 或 missing_fields 中存在语义对应字段，请直接把该字段填入 fields，不要只写 document_type。

输出固定字段：
fields, document_types, image_summary, rationale_brief

fields 是对象，key 必须从 active_ticket.required_fields 或 active_ticket.missing_fields 中选择。
每个字段 value 使用对象，建议包含：
value, document_type, confidence, extracted
其中 confidence 是 0 到 1；extracted 放可读出的结构化信息。
""",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _allowed_image_fields(active_ticket: Dict[str, Any]) -> set:
    clarification = _active_ticket_clarification(active_ticket)
    fields = set()
    for field in _as_list(clarification.get("required_fields")) + _as_list(clarification.get("missing_fields")):
        key = str(field or "").strip()
        if key:
            fields.add(key)
    return fields


def _normalize_image_field_value(
    value: Any,
    *,
    query_image_oss_key: Optional[str],
    query_image_oss_keys: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, dict):
        normalized = {str(k): v for k, v in value.items() if v not in (None, "", [], {})}
    else:
        normalized = {"value": str(value)}
    if not normalized.get("value"):
        normalized["value"] = "已上传图片资料"
    normalized["source"] = "image"
    normalized_oss_keys = [key for key in (query_image_oss_keys or []) if str(key or "").strip()]
    if query_image_oss_key and query_image_oss_key not in normalized_oss_keys:
        normalized_oss_keys.insert(0, query_image_oss_key)
    if query_image_oss_key:
        normalized["oss_key"] = query_image_oss_key
    if normalized_oss_keys:
        normalized["oss_keys"] = normalized_oss_keys
    if "confidence" in normalized:
        normalized["confidence"] = _as_float(normalized.get("confidence"))
    return normalized


def _normalize_image_field_payload(
    payload: Dict[str, Any],
    *,
    active_ticket: Dict[str, Any],
    query_image_oss_key: Optional[str],
    query_image_oss_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    raw_fields = payload.get("fields") if isinstance(payload, dict) else {}
    raw_fields = raw_fields if isinstance(raw_fields, dict) else {}
    allowed = _allowed_image_fields(active_ticket)
    fields: Dict[str, Any] = {}
    for raw_key, raw_value in raw_fields.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        if allowed and key not in allowed:
            continue
        normalized_value = _normalize_image_field_value(
            raw_value,
            query_image_oss_key=query_image_oss_key,
            query_image_oss_keys=query_image_oss_keys,
        )
        if normalized_value:
            fields[key] = normalized_value

    document_types = payload.get("document_types") if isinstance(payload, dict) else []
    return {
        "fields": fields,
        "document_types": document_types if isinstance(document_types, list) else [],
        "image_summary": str(payload.get("image_summary") or "") if isinstance(payload, dict) else "",
        "rationale_brief": str(payload.get("rationale_brief") or "") if isinstance(payload, dict) else "",
        "source": "llm",
    }


def analyze_ticket_image_fields_with_llm(
    *,
    query: str,
    active_ticket: Dict[str, Any],
    image_url: Optional[str] = None,
    image_urls: Optional[List[str]] = None,
    query_image_oss_key: Optional[str] = None,
    query_image_oss_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    normalized_image_urls = [url for url in (image_urls or []) if str(url or "").strip()]
    if image_url and image_url not in normalized_image_urls:
        normalized_image_urls.insert(0, image_url)
    normalized_oss_keys = [key for key in (query_image_oss_keys or []) if str(key or "").strip()]
    if query_image_oss_key and query_image_oss_key not in normalized_oss_keys:
        normalized_oss_keys.insert(0, query_image_oss_key)
    if not normalized_image_urls:
        return _image_field_default("no_image_url")
    try:
        text = get_llm_service().chat_with_images(
            _ticket_image_field_prompt(
                query=query,
                active_ticket=active_ticket,
                image_url=normalized_image_urls[0],
                image_urls=normalized_image_urls,
                query_image_oss_key=query_image_oss_key,
                query_image_oss_keys=normalized_oss_keys,
            ),
            model=IMAGE_FIELD_ANALYSIS_MODEL,
            temperature=0.0,
            max_tokens=IMAGE_FIELD_ANALYSIS_MAX_TOKENS,
            timeout=IMAGE_FIELD_ANALYSIS_TIMEOUT,
            max_retries=0,
            disable_thinking=True,
        )
        payload = _extract_json_payload(text)
        return _normalize_image_field_payload(
            payload,
            active_ticket=active_ticket,
            query_image_oss_key=query_image_oss_key,
            query_image_oss_keys=normalized_oss_keys,
        )
    except Exception as exc:
        logger.warning("ticket image field analyzer failed: %s", exc)
        return _image_field_default()


def merge_ticket_image_analysis_into_decision(
    decision: Dict[str, Any],
    image_analysis: Optional[Dict[str, Any]],
    active_ticket: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(image_analysis, dict) or not isinstance(image_analysis.get("fields"), dict) or not image_analysis["fields"]:
        return decision

    merged = dict(decision or {})
    decision_fields = merged.get("fields") if isinstance(merged.get("fields"), dict) else {}
    fields = {**decision_fields, **image_analysis["fields"]}
    merged["fields"] = fields
    merged["image_analysis"] = image_analysis

    clarification = _active_ticket_clarification(active_ticket)
    active_intent = str(clarification.get("intent_class") or "").upper()
    if active_intent in _VALID_INTENTS and active_intent != INTENT_NORMAL:
        merged["intent_class"] = active_intent
        merged["needs_ticket_flow"] = True

    required_fields = _as_list(clarification.get("required_fields")) or _as_list(merged.get("missing_fields"))
    collected = dict(clarification.get("collected") if isinstance(clarification.get("collected"), dict) else {})
    collected.update(fields)
    if required_fields:
        missing_fields = _collected_missing_fields(required_fields, collected)
        merged["missing_fields"] = missing_fields
        if not missing_fields:
            merged["question"] = ""
    return merged


MAX_CLARIFICATION_ROUNDS = {
    INTENT_OPERATION: 20,
    INTENT_MISSING_KNOWLEDGE: 5,
    INTENT_AMBIGUOUS: 3,
}

EXIT_FIELDS_COMPLETE = "fields_complete"
EXIT_NO_STANDARD_WORKFLOW = "no_standard_workflow"
EXIT_MAX_ROUNDS = "max_rounds"
EXIT_SEMANTIC_UNRESOLVED = "semantic_unresolved"
COMPLETION_COMPLETE = "complete"
COMPLETION_INCOMPLETE = "incomplete"


def get_service_ticket_repository():
    from app.db.service_ticket_repository import get_service_ticket_repository as _get_repo

    return _get_repo()


def find_active_clarification_ticket(
    *,
    session_id: str,
    user_id: str,
    kb_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    return get_service_ticket_repository().find_active_clarification(
        session_id=session_id,
        user_id=_normal_ticket_user_id(user_id),
        kb_name=kb_name,
    )


def list_tickets(
    *,
    status: Optional[str] = None,
    kb_name: Optional[str] = None,
    user_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    repo = get_service_ticket_repository()
    filters = {
        "status": status,
        "kb_name": kb_name,
        "user_id": user_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    total = repo.count(**filters)
    items = repo.list(limit=limit, offset=offset, **filters)
    return {"total": total, "items": items}


def ticket_stats(
    *,
    kb_name: Optional[str] = None,
    user_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    return get_service_ticket_repository().stats(
        kb_name=kb_name,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )


def get_ticket(ticket_id: str) -> Dict[str, Any]:
    ticket = get_service_ticket_repository().get_with_contexts(ticket_id)
    if not ticket:
        raise NotFoundError(f"service ticket not found: {ticket_id}")
    return ticket


def update_ticket(
    ticket_id: str,
    *,
    status: Optional[str] = None,
    answer: Optional[str] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    ticket = get_service_ticket_repository().update(
        ticket_id,
        status=status,
        answer=answer,
        note=note,
    )
    if not ticket:
        raise NotFoundError(f"service ticket not found: {ticket_id}")
    return ticket


def delete_ticket(ticket_id: str) -> Dict[str, Any]:
    deleted = get_service_ticket_repository().delete(ticket_id)
    if not deleted:
        raise NotFoundError(f"service ticket not found: {ticket_id}")
    return {"id": ticket_id, "deleted": True}


def _find_ticket_context(ticket: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
    for context in ticket.get("contexts") or []:
        if str(context.get("chunk_id") or "") == str(chunk_id):
            return context
    raise NotFoundError(f"ticket context chunk not found: {chunk_id}")


def update_original_chunk_from_ticket(ticket_id: str, chunk_id: str, content: str) -> Dict[str, Any]:
    if not (content or "").strip():
        raise ValidationError("chunk content cannot be empty")

    ticket = get_ticket(ticket_id)
    context = _find_ticket_context(ticket, chunk_id)

    from app.db import get_chunk_repository, get_job_repository

    get_chunk_repository().update_content(chunk_id, content, "edited")
    job_id = context.get("job_id")
    if job_id:
        get_job_repository().mark_needs_vectorization(str(job_id))

    return {
        "ticket_id": ticket_id,
        "chunk_id": chunk_id,
        "job_id": str(job_id) if job_id else None,
        "status": "chunk_updated",
    }


async def revectorize_ticket_context(ticket_id: str) -> Dict[str, Any]:
    ticket = get_ticket(ticket_id)
    job_ids = []
    for context in ticket.get("contexts") or []:
        job_id = context.get("job_id")
        if job_id and str(job_id) not in job_ids:
            job_ids.append(str(job_id))

    from app.services import job_service

    succeeded = []
    failed = []
    for job_id in job_ids:
        try:
            result = await job_service.upsert_job_to_milvus(job_id)
            succeeded.append({"job_id": job_id, "result": result})
        except Exception as exc:
            failed.append({"job_id": job_id, "error": str(exc)})

    return {"succeeded": succeeded, "failed": failed}


def resolve_active_clarification_with_rag(
    *,
    active_ticket: Dict[str, Any],
    query: str,
    answer: str,
    rag_result: Optional[Dict[str, Any]] = None,
    has_image: bool = False,
    query_image_oss_key: Optional[str] = None,
    query_image_oss_keys: Optional[List[str]] = None,
    sender_id: Optional[str] = None,
    requester_name: Optional[str] = None,
    user_name: Optional[str] = None,
    entry_user_id: Optional[str] = None,
    entry_user_name: Optional[str] = None,
    entry_source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(active_ticket, dict) or not active_ticket.get("id"):
        return None

    repo = get_service_ticket_repository()
    locked_user_id = _normal_ticket_user_id(active_ticket.get("user_id"))
    locked_session_id = str(active_ticket.get("session_id") or "")
    locked_kb_name = active_ticket.get("kb_name")

    def _run(active: Optional[Dict[str, Any]], conn) -> Optional[Dict[str, Any]]:
        if not isinstance(active, dict) or not active.get("id"):
            return None

        existing = _active_ticket_clarification(active)
        turns = list(existing.get("turns") or [])
        round_no = max(int(active.get("clarification_round") or 0), len(turns)) + 1
        turns.append(
            {
                "round": round_no,
                "query": query or "",
                "answer": answer or "",
                "status": "resolved_ai",
                "image_key": query_image_oss_key,
            }
        )
        clarification = {
            **existing,
            "turns": turns,
            "chat_history": _build_ticket_chat_history(
                session_id=locked_session_id,
                existing=existing,
                query=query,
                answer=answer,
                has_image=has_image,
                query_image_oss_key=query_image_oss_key,
                query_image_oss_keys=query_image_oss_keys,
            ),
            "kb_result": rag_result if isinstance(rag_result, dict) else existing.get("kb_result") or {},
            "completion_status": COMPLETION_COMPLETE,
            "exit_reason": "resolved_by_rag",
            "ready_for_manual": False,
            "image_analysis": _merged_ticket_image_analysis(
                existing,
                {},
                has_image=has_image,
                query_image_oss_key=query_image_oss_key,
                query_image_oss_keys=query_image_oss_keys,
            ),
        }
        ticket = repo.update_clarification(
            active["id"],
            status="resolved_ai",
            answer=answer or "",
            clarification_round=round_no,
            clarification=clarification,
            sender_id=sender_id,
            requester_name=requester_name or user_name,
            conn=conn,
        )
        context_snapshots = _ticket_contexts_from_rag_result(rag_result)
        if context_snapshots:
            repo.replace_contexts(active["id"], context_snapshots, conn=conn)
        return ticket

    return repo.run_clarification_transaction(
        session_id=locked_session_id,
        user_id=locked_user_id,
        kb_name=locked_kb_name,
        callback=_run,
    )


def resolve_active_clarification_as_unanswered_normal(
    *,
    active_ticket: Dict[str, Any],
    query: str,
    answer: str,
    rag_result: Optional[Dict[str, Any]] = None,
    has_image: bool = False,
    query_image_oss_key: Optional[str] = None,
    query_image_oss_keys: Optional[List[str]] = None,
    sender_id: Optional[str] = None,
    requester_name: Optional[str] = None,
    user_name: Optional[str] = None,
    entry_user_id: Optional[str] = None,
    entry_user_name: Optional[str] = None,
    entry_source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(active_ticket, dict) or not active_ticket.get("id"):
        return None

    repo = get_service_ticket_repository()
    locked_user_id = _normal_ticket_user_id(active_ticket.get("user_id"))
    locked_session_id = str(active_ticket.get("session_id") or "")
    locked_kb_name = active_ticket.get("kb_name")

    def _run(active: Optional[Dict[str, Any]], conn) -> Optional[Dict[str, Any]]:
        if not isinstance(active, dict) or not active.get("id"):
            return None

        existing = _active_ticket_clarification(active)
        turns = list(existing.get("turns") or [])
        round_no = max(int(active.get("clarification_round") or 0), len(turns)) + 1
        normalized_oss_keys = [key for key in (query_image_oss_keys or []) if str(key or "").strip()]
        if query_image_oss_key and query_image_oss_key not in normalized_oss_keys:
            normalized_oss_keys.insert(0, query_image_oss_key)
        turns.append(
            {
                "round": round_no,
                "query": query or "",
                "answer": answer or "",
                "status": STATUS_UNANSWERED_NORMAL,
                "image_key": normalized_oss_keys[0] if normalized_oss_keys else query_image_oss_key,
                "image_keys": normalized_oss_keys,
            }
        )
        clarification = {
            **existing,
            "intent_class": INTENT_NORMAL,
            "reason": "unanswered_normal_after_rag_fallback",
            "missing_fields": [],
            "turns": turns,
            "chat_history": _build_ticket_chat_history(
                session_id=locked_session_id,
                existing=existing,
                query=query,
                answer=answer,
                has_image=has_image,
                query_image_oss_key=query_image_oss_key,
                query_image_oss_keys=query_image_oss_keys,
            ),
            "kb_result": rag_result if isinstance(rag_result, dict) else existing.get("kb_result") or {},
            "completion_status": COMPLETION_COMPLETE,
            "exit_reason": "returned_to_rag_but_unanswered",
            "ready_for_manual": False,
            "image_analysis": _merged_ticket_image_analysis(
                existing,
                {},
                has_image=has_image,
                query_image_oss_key=query_image_oss_key,
                query_image_oss_keys=query_image_oss_keys,
            ),
        }
        ticket = repo.update_clarification(
            active["id"],
            status=STATUS_UNANSWERED_NORMAL,
            answer=answer or "",
            clarification_round=round_no,
            clarification=clarification,
            sender_id=sender_id,
            requester_name=requester_name or user_name,
            conn=conn,
        )
        context_snapshots = _ticket_contexts_from_rag_result(rag_result)
        if context_snapshots:
            repo.replace_contexts(active["id"], context_snapshots, conn=conn)
        return ticket

    return repo.run_clarification_transaction(
        session_id=locked_session_id,
        user_id=locked_user_id,
        kb_name=locked_kb_name,
        callback=_run,
    )


def record_ai_resolved_ticket(
    *,
    session_id: str,
    user_id: str,
    user_name: Optional[str] = None,
    kb_name: Optional[str] = None,
    query: str,
    answer: str,
    status: str = "resolved_ai",
    confidence: Optional[float] = None,
    rag_result: Optional[Dict[str, Any]] = None,
    channel: str = "web",
    sender_id: Optional[str] = None,
    requester_name: Optional[str] = None,
    entry_user_id: Optional[str] = None,
    entry_user_name: Optional[str] = None,
    entry_source: Optional[str] = None,
    has_image: bool = False,
    query_image_oss_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not (query or "").strip() or not (answer or "").strip():
        return None

    clarification = {
        "intent_class": INTENT_NORMAL,
        "reason": "resolved_by_ai",
        "original_query": query or "",
        "required_fields": [],
        "required_field_details": [],
        "missing_fields": [],
        "collected": {"issue_detail": query or ""},
        "workflow_found": False,
        "workflow_summary": "",
        "workflow_sources": [],
        "kb_result": rag_result if isinstance(rag_result, dict) else {},
        "image_analysis": {"has_image": bool(has_image), "query_image_oss_key": query_image_oss_key},
        "turns": [
            {
                "round": 1,
                "query": query or "",
                "answer": answer or "",
                "status": status,
                "image_key": query_image_oss_key,
            }
        ],
        "chat_history": _build_ticket_chat_history(
            session_id=session_id,
            existing={},
            query=query,
            answer=answer,
            has_image=has_image,
            query_image_oss_key=query_image_oss_key,
        ),
        "completion_status": COMPLETION_COMPLETE,
        "exit_reason": "resolved_by_ai",
        "ready_for_manual": False,
    }
    sources = rag_result.get("sources") if isinstance(rag_result, dict) else []
    return get_service_ticket_repository().create_with_contexts(
        session_id=session_id,
        user_id=_normal_ticket_user_id(user_id),
        user_name=user_name,
        kb_name=kb_name,
        query=query,
        answer=answer,
        status=status,
        confidence=confidence,
        fallback_reason=rag_result.get("fallback_reason") if isinstance(rag_result, dict) else None,
        quality_level=rag_result.get("quality_level") if isinstance(rag_result, dict) else None,
        sources=sources,
        channel=channel or "web",
        sender_id=sender_id,
        requester_name=requester_name or user_name,
        entry_user_id=entry_user_id,
        entry_user_name=entry_user_name,
        entry_source=entry_source,
        clarification_round=1,
        clarification=clarification,
        contexts=_ticket_contexts_from_rag_result(rag_result),
    )


def record_unanswered_normal_ticket(
    *,
    session_id: str,
    user_id: str,
    user_name: Optional[str] = None,
    kb_name: Optional[str] = None,
    query: str,
    answer: str,
    confidence: Optional[float] = None,
    rag_result: Optional[Dict[str, Any]] = None,
    channel: str = "web",
    sender_id: Optional[str] = None,
    requester_name: Optional[str] = None,
    entry_user_id: Optional[str] = None,
    entry_user_name: Optional[str] = None,
    entry_source: Optional[str] = None,
    has_image: bool = False,
    query_image_oss_key: Optional[str] = None,
    query_image_oss_keys: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    if not (query or "").strip() or not (answer or "").strip():
        return None

    normalized_oss_keys = [key for key in (query_image_oss_keys or []) if str(key or "").strip()]
    if query_image_oss_key and query_image_oss_key not in normalized_oss_keys:
        normalized_oss_keys.insert(0, query_image_oss_key)
    clarification = {
        "intent_class": INTENT_NORMAL,
        "reason": "unanswered_normal_after_rag_fallback",
        "original_query": query or "",
        "required_fields": [],
        "required_field_details": [],
        "missing_fields": [],
        "collected": {
            "issue_detail": query or "",
            "has_image": bool(has_image),
            "image_keys": normalized_oss_keys,
        },
        "workflow_found": False,
        "workflow_summary": "",
        "workflow_sources": [],
        "kb_result": rag_result if isinstance(rag_result, dict) else {},
        "image_analysis": {
            "has_image": bool(has_image),
            "query_image_oss_key": normalized_oss_keys[0] if normalized_oss_keys else query_image_oss_key,
            "query_image_oss_keys": normalized_oss_keys,
        },
        "turns": [
            {
                "round": 1,
                "query": query or "",
                "answer": answer or "",
                "status": STATUS_UNANSWERED_NORMAL,
                "image_key": normalized_oss_keys[0] if normalized_oss_keys else query_image_oss_key,
                "image_keys": normalized_oss_keys,
            }
        ],
        "chat_history": _build_ticket_chat_history(
            session_id=session_id,
            existing={},
            query=query,
            answer=answer,
            has_image=has_image,
            query_image_oss_key=query_image_oss_key,
            query_image_oss_keys=query_image_oss_keys,
        ),
        "completion_status": COMPLETION_COMPLETE,
        "exit_reason": "returned_to_rag_but_unanswered",
        "ready_for_manual": False,
    }
    sources = rag_result.get("sources") if isinstance(rag_result, dict) else []
    return get_service_ticket_repository().create_with_contexts(
        session_id=session_id,
        user_id=_normal_ticket_user_id(user_id),
        user_name=user_name,
        kb_name=kb_name,
        query=query,
        answer=answer,
        status=STATUS_UNANSWERED_NORMAL,
        confidence=confidence,
        fallback_reason=rag_result.get("fallback_reason") if isinstance(rag_result, dict) else None,
        quality_level=rag_result.get("quality_level") if isinstance(rag_result, dict) else None,
        sources=sources,
        channel=channel or "web",
        sender_id=sender_id,
        requester_name=requester_name or user_name,
        entry_user_id=entry_user_id,
        entry_user_name=entry_user_name,
        entry_source=entry_source,
        clarification_round=1,
        clarification=clarification,
        contexts=_ticket_contexts_from_rag_result(rag_result),
    )


def should_start_missing_knowledge_flow(rag_result: Dict[str, Any]) -> bool:
    return _rag_result_proves_miss(rag_result)


def build_forced_missing_knowledge_decision(
    *,
    query: str,
    rag_result: Optional[Dict[str, Any]] = None,
    prior_decision: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    decision = prior_decision if isinstance(prior_decision, dict) else {}
    fields = decision.get("fields") if isinstance(decision.get("fields"), dict) else {}
    collected = dict(fields)
    issue_detail = str(collected.get("issue_detail") or query or "").strip()
    if issue_detail:
        collected["issue_detail"] = issue_detail

    missing_fields = [str(field) for field in (decision.get("missing_fields") or []) if str(field or "").strip()]
    if "manual_answer" not in missing_fields:
        missing_fields.append("manual_answer")

    fallback_reason = ""
    if isinstance(rag_result, dict):
        fallback_reason = str(rag_result.get("fallback_reason") or "")

    return {
        "intent_class": INTENT_MISSING_KNOWLEDGE,
        "reason": "knowledge_base_miss_requires_manual_followup",
        "needs_ticket_flow": True,
        "confidence": float(decision.get("confidence") or 0.0),
        "fields": collected,
        "missing_fields": missing_fields,
        "question": "",
        "ready_for_manual": True,
        "source": "fallback",
        "rationale_brief": f"rag_proven_miss:{fallback_reason or 'no_relevant_documents'}",
    }


def _manual_ticket_answer(ticket_id: Optional[str], completion_status: str) -> str:
    prefix = "当前信息未完全收集。" if completion_status == COMPLETION_INCOMPLETE else ""
    return f"{prefix}已为您记录问题，工单号 {ticket_id or '后台工单'}，后续将由专人处理。"


def _round_limit(intent_class: str) -> int:
    return MAX_CLARIFICATION_ROUNDS.get(intent_class, 3)


def _round_limit_exit_reason(intent_class: str) -> str:
    if intent_class == INTENT_AMBIGUOUS:
        return EXIT_SEMANTIC_UNRESOLVED
    return EXIT_MAX_ROUNDS


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _collected_missing_fields(required_fields: List[str], collected: Dict[str, Any]) -> List[str]:
    return [field for field in required_fields if not collected.get(field)]


def _workflow_sources(workflow: Dict[str, Any], existing: Dict[str, Any]) -> List[Any]:
    for key in ("workflow_sources", "sources"):
        sources = workflow.get(key)
        if isinstance(sources, list):
            return sources
    return _as_list(existing.get("workflow_sources"))


def _workflow_required_field_details(workflow: Dict[str, Any], existing: Dict[str, Any]) -> List[Any]:
    details = workflow.get("required_field_details")
    if isinstance(details, list):
        return details
    return _as_list(existing.get("required_field_details"))


def _normal_ticket_user_id(user_id: Optional[str]) -> str:
    return user_id or "guest_default"


def _ticket_contexts_from_rag_result(rag_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(rag_result, dict):
        return []

    contexts: List[Dict[str, Any]] = []
    seen = set()
    for idx, source in enumerate(rag_result.get("sources") or []):
        if not isinstance(source, dict):
            continue

        chunk_id = source.get("chunk_id") or source.get("id") or None
        job_id = source.get("job_id") or None
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        content = source.get("content") or source.get("text") or source.get("snippet") or ""
        file_name = source.get("file_name") or source.get("title") or metadata.get("file_name")
        chunk_index = source.get("chunk_index")
        if chunk_index is None:
            chunk_index = metadata.get("chunk_index")

        dedupe_key = chunk_id or (file_name, chunk_index, content[:80])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        context_metadata = dict(metadata)
        for key in ("title", "retrieval_source", "source", "parent_id", "parent_content", "child_index", "parent_index"):
            if key in source and key not in context_metadata:
                context_metadata[key] = source.get(key)

        contexts.append(
            {
                "chunk_id": chunk_id,
                "job_id": job_id,
                "file_name": file_name,
                "chunk_index": chunk_index,
                "score": source.get("score"),
                "content": content,
                "metadata": context_metadata,
                "sort_order": idx,
            }
        )

    backfill_chunk_ids = [
        context["chunk_id"]
        for context in contexts
        if context.get("chunk_id")
        and (
            not context.get("file_name")
            or context.get("chunk_index") is None
            or not context.get("job_id")
            or not context.get("content")
            or not (context.get("metadata") or {}).get("parent_content")
        )
    ]
    if not backfill_chunk_ids:
        return contexts

    try:
        from app.db import get_chunk_repository

        chunk_rows = get_chunk_repository().get_by_ids_with_file_names(backfill_chunk_ids)
    except Exception as exc:
        logger.warning("ticket context backfill failed: %s", exc)
        return contexts

    chunk_map = {
        str(row.get("chunk_id")): row
        for row in chunk_rows
        if isinstance(row, dict) and row.get("chunk_id")
    }
    for context in contexts:
        chunk_id = context.get("chunk_id")
        if not chunk_id:
            continue
        chunk_row = chunk_map.get(str(chunk_id))
        if not isinstance(chunk_row, dict):
            continue

        row_metadata = chunk_row.get("metadata") if isinstance(chunk_row.get("metadata"), dict) else {}
        context_metadata = context.get("metadata") if isinstance(context.get("metadata"), dict) else {}
        merged_metadata = dict(row_metadata)
        merged_metadata.update(context_metadata)

        if not context.get("job_id"):
            context["job_id"] = chunk_row.get("job_id")
        if not context.get("file_name"):
            context["file_name"] = chunk_row.get("file_name") or merged_metadata.get("file_name")
        if context.get("chunk_index") is None:
            context["chunk_index"] = chunk_row.get("chunk_index")
        if not context.get("content"):
            context["content"] = (
                chunk_row.get("content")
                or chunk_row.get("current_content")
                or chunk_row.get("original_content")
                or ""
            )
        context["metadata"] = merged_metadata
    return contexts


def _compact_chat_sources(sources: Any) -> List[Dict[str, Any]]:
    if not isinstance(sources, list):
        return []

    compacted = []
    for source in sources[:5]:
        if not isinstance(source, dict):
            continue
        item = {}
        for key in ("file_name", "title", "chunk_id", "chunk_index", "score"):
            value = source.get(key)
            if value is not None:
                item[key] = value
        content = source.get("content") or source.get("text") or source.get("snippet")
        if content:
            item["content_preview"] = str(content)[:240]
        if item:
            compacted.append(item)
    return compacted


def _chat_history_item(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(message, dict):
        return None

    role = str(message.get("role") or "").strip() or "user"
    content = str(message.get("content") or "")
    item: Dict[str, Any] = {"role": role, "content": content}

    for key in ("created_at", "query_image_oss_key", "fallback_reason", "quality_level"):
        value = message.get(key)
        if value not in (None, "", [], {}):
            item[key] = value

    for key in ("confidence", "used_fallback", "quality_passed"):
        value = message.get(key)
        if value is not None:
            item[key] = value

    image_placeholders = message.get("image_placeholders")
    if isinstance(image_placeholders, list) and image_placeholders:
        item["image_placeholders"] = image_placeholders[:10]

    sources = _compact_chat_sources(message.get("sources"))
    if sources:
        item["sources"] = sources

    if message.get("has_image"):
        item["has_image"] = True

    return item


def _existing_chat_history(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    history = []
    for item in value[-CHAT_HISTORY_LIMIT:]:
        normalized = _chat_history_item(item)
        if normalized:
            history.append(normalized)
    return history


def _chat_history_from_turns(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    history = []
    for turn in value[-CHAT_HISTORY_LIMIT:]:
        if not isinstance(turn, dict):
            continue
        query = turn.get("query")
        answer = turn.get("answer")
        image_key = turn.get("image_key")
        if query is not None or image_key:
            user_item = {"role": "user", "content": str(query or "")}
            if image_key:
                user_item["query_image_oss_key"] = image_key
                user_item["has_image"] = True
            _append_chat_item(history, user_item)
        if answer is not None:
            _append_chat_item(history, {"role": "assistant", "content": str(answer or "")})
    return history[-CHAT_HISTORY_LIMIT:]


def _history_from_active_ticket(active_ticket: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clarification = _active_ticket_clarification(active_ticket)
    history = _existing_chat_history(clarification.get("chat_history"))
    if not history:
        history = _chat_history_from_turns(clarification.get("turns"))
    return history


def load_ticket_history_context(
    session_id: str,
    active_ticket: Optional[Dict[str, Any]] = None,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for item in _load_session_chat_history(session_id):
        _append_chat_item(history, dict(item))
    for item in _history_from_active_ticket(active_ticket):
        _append_chat_item(history, dict(item))
    return history[-max(int(limit or 0), 0):] if limit else history


def _merged_ticket_image_analysis(
    existing: Dict[str, Any],
    decision: Dict[str, Any],
    *,
    has_image: bool,
    query_image_oss_key: Optional[str],
    query_image_oss_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    analysis = dict(existing.get("image_analysis") if isinstance(existing.get("image_analysis"), dict) else {})
    if has_image:
        analysis["has_image"] = True
    elif "has_image" not in analysis:
        analysis["has_image"] = False

    normalized_oss_keys = [key for key in (query_image_oss_keys or []) if str(key or "").strip()]
    if query_image_oss_key and query_image_oss_key not in normalized_oss_keys:
        normalized_oss_keys.insert(0, query_image_oss_key)

    if normalized_oss_keys:
        image_keys = list(analysis.get("image_keys") or [])
        for key in normalized_oss_keys:
            if key not in image_keys:
                image_keys.append(key)
        analysis["image_keys"] = image_keys
        analysis["query_image_oss_key"] = normalized_oss_keys[0]
        analysis["query_image_oss_keys"] = normalized_oss_keys

    field_analysis = decision.get("image_analysis") if isinstance(decision.get("image_analysis"), dict) else None
    if field_analysis:
        analysis["field_analysis"] = field_analysis
        field_analyses = list(analysis.get("field_analyses") or [])
        field_analyses.append(field_analysis)
        analysis["field_analyses"] = field_analyses[-10:]

    return analysis


def _load_session_chat_history(session_id: str) -> List[Dict[str, Any]]:
    if not session_id:
        return []
    try:
        from app.db import get_conversation_repository

        messages = get_conversation_repository().list_messages(session_id, limit=CHAT_HISTORY_LIMIT)
    except Exception as exc:
        logger.debug("load conversation chat history failed for service ticket: %s", exc)
        return []

    history = []
    for message in messages:
        item = _chat_history_item(message)
        if item:
            history.append(item)
    return history[-CHAT_HISTORY_LIMIT:]


def _chat_signature(item: Dict[str, Any]) -> tuple:
    return (
        str(item.get("role") or ""),
        str(item.get("content") or ""),
        str(item.get("query_image_oss_key") or ""),
    )


def _find_recent_chat_item(history: List[Dict[str, Any]], target: Dict[str, Any], max_scan: int = 8) -> int:
    target_signature = _chat_signature(target)
    start = max(0, len(history) - max_scan)
    for idx in range(len(history) - 1, start - 1, -1):
        if _chat_signature(history[idx]) == target_signature:
            return idx
    return -1


def _append_chat_item(history: List[Dict[str, Any]], item: Optional[Dict[str, Any]]) -> None:
    if not item:
        return
    if history and _chat_signature(history[-1]) == _chat_signature(item):
        history[-1].update({key: value for key, value in item.items() if value not in (None, "", [], {})})
        return
    if _find_recent_chat_item(history, item) >= 0:
        return
    history.append(item)
    del history[:-CHAT_HISTORY_LIMIT]


def _find_assistant_after(history: List[Dict[str, Any]], user_idx: int) -> int:
    if user_idx < 0:
        return -1
    for idx in range(user_idx + 1, min(len(history), user_idx + 4)):
        if history[idx].get("role") == "assistant":
            return idx
    return -1


def _append_current_turn_to_history(
    history: List[Dict[str, Any]],
    *,
    query: str,
    answer: str,
    has_image: bool,
    query_image_oss_key: Optional[str],
    query_image_oss_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    user_item = {"role": "user", "content": query or ""}
    normalized_oss_keys = [key for key in (query_image_oss_keys or []) if str(key or "").strip()]
    if query_image_oss_key and query_image_oss_key not in normalized_oss_keys:
        normalized_oss_keys.insert(0, query_image_oss_key)
    if has_image or query_image_oss_key:
        user_item["has_image"] = True
    if normalized_oss_keys:
        user_item["query_image_oss_key"] = normalized_oss_keys[0]
        user_item["query_image_oss_keys"] = normalized_oss_keys

    user_idx = _find_recent_chat_item(history, user_item)
    if user_idx < 0:
        _append_chat_item(history, user_item)
        user_idx = len(history) - 1

    assistant_item = {"role": "assistant", "content": answer or ""}
    assistant_idx = _find_assistant_after(history, user_idx)
    if assistant_idx >= 0:
        history[assistant_idx].update(assistant_item)
    else:
        _append_chat_item(history, assistant_item)

    return history[-CHAT_HISTORY_LIMIT:]


def _build_ticket_chat_history(
    *,
    session_id: str,
    existing: Dict[str, Any],
    query: str,
    answer: str,
    has_image: bool,
    query_image_oss_key: Optional[str],
    query_image_oss_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    existing_history = _existing_chat_history(existing.get("chat_history"))
    if not existing_history:
        existing_history = _chat_history_from_turns(existing.get("turns"))
    session_history = _load_session_chat_history(session_id)
    history = existing_history if len(existing_history) >= len(session_history) else session_history
    history = [dict(item) for item in history]
    return _append_current_turn_to_history(
        history,
        query=query,
        answer=answer,
        has_image=has_image,
        query_image_oss_key=query_image_oss_key,
        query_image_oss_keys=query_image_oss_keys,
    )


def process_ticket_clarification_turn(
    *,
    session_id: str,
    user_id: str,
    user_name: Optional[str],
    sender_id: Optional[str],
    requester_name: Optional[str],
    kb_name: Optional[str],
    query: str,
    channel: str,
    decision: Dict[str, Any],
    workflow: Optional[Dict[str, Any]] = None,
    rag_result: Optional[Dict[str, Any]] = None,
    has_image: bool = False,
    query_image_oss_key: Optional[str] = None,
    query_image_oss_keys: Optional[List[str]] = None,
    continue_only: bool = False,
    entry_user_id: Optional[str] = None,
    entry_user_name: Optional[str] = None,
    entry_source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    repo = get_service_ticket_repository()
    locked_user_id = _normal_ticket_user_id(user_id)

    def _run(active: Optional[Dict[str, Any]], conn) -> Optional[Dict[str, Any]]:
        if continue_only and not active:
            return None

        existing = active.get("clarification") if active else {}
        existing = existing if isinstance(existing, dict) else {}
        workflow_payload = workflow or {}
        decision_fields = decision.get("fields") if isinstance(decision, dict) else {}
        decision_fields = decision_fields if isinstance(decision_fields, dict) else {}

        existing_intent = str(existing.get("intent_class") or "").upper()
        incoming_intent = str(decision.get("intent_class") or "").upper()
        if existing_intent == INTENT_AMBIGUOUS and incoming_intent in {INTENT_OPERATION, INTENT_MISSING_KNOWLEDGE}:
            intent_class = incoming_intent
        else:
            intent_class = str(existing.get("intent_class") or decision.get("intent_class") or INTENT_AMBIGUOUS).upper()
        if intent_class not in _VALID_INTENTS:
            intent_class = INTENT_AMBIGUOUS
        reason = str(existing.get("reason") or decision.get("reason") or "")
        original_query = str(existing.get("original_query") or query or "")

        turns = list(existing.get("turns") or [])
        current_round = int(active.get("clarification_round") or 0) if active else 0
        round_no = max(current_round, len(turns)) + 1

        collected = dict(existing.get("collected") or {})
        collected.update(decision_fields)
        if intent_class in {INTENT_OPERATION, INTENT_MISSING_KNOWLEDGE, INTENT_AMBIGUOUS} and not collected.get("issue_detail"):
            detail = original_query or query or ""
            if detail:
                collected["issue_detail"] = detail
        if has_image:
            collected["has_image"] = True
            normalized_oss_keys = [key for key in (query_image_oss_keys or []) if str(key or "").strip()]
            if query_image_oss_key and query_image_oss_key not in normalized_oss_keys:
                normalized_oss_keys.insert(0, query_image_oss_key)
            if normalized_oss_keys:
                image_keys = list(collected.get("image_keys") or [])
                for key in normalized_oss_keys:
                    if key not in image_keys:
                        image_keys.append(key)
                collected["image_keys"] = image_keys

        workflow_found = bool(
            workflow_payload.get("workflow_found")
            if "workflow_found" in workflow_payload
            else existing.get("workflow_found")
        )
        required_fields = (
            _as_list(workflow_payload.get("required_fields"))
            or _as_list(existing.get("required_fields"))
            or _as_list(decision.get("missing_fields"))
            or ["issue_detail"]
        )
        missing_fields = _collected_missing_fields(required_fields, collected)

        workflow_source = str(workflow_payload.get("source") or "")
        workflow_reason = str(workflow_payload.get("rationale_brief") or "")
        workflow_analysis_failed = workflow_source == "fallback" and workflow_reason not in {"no_knowledge_candidates"}
        operation_fields_complete = intent_class == INTENT_OPERATION and workflow_found and not missing_fields
        round_limit_reached = round_no >= _round_limit(intent_class)
        forced_manual = bool(decision.get("ready_for_manual") or decision.get("user_refused"))
        should_finalize = operation_fields_complete or round_limit_reached or forced_manual

        if operation_fields_complete:
            exit_reason = EXIT_FIELDS_COMPLETE
            completion_status = COMPLETION_COMPLETE
        elif forced_manual:
            exit_reason = "user_refused" if decision.get("user_refused") else "ready_for_manual"
            completion_status = COMPLETION_INCOMPLETE if missing_fields or decision.get("user_refused") else COMPLETION_COMPLETE
        elif round_limit_reached:
            exit_reason = _round_limit_exit_reason(intent_class)
            completion_status = COMPLETION_INCOMPLETE
        else:
            exit_reason = ""
            completion_status = existing.get("completion_status") or COMPLETION_INCOMPLETE

        status = "pending_manual" if should_finalize else "clarifying"
        answer_ticket_id = active.get("id") if active else None
        clarification_question = (
            workflow_payload.get("question")
            or decision.get("question")
            or _default_clarification_question(
                intent_class=intent_class,
                missing_fields=missing_fields,
                round_no=round_no,
                has_image=has_image,
            )
        )
        answer = (
            _manual_ticket_answer(answer_ticket_id, completion_status)
            if should_finalize
            else clarification_question
        )

        turns.append(
            {
                "round": round_no,
                "query": query or "",
                "answer": answer,
                "status": status,
                "image_key": query_image_oss_key,
                "image_keys": [key for key in (query_image_oss_keys or []) if str(key or "").strip()]
                or ([query_image_oss_key] if query_image_oss_key else []),
            }
        )
        chat_history = _build_ticket_chat_history(
            session_id=session_id,
            existing=existing,
            query=query,
            answer=answer,
            has_image=has_image,
            query_image_oss_key=query_image_oss_key,
            query_image_oss_keys=query_image_oss_keys,
        )

        clarification = {
            "intent_class": intent_class,
            "reason": reason,
            "original_query": original_query,
            "required_fields": required_fields,
            "required_field_details": _workflow_required_field_details(workflow_payload, existing),
            "missing_fields": missing_fields,
            "collected": collected,
            "workflow_found": workflow_found,
            "workflow_summary": workflow_payload.get("workflow_summary") or existing.get("workflow_summary") or "",
            "workflow_confidence": workflow_payload.get("confidence") if "confidence" in workflow_payload else existing.get("workflow_confidence"),
            "workflow_rationale": workflow_payload.get("rationale_brief") or existing.get("workflow_rationale") or "",
            "workflow_source": workflow_payload.get("source") or existing.get("workflow_source") or "",
            "workflow_sources": _workflow_sources(workflow_payload, existing),
            "kb_result": existing.get("kb_result") or (rag_result if isinstance(rag_result, dict) else {}),
            "image_analysis": _merged_ticket_image_analysis(
                existing,
                decision,
                has_image=has_image,
                query_image_oss_key=query_image_oss_key,
                query_image_oss_keys=query_image_oss_keys,
            ),
            "turns": turns,
            "chat_history": chat_history,
            "completion_status": completion_status,
            "exit_reason": exit_reason,
            "ready_for_manual": should_finalize,
        }

        common_update = {
            "status": status,
            "answer": answer,
            "clarification_round": round_no,
            "clarification": clarification,
            "sender_id": sender_id,
            "requester_name": requester_name or user_name,
            "conn": conn,
        }
        context_snapshots = _ticket_contexts_from_rag_result(rag_result)
        if active:
            ticket = repo.update_clarification(active["id"], **common_update)
            if context_snapshots:
                repo.replace_contexts(active["id"], context_snapshots, conn=conn)
        else:
            ticket = repo.create_with_contexts(
                session_id=session_id,
                user_id=locked_user_id,
                user_name=user_name,
                kb_name=kb_name,
                query=original_query,
                answer=answer,
                status=status,
                confidence=0.0,
                fallback_reason=reason,
                quality_level="clarifying",
                sources=(rag_result.get("sources") if isinstance(rag_result, dict) else []),
                channel=channel or "web",
                sender_id=sender_id,
                requester_name=requester_name or user_name,
                entry_user_id=entry_user_id,
                entry_user_name=entry_user_name,
                entry_source=entry_source,
                clarification_round=round_no,
                clarification=clarification,
                contexts=context_snapshots,
                conn=conn,
            )

        if ticket and should_finalize and not answer_ticket_id:
            answer = _manual_ticket_answer(ticket.get("id"), completion_status)
            turns[-1]["answer"] = answer
            clarification["turns"] = turns
            clarification["chat_history"] = _append_current_turn_to_history(
                list(clarification.get("chat_history") or []),
                query=query,
                answer=answer,
                has_image=has_image,
                query_image_oss_key=query_image_oss_key,
                query_image_oss_keys=query_image_oss_keys,
            )
            ticket = repo.update_clarification(
                ticket["id"],
                status=status,
                answer=answer,
                clarification_round=round_no,
                clarification=clarification,
                sender_id=sender_id,
                requester_name=requester_name or user_name,
                conn=conn,
            )

        return {
            "ticket_id": ticket.get("id") if ticket else None,
            "status": status,
            "answer": answer,
            "clarification_round": round_no,
            "clarification": clarification,
            "confidence": 0.0,
            "finish_reason": "manual_ticket_created" if should_finalize else "clarification",
        }

    return repo.run_clarification_transaction(
        session_id=session_id,
        user_id=locked_user_id,
        kb_name=kb_name,
        callback=_run,
    )


process_clarification_turn = process_ticket_clarification_turn
