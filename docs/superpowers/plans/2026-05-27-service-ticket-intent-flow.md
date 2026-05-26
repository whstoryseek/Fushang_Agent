# LLM Service Ticket Intent Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM-only service-ticket intent flow that collects A/B/C ticket information while preserving normal RAG latency and response behavior for D-class questions.

**Architecture:** `backend/app/services/service_ticket_service.py` owns the mini-LLM classifier, canonical ticket state, field collection, and manual-ticket finalization. `backend/app/api/v1/knowledge.py` orchestrates active ticket recovery and races the pre-RAG classifier with the already-started RAG task. `backend/app/services/knowledge_service.py` exposes raw RAG fallback metadata and an optional streaming B conversion branch without changing default callers.

**Tech Stack:** Python 3.10, FastAPI, asyncio, unittest/pytest, existing `LLMService.responses_text()` with thinking disabled, LangGraph RAG services, PostgreSQL-backed service-ticket repository.

---

## File Structure

- Create or modify `backend/app/services/service_ticket_service.py`
  - LLM-only intent classifier using `doubao-seed-2-0-mini-260428`.
  - JSON normalization and safe fallback behavior.
  - Canonical A/B/C clarification state machine.
  - B-class post-RAG eligibility gate.

- Modify `backend/app/services/knowledge_service.py`
  - Add `persist: bool = True` to `invoke_knowledge_qa()`.
  - Return raw fallback fields needed by the B gate.
  - Add optional stream B conversion after final RAG state, default off.

- Modify `backend/app/api/v1/knowledge.py`
  - Check active clarification first.
  - For normal text, start RAG immediately and race it with the mini-classifier interception deadline.
  - Keep D-class response shape unchanged.
  - Enable post-RAG B conversion for non-stream and stream endpoints.

- Create or modify tests:
  - `backend/tests/test_service_ticket_service.py`
  - `backend/tests/test_knowledge_service_tickets.py`
  - `backend/tests/test_knowledge_api_llm_clarification.py`
  - `backend/tests/test_knowledge_stream_ticket_flow.py`

---

### Task 1: LLM Intent Classifier Contract

**Files:**
- Create: `backend/app/services/service_ticket_service.py`
- Create/modify: `backend/tests/test_service_ticket_service.py`

- [ ] **Step 1: Write classifier tests**

Create `backend/tests/test_service_ticket_service.py` if it does not exist. Include lightweight `psycopg2` stubs at the top because local test environments often lack that package:

```python
# -*- coding: utf-8 -*-
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

if "psycopg2" not in sys.modules:
    psycopg2_module = types.ModuleType("psycopg2")
    psycopg2_module.pool = types.SimpleNamespace(ThreadedConnectionPool=object)
    psycopg2_module.extras = types.SimpleNamespace(RealDictCursor=object)
    sys.modules["psycopg2"] = psycopg2_module
    sys.modules["psycopg2.pool"] = psycopg2_module.pool
    sys.modules["psycopg2.extras"] = psycopg2_module.extras

from app.services.service_ticket_service import (
    INTENT_AMBIGUOUS,
    INTENT_MISSING_KNOWLEDGE,
    INTENT_NORMAL,
    INTENT_OPERATION,
    classify_ticket_intent_with_llm,
)


class ServiceTicketClassifierTests(unittest.TestCase):
    @patch("app.services.service_ticket_service.get_llm_service")
    def test_classifier_uses_mini_model_with_thinking_disabled_path(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = json.dumps(
            {
                "intent_class": "A",
                "reason": "operation_required",
                "needs_ticket_flow": True,
                "confidence": 0.91,
                "fields": {"issue_detail": "绑定富友账户"},
                "missing_fields": ["phone"],
                "question": "请补充可联系手机号。",
                "rationale_brief": "用户要求代办具体动作",
            },
            ensure_ascii=False,
        )
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(
            query="帮我绑定一下富友账户",
            history=[],
            has_image=False,
        )

        self.assertEqual(result["intent_class"], INTENT_OPERATION)
        self.assertTrue(result["needs_ticket_flow"])
        self.assertEqual(result["missing_fields"], ["phone"])
        llm.responses_text.assert_called_once()
        kwargs = llm.responses_text.call_args.kwargs
        self.assertEqual(kwargs["model"], "doubao-seed-2-0-mini-260428")
        self.assertLessEqual(kwargs["max_tokens"], 300)
        self.assertLessEqual(kwargs["timeout"], 1.2)
        self.assertEqual(kwargs["max_retries"], 0)

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_classifier_accepts_llm_d_for_tutorial_question(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = json.dumps(
            {
                "intent_class": "D",
                "reason": "knowledge_question",
                "needs_ticket_flow": False,
                "confidence": 0.88,
                "fields": {},
                "missing_fields": [],
                "question": "",
                "rationale_brief": "询问教程，不是代办",
            },
            ensure_ascii=False,
        )
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(
            query="富友账户怎么绑定",
            history=[],
            has_image=False,
        )

        self.assertEqual(result["intent_class"], INTENT_NORMAL)
        self.assertFalse(result["needs_ticket_flow"])

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_malformed_classifier_json_defaults_text_to_d(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = "不是 JSON"
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(
            query="富友账户怎么绑定",
            history=[],
            has_image=False,
        )

        self.assertEqual(result["intent_class"], INTENT_NORMAL)
        self.assertFalse(result["needs_ticket_flow"])
        self.assertEqual(result["source"], "fallback")

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_classifier_failure_defaults_image_only_to_c(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.side_effect = RuntimeError("llm timeout")
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(
            query="",
            history=[],
            has_image=True,
        )

        self.assertEqual(result["intent_class"], INTENT_AMBIGUOUS)
        self.assertTrue(result["needs_ticket_flow"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/test_service_ticket_service.py -q
```

Expected: import failure because `service_ticket_service.py` is not implemented yet.

- [ ] **Step 3: Implement classifier**

Create `backend/app/services/service_ticket_service.py` with the classifier contract:

```python
# -*- coding: utf-8 -*-
import json
import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

INTENT_OPERATION = "A"
INTENT_MISSING_KNOWLEDGE = "B"
INTENT_AMBIGUOUS = "C"
INTENT_NORMAL = "D"

REASON_OPERATION = "operation_required"
REASON_MISSING_KNOWLEDGE = "knowledge_missing"
REASON_AMBIGUOUS = "ambiguous_query"
REASON_KNOWLEDGE = "knowledge_question"

CLASSIFIER_MODEL = "doubao-seed-2-0-mini-260428"
CLASSIFIER_MAX_TOKENS = 300
CLASSIFIER_TIMEOUT = min(float(getattr(settings, "operation_classifier_timeout", 1.2)), 1.2)


def _default_decision(*, has_image: bool) -> Dict[str, Any]:
    if has_image:
        return {
            "intent_class": INTENT_AMBIGUOUS,
            "reason": REASON_AMBIGUOUS,
            "needs_ticket_flow": True,
            "confidence": 0.0,
            "fields": {},
            "missing_fields": ["issue_detail"],
            "question": "请补充说明截图对应的业务场景、操作步骤或报错信息。",
            "rationale_brief": "image_only_or_classifier_failed",
            "source": "fallback",
        }
    return {
        "intent_class": INTENT_NORMAL,
        "reason": REASON_KNOWLEDGE,
        "needs_ticket_flow": False,
        "confidence": 0.0,
        "fields": {},
        "missing_fields": [],
        "question": "",
        "rationale_brief": "classifier_failed",
        "source": "fallback",
    }


def _extract_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    return json.loads(raw)


def _normalize_classifier_payload(payload: Dict[str, Any], *, has_image: bool) -> Dict[str, Any]:
    intent = str(payload.get("intent_class") or "").upper()
    if intent not in {INTENT_OPERATION, INTENT_MISSING_KNOWLEDGE, INTENT_AMBIGUOUS, INTENT_NORMAL}:
        return _default_decision(has_image=has_image)
    needs_ticket = bool(payload.get("needs_ticket_flow"))
    if intent == INTENT_NORMAL:
        needs_ticket = False
    if intent in {INTENT_OPERATION, INTENT_MISSING_KNOWLEDGE, INTENT_AMBIGUOUS}:
        needs_ticket = True
    return {
        "intent_class": intent,
        "reason": payload.get("reason") or REASON_KNOWLEDGE,
        "needs_ticket_flow": needs_ticket,
        "confidence": float(payload.get("confidence") or 0.0),
        "fields": payload.get("fields") if isinstance(payload.get("fields"), dict) else {},
        "missing_fields": payload.get("missing_fields") if isinstance(payload.get("missing_fields"), list) else [],
        "question": str(payload.get("question") or ""),
        "rationale_brief": str(payload.get("rationale_brief") or ""),
        "source": "llm",
    }


def _classifier_prompt(query: str, history: List[Dict[str, str]], has_image: bool, rag_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    history_lines = []
    for turn in (history or [])[-4:]:
        role = turn.get("role", "user")
        content = str(turn.get("content") or "")[:200]
        history_lines.append(f"{role}: {content}")
    rag_hint = ""
    if rag_result:
        rag_hint = json.dumps(
            {
                "used_fallback": rag_result.get("used_fallback"),
                "fallback_reason": rag_result.get("fallback_reason"),
                "quality_passed": rag_result.get("quality_passed"),
                "confidence": rag_result.get("confidence"),
                "source_count": len(rag_result.get("sources") or []),
            },
            ensure_ascii=False,
        )
    return [
        {
            "role": "system",
            "content": (
                "你是工单意图分类器，只输出 JSON。"
                "A=用户要求代办具体动作；B=RAG 已确认知识库未覆盖；"
                "C=语义不明或仅图片；D=正常知识库问题。"
                "不要把怎么/如何/能否/有什么用等教程咨询误判为 A。"
                "输出字段：intent_class, reason, needs_ticket_flow, confidence, "
                "fields, missing_fields, question, rationale_brief。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": query or "",
                    "has_image": has_image,
                    "history": history_lines,
                    "rag_result": rag_hint,
                },
                ensure_ascii=False,
            ),
        },
    ]


def classify_ticket_intent_with_llm(
    *,
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    has_image: bool = False,
    rag_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        text = get_llm_service().responses_text(
            _classifier_prompt(query, history or [], has_image, rag_result),
            model=CLASSIFIER_MODEL,
            temperature=0.0,
            max_tokens=CLASSIFIER_MAX_TOKENS,
            timeout=CLASSIFIER_TIMEOUT,
            max_retries=0,
        )
        return _normalize_classifier_payload(_extract_json(text), has_image=has_image)
    except Exception as exc:
        logger.warning("ticket intent classifier failed: %s", exc)
        return _default_decision(has_image=has_image)
```

- [ ] **Step 4: Run tests and verify GREEN**

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/test_service_ticket_service.py -q
$env:PYTHONPATH='backend'; python -m py_compile backend/app/services/service_ticket_service.py backend/tests/test_service_ticket_service.py
```

Expected: all tests in `test_service_ticket_service.py` pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/service_ticket_service.py backend/tests/test_service_ticket_service.py
git commit -m "feat: add llm ticket intent classifier"
```

---

### Task 2: Canonical Ticket State Machine

**Files:**
- Modify: `backend/app/services/service_ticket_service.py`
- Modify: `backend/tests/test_service_ticket_service.py`

- [ ] **Step 1: Add state-machine tests**

Append these tests to `backend/tests/test_service_ticket_service.py`:

```python
class FakeClarificationRepo:
    def __init__(self):
        self.ticket = None

    def find_active_clarification(self, **kwargs):
        return self.ticket

    def create_with_contexts(self, **kwargs):
        self.ticket = {
            "id": "ticket-1",
            **kwargs,
            "clarification_round": kwargs.get("clarification_round", 0),
            "clarification": kwargs.get("clarification") or {},
            "contexts": kwargs.get("contexts") or [],
        }
        return self.ticket

    def update_clarification(self, ticket_id, **kwargs):
        self.ticket.update(kwargs)
        return self.ticket


class ServiceTicketStateTests(unittest.TestCase):
    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_operation_without_sop_finalizes_manual_ticket(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo
        decision = {
            "intent_class": "A",
            "reason": "operation_required",
            "needs_ticket_flow": True,
            "fields": {"issue_detail": "绑定富友账户"},
            "missing_fields": [],
            "question": "",
            "source": "llm",
        }

        result = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="帮我绑定富友账户",
            channel="h5",
            decision=decision,
            workflow={"workflow_found": False, "required_fields": []},
        )

        self.assertEqual(result["status"], "pending_manual")
        self.assertEqual(result["finish_reason"], "manual_ticket_created")
        self.assertEqual(repo.ticket["clarification"]["intent_class"], "A")
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "no_standard_workflow")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_ambiguous_flow_finalizes_after_three_rounds(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo
        decision = {
            "intent_class": "C",
            "reason": "ambiguous_query",
            "needs_ticket_flow": True,
            "fields": {},
            "missing_fields": ["issue_detail"],
            "question": "请补充具体场景。",
            "source": "llm",
        }

        first = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="这个问题",
            channel="h5",
            decision=decision,
        )
        second = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="还是不清楚",
            channel="h5",
            decision=decision,
            continue_only=True,
        )
        third = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="就是那个",
            channel="h5",
            decision=decision,
            continue_only=True,
        )

        self.assertEqual(first["status"], "clarifying")
        self.assertEqual(second["status"], "clarifying")
        self.assertEqual(third["status"], "pending_manual")
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "semantic_unresolved")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_missing_knowledge_finalizes_after_five_rounds(self, mock_repo):
        repo = FakeClarificationRepo()
        repo.ticket = {
            "id": "ticket-1",
            "session_id": "session-1",
            "user_id": "store-1",
            "kb_name": "fushang",
            "status": "clarifying",
            "clarification_round": 4,
            "clarification": {
                "intent_class": "B",
                "reason": "knowledge_missing",
                "original_query": "知识库没有的新问题",
                "required_fields": ["issue_detail", "phone"],
                "missing_fields": ["phone"],
                "collected": {"issue_detail": "知识库没有的新问题"},
                "turns": [{"round": i, "query": f"turn-{i}"} for i in range(1, 5)],
                "kb_result": {"hit": False},
            },
        }
        mock_repo.return_value = repo
        decision = {
            "intent_class": "B",
            "reason": "knowledge_missing",
            "needs_ticket_flow": True,
            "fields": {"phone": "13800138000"},
            "missing_fields": [],
            "question": "",
            "source": "llm",
        }

        result = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="手机号 13800138000",
            channel="h5",
            decision=decision,
            continue_only=True,
        )

        self.assertEqual(result["status"], "pending_manual")
        self.assertEqual(result["finish_reason"], "manual_ticket_created")
        self.assertEqual(repo.ticket["clarification_round"], 5)
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "max_rounds")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "incomplete")
        self.assertEqual(repo.ticket["clarification"]["collected"]["phone"], "13800138000")
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/test_service_ticket_service.py -q
```

Expected: failures for missing `process_ticket_clarification_turn`.

- [ ] **Step 3: Implement canonical state machine**

Append to `backend/app/services/service_ticket_service.py`:

```python
MAX_ROUNDS = {INTENT_OPERATION: 20, INTENT_MISSING_KNOWLEDGE: 5, INTENT_AMBIGUOUS: 3}
EXIT_FIELDS_COMPLETE = "fields_complete"
EXIT_NO_STANDARD_WORKFLOW = "no_standard_workflow"
EXIT_MAX_ROUNDS = "max_rounds"
EXIT_SEMANTIC_UNRESOLVED = "semantic_unresolved"
EXIT_USER_REFUSED = "user_refused_or_unknown"
COMPLETION_COMPLETE = "complete"
COMPLETION_INCOMPLETE = "incomplete"


def _manual_ticket_answer(ticket_id: Optional[str], completion_status: str) -> str:
    prefix = "当前信息未完全收集。" if completion_status == COMPLETION_INCOMPLETE else ""
    return f"{prefix}已为您记录问题，工单号 {ticket_id or '后台工单'}，后续将由专人处理。"


def _missing_fields(required: List[str], collected: Dict[str, Any]) -> List[str]:
    return [field for field in required if not collected.get(field)]


def _round_cap(intent_class: str) -> int:
    return MAX_ROUNDS.get(intent_class, 3)


def _round_cap_exit(intent_class: str) -> str:
    return EXIT_SEMANTIC_UNRESOLVED if intent_class == INTENT_AMBIGUOUS else EXIT_MAX_ROUNDS


def should_start_missing_knowledge_flow(rag_result: Dict[str, Any]) -> bool:
    if not isinstance(rag_result, dict):
        return False
    if not rag_result.get("used_fallback"):
        return False
    fallback_reason = str(rag_result.get("fallback_reason") or "").lower()
    no_relevant = "no_relevant" in fallback_reason or "no relevant" in fallback_reason
    return bool(no_relevant or rag_result.get("quality_passed") is False)


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
    continue_only: bool = False,
) -> Optional[Dict[str, Any]]:
    from app.db import get_service_ticket_repository

    repo = get_service_ticket_repository()
    active = repo.find_active_clarification(
        session_id=session_id,
        user_id=user_id or "guest_default",
        kb_name=kb_name,
    )
    if continue_only and not active:
        return None

    existing = active.get("clarification") if active else {}
    intent_class = existing.get("intent_class") or decision.get("intent_class") or INTENT_AMBIGUOUS
    reason = existing.get("reason") or decision.get("reason") or REASON_AMBIGUOUS
    original_query = existing.get("original_query") or query
    turns = list(existing.get("turns") or [])
    round_no = max(int(active.get("clarification_round") or 0) if active else 0, len(turns)) + 1
    workflow = workflow or {}
    workflow_found = bool(workflow.get("workflow_found") or existing.get("workflow_found"))
    collected = {**(existing.get("collected") or {}), **(decision.get("fields") or {})}
    if has_image:
        collected["has_image"] = True
        if query_image_oss_key:
            collected.setdefault("image_keys", []).append(query_image_oss_key)
    required = (
        workflow.get("required_fields")
        or existing.get("required_fields")
        or decision.get("missing_fields")
        or ["issue_detail"]
    )
    missing = _missing_fields(required, collected)
    no_sop = intent_class == INTENT_OPERATION and workflow is not None and not workflow_found
    fields_complete = intent_class == INTENT_OPERATION and not missing
    round_cap_reached = round_no >= _round_cap(intent_class)
    refused = reason == EXIT_USER_REFUSED or decision.get("refusal") is True
    should_finalize = refused or no_sop or fields_complete or round_cap_reached
    status = "pending_manual" if should_finalize else "clarifying"

    if refused:
        exit_reason = EXIT_USER_REFUSED
        completion_status = COMPLETION_INCOMPLETE
    elif no_sop:
        exit_reason = EXIT_NO_STANDARD_WORKFLOW
        completion_status = COMPLETION_INCOMPLETE
    elif fields_complete:
        exit_reason = EXIT_FIELDS_COMPLETE
        completion_status = COMPLETION_COMPLETE
    elif round_cap_reached:
        exit_reason = _round_cap_exit(intent_class)
        completion_status = COMPLETION_INCOMPLETE
    else:
        exit_reason = ""
        completion_status = existing.get("completion_status") or COMPLETION_INCOMPLETE

    answer = (
        _manual_ticket_answer(active.get("id") if active else None, completion_status)
        if should_finalize
        else decision.get("question") or "请补充具体业务场景、操作步骤或报错信息。"
    )
    turns.append({"round": round_no, "query": query, "answer": answer, "status": status, "image_key": query_image_oss_key})
    clarification = {
        "intent_class": intent_class,
        "reason": reason,
        "original_query": original_query,
        "required_fields": required,
        "missing_fields": missing,
        "collected": collected,
        "workflow_found": workflow_found,
        "workflow_summary": workflow.get("workflow_summary") or existing.get("workflow_summary") or "",
        "workflow_sources": existing.get("workflow_sources") or [],
        "kb_result": existing.get("kb_result") or (rag_result or {}),
        "image_analysis": existing.get("image_analysis") or {"has_image": has_image, "categories": []},
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
            fallback_reason=reason,
            quality_level="clarifying",
            sources=[],
            channel=channel or "web",
            sender_id=sender_id,
            requester_name=requester_name or user_name,
            clarification_round=round_no,
            clarification=clarification,
            contexts=[],
        )
    if ticket and status == "pending_manual" and "后台工单" in answer:
        answer = _manual_ticket_answer(ticket.get("id"), completion_status)
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
        "ticket_id": ticket.get("id") if ticket else None,
        "status": status,
        "answer": answer,
        "clarification_round": round_no,
        "clarification": clarification,
        "confidence": 0.0,
        "finish_reason": "manual_ticket_created" if should_finalize else "clarification",
    }
```

- [ ] **Step 4: Run tests**

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/test_service_ticket_service.py -q
$env:PYTHONPATH='backend'; python -m py_compile backend/app/services/service_ticket_service.py backend/tests/test_service_ticket_service.py
```

Expected: service ticket tests pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/service_ticket_service.py backend/tests/test_service_ticket_service.py
git commit -m "feat: add ticket clarification state machine"
```

---

### Task 3: Non-Stream RAG Race And B Conversion

**Files:**
- Modify: `backend/app/services/knowledge_service.py`
- Modify: `backend/app/api/v1/knowledge.py`
- Create/modify: `backend/tests/test_knowledge_service_tickets.py`
- Create/modify: `backend/tests/test_knowledge_api_llm_clarification.py`

- [ ] **Step 1: Add knowledge service tests for raw fallback metadata**

Create `backend/tests/test_knowledge_service_tickets.py` with tests proving `invoke_knowledge_qa(..., persist=False)` returns raw fallback fields and does not persist messages. Mock `agents.knowledge.get_knowledge_agent`, `agents.knowledge.create_initial_state`, `_load_kb_retrieval`, and `_persist_conversation_messages`.

- [ ] **Step 2: Implement `persist` flag and raw metadata**

In `backend/app/services/knowledge_service.py`, add `persist: bool = True` to `invoke_knowledge_qa()` and include these fields in `return_data`:

```python
"finish_reason": "stop",
"used_fallback": result.get("used_fallback", False),
"fallback_reason": result.get("fallback_reason"),
"quality_passed": result.get("quality_passed"),
"quality_level": _extract_quality_level(result.get("answer_quality")),
```

Wrap the `_persist_conversation_messages(...)` call:

```python
if persist:
    _persist_conversation_messages(...)
```

- [ ] **Step 3: Add API tests for non-blocking classifier race**

Create `backend/tests/test_knowledge_api_llm_clarification.py` with async tests:

```python
@patch("app.api.v1.knowledge.asyncio.wait")
@patch("app.api.v1.knowledge.asyncio.create_task")
async def test_normal_text_starts_rag_even_when_classifier_is_pending(...):
    # create_task should be called for RAG before the classifier result is required
    # pending classifier should cause D fallback and normal RAG response
```

Also add tests for:

- classifier returns A before deadline and API returns clarification without waiting for RAG completion
- malformed/timeout classifier result defaults to D and returns original RAG answer
- RAG fallback triggers B-class ticket only after `should_start_missing_knowledge_flow(result)` is true

- [ ] **Step 4: Implement non-stream orchestration**

In `backend/app/api/v1/knowledge.py`:

- import `asyncio`
- import `classify_ticket_intent_with_llm`, `process_ticket_clarification_turn`, `should_start_missing_knowledge_flow`
- create `rag_task = asyncio.create_task(invoke_knowledge_qa(..., persist=False))`
- create `classifier_task = asyncio.create_task(asyncio.to_thread(classify_ticket_intent_with_llm, ...))`
- wait for the classifier up to `0.6` seconds
- if classifier returns A/C and `needs_ticket_flow=True`, return a clarification response and cancel/ignore the RAG task
- otherwise await `rag_task`
- after RAG result, if B gate is true, call the classifier with `rag_result` and create B clarification
- if no B, persist the normal conversation and return original `KnowledgeResponse`

Use this response shape for clarification:

```python
KnowledgeResponse(
    status_code=200,
    request_id=result_request_id,
    session_id=session_id,
    answer=ticket_result["answer"],
    confidence=0.0,
    sources=[],
    model=model_name,
    finish_reason=ticket_result.get("finish_reason") or "clarification",
    thoughts={
        "clarification_required": True,
        "clarification": ticket_result.get("clarification"),
        "ticket_intent": decision,
    },
    image_map=None,
)
```

- [ ] **Step 5: Run tests and commit**

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/test_knowledge_service_tickets.py backend/tests/test_knowledge_api_llm_clarification.py -q
$env:PYTHONPATH='backend'; python -m py_compile backend/app/services/knowledge_service.py backend/app/api/v1/knowledge.py
git add backend/app/services/knowledge_service.py backend/app/api/v1/knowledge.py backend/tests/test_knowledge_service_tickets.py backend/tests/test_knowledge_api_llm_clarification.py
git commit -m "feat: race ticket classifier with rag"
```

---

### Task 4: Stream B Conversion Without Slowing First Delta

**Files:**
- Modify: `backend/app/services/knowledge_service.py`
- Modify: `backend/app/api/v1/knowledge.py`
- Create/modify: `backend/tests/test_knowledge_stream_ticket_flow.py`

- [ ] **Step 1: Add stream tests**

Create tests proving:

- `stream_knowledge_qa_sse(..., convert_missing_knowledge_to_ticket=True)` converts fallback final state into a clarification `done` event.
- conversion skips `_persist_conversation_messages`.
- conversion exceptions fall back to normal RAG `done` and persist.
- default `convert_missing_knowledge_to_ticket=False` leaves normal stream unchanged.

- [ ] **Step 2: Implement optional stream conversion**

Add optional parameters to `stream_knowledge_qa_sse()`:

```python
convert_missing_knowledge_to_ticket: bool = False
user_id: Optional[str] = None
user_name: Optional[str] = None
channel: str = "web"
sender_id: Optional[str] = None
requester_name: Optional[str] = None
```

After final LangGraph state and before `done`, construct `rag_result`. If B gate is true, call `classify_ticket_intent_with_llm(..., rag_result=rag_result)` and `process_ticket_clarification_turn(...)`. If this fails, log and keep normal stream behavior.

- [ ] **Step 3: Wire stream API**

Pass identity and `convert_missing_knowledge_to_ticket=True` from `knowledge_qa_stream()`.

- [ ] **Step 4: Run tests and commit**

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/test_knowledge_stream_ticket_flow.py -q
$env:PYTHONPATH='backend'; python -m py_compile backend/app/services/knowledge_service.py backend/app/api/v1/knowledge.py backend/tests/test_knowledge_stream_ticket_flow.py
git add backend/app/services/knowledge_service.py backend/app/api/v1/knowledge.py backend/tests/test_knowledge_stream_ticket_flow.py
git commit -m "feat: convert streaming rag misses to ticket clarification"
```

---

### Task 5: Full Regression Verification

**Files:**
- No source edits expected unless verification exposes a compatibility bug.

- [ ] **Step 1: Run focused tests**

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/test_service_ticket_service.py backend/tests/test_knowledge_service_tickets.py backend/tests/test_knowledge_api_llm_clarification.py backend/tests/test_knowledge_stream_ticket_flow.py -q
```

Expected: pass. If local dependencies such as `psycopg2` or `dashscope` are missing, rerun with the lightweight in-memory stubs used in existing tests and document the blocker.

- [ ] **Step 2: Run broader touched-path tests**

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/test_service_ticket_repository.py backend/tests/test_knowledge_service_clarification.py backend/tests/test_quality_check_unanswered.py backend/tests/test_generate_sources.py -q
```

Expected: pass or environment dependency blocker documented.

- [ ] **Step 3: Compile changed backend modules**

```powershell
$env:PYTHONPATH='backend'; python -m py_compile backend/app/services/service_ticket_service.py backend/app/services/knowledge_service.py backend/app/api/v1/knowledge.py backend/tests/test_service_ticket_service.py backend/tests/test_knowledge_service_tickets.py backend/tests/test_knowledge_api_llm_clarification.py backend/tests/test_knowledge_stream_ticket_flow.py
```

Expected: exit code 0.

- [ ] **Step 4: Inspect final diff**

```powershell
git diff --stat
git diff -- backend/app/services/service_ticket_service.py backend/app/services/knowledge_service.py backend/app/api/v1/knowledge.py
```

Expected: changes are limited to LLM classifier, ticket state, RAG orchestration, and tests.

---

## Self-Review

Spec coverage:

- LLM-only A/B/C/D classification is covered in Task 1.
- `doubao-seed-2-0-mini-260428` and thinking-disabled path are asserted in Task 1.
- Normal text classifier timeout/failure defaulting to D is covered in Tasks 1 and 3.
- RAG starts immediately and is not delayed by the classifier in Task 3.
- A/C pre-RAG interception is covered in Task 3.
- B only after RAG fallback is covered in Tasks 3 and 4.
- A/C/B round caps and canonical ticket payload are covered in Task 2.
- Stream conversion and normal stream preservation are covered in Task 4.

Placeholder scan:

- No TBD/TODO placeholders remain.
- Steps name exact files and expected commands.

Type consistency:

- Intent constants are `A/B/C/D`.
- Classifier returns dict payloads used by the state machine and API.
- `invoke_knowledge_qa(..., persist=False)` is backward-compatible.
- `stream_knowledge_qa_sse(..., convert_missing_knowledge_to_ticket=False)` is backward-compatible.
