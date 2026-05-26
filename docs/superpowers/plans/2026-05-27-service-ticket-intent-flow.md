# Service Ticket Intent Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing service-ticket clarification flow so A/B/C special intents collect complete ticket information while D-class RAG questions keep the current answer path.

**Architecture:** Keep `service_ticket` as the durable state store and `service_ticket.clarification` as the multi-turn payload. `backend/app/services/service_ticket_service.py` owns classification, canonical payloads, field collection, refusal handling, and ticket finalization; `backend/app/api/v1/knowledge.py` remains the thin orchestrator that uploads images, invokes RAG, and maps service results into `KnowledgeResponse` or SSE events.

**Tech Stack:** Python 3.10, FastAPI, Pydantic, unittest, LangGraph RAG services, PostgreSQL-backed repositories.

---

## File Structure

- Modify `backend/app/services/service_ticket_service.py`
  - Add stable intent constants and round limits for A/B/C.
  - Add refusal detection, canonical clarification payload helpers, and final ticket message helpers.
  - Extend `process_clarification_turn()` to support A direct finalization when no SOP exists, A field-complete finalization, B/C round limits, and canonical payload keys.
  - Add B-class helpers for post-RAG fallback detection and missing-knowledge ticket creation.

- Modify `backend/app/services/knowledge_service.py`
  - Add optional persistence control to non-stream RAG so the API can inspect fallback results before recording D/B tickets.
  - Include raw quality fields in non-stream return data.
  - Add a stream-time conversion branch that can replace a fallback answer with a B-class clarification done event before persistence.

- Modify `backend/app/api/v1/knowledge.py`
  - Route active clarification turns through the enhanced service state.
  - Convert post-RAG fallback results into B-class clarification for non-stream requests.
  - Pass stream conversion flags into `stream_knowledge_qa_sse()`.
  - Keep D-class response shape unchanged.

- Modify `backend/tests/test_service_ticket_service.py`
  - Add regression coverage for canonical A/B/C payloads, no-SOP finalization, C round cap, B round cap, and refusal handling.

- Modify `backend/tests/test_knowledge_service_tickets.py`
  - Add coverage for `invoke_knowledge_qa(..., persist=False)` returning raw quality fields without writing tickets.

- Modify `backend/tests/test_knowledge_api_llm_clarification.py`
  - Add non-stream API tests for B conversion after RAG fallback and D preservation.

- Create `backend/tests/test_knowledge_stream_ticket_flow.py`
  - Add focused SSE tests proving fallback can be emitted as a clarification done event and normal D results remain unchanged.

---

### Task 1: Canonical Payload And Refusal Service Tests

**Files:**
- Modify: `backend/tests/test_service_ticket_service.py`
- Modify: `backend/app/services/service_ticket_service.py`

- [ ] **Step 1: Add failing tests for canonical clarification state**

Append these imports at the existing import block in `backend/tests/test_service_ticket_service.py`:

```python
from app.services.service_ticket_service import (
    INTENT_AMBIGUOUS,
    INTENT_MISSING_KNOWLEDGE,
    INTENT_OPERATION,
    _is_user_refusal,
    should_start_missing_knowledge_flow,
)
```

Append these test methods inside `ServiceTicketServiceTests`:

```python
    def test_refusal_detection_matches_explicit_stop_phrases(self):
        examples = [
            "不知道",
            "不想回答了",
            "我也不清楚",
            "不用问了，直接处理",
            "没有更多信息",
        ]

        for query in examples:
            with self.subTest(query=query):
                self.assertTrue(_is_user_refusal(query))

        self.assertFalse(_is_user_refusal("手机号 13800138000，门店是A店"))

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_no_workflow_operation_finalizes_manual_ticket_immediately(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo

        result = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="帮我绑定一下富友账户",
            channel="h5",
            reason="operation_required",
            force_start=True,
            workflow={
                "workflow_found": False,
                "workflow_summary": "",
                "required_fields": [],
                "field_labels": {},
                "fallback_used": True,
            },
            workflow_sources=[],
        )

        self.assertEqual(result["status"], "pending_manual")
        self.assertEqual(result["finish_reason"], "manual_ticket_created")
        self.assertIn("工单号 ticket-1", result["answer"])
        self.assertEqual(repo.ticket["clarification_round"], 1)
        self.assertEqual(repo.ticket["clarification"]["intent_class"], INTENT_OPERATION)
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "no_standard_workflow")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "incomplete")
        self.assertTrue(repo.ticket["clarification"]["ready_for_manual"])

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_operation_with_workflow_finalizes_when_required_fields_complete(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo
        workflow = {
            "workflow_found": True,
            "workflow_summary": "富友处理需要手机号。",
            "required_fields": ["phone"],
            "field_labels": {"phone": "可联系手机号"},
            "fallback_used": False,
        }

        first = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="帮我处理富友收款",
            channel="h5",
            reason="operation_required",
            force_start=True,
            workflow=workflow,
            workflow_sources=[],
        )
        second = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="手机号 13800138000",
            channel="h5",
        )

        self.assertEqual(first["status"], "clarifying")
        self.assertEqual(second["status"], "pending_manual")
        self.assertIn("工单号 ticket-1", second["answer"])
        self.assertEqual(repo.ticket["clarification"]["intent_class"], INTENT_OPERATION)
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "fields_complete")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "complete")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_ambiguous_flow_finalizes_after_three_unclear_rounds(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo

        first = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="这个怎么处理",
            channel="h5",
        )
        second = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="就是那个问题",
            channel="h5",
        )
        third = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="还是不清楚",
            channel="h5",
        )

        self.assertEqual(first["status"], "clarifying")
        self.assertEqual(second["status"], "clarifying")
        self.assertEqual(third["status"], "pending_manual")
        self.assertEqual(repo.ticket["clarification"]["intent_class"], INTENT_AMBIGUOUS)
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "semantic_unresolved")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "incomplete")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_user_refusal_finalizes_active_ticket_as_incomplete(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo

        first = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="帮我开通直播权限",
            channel="h5",
            reason="operation_required",
            force_start=True,
            workflow={
                "workflow_found": True,
                "workflow_summary": "开通权限需要门店和手机号。",
                "required_fields": ["issue_detail", "phone"],
                "field_labels": {},
                "fallback_used": False,
            },
            workflow_sources=[],
        )
        second = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="不知道，不想回答了",
            channel="h5",
        )

        self.assertEqual(first["status"], "clarifying")
        self.assertEqual(second["status"], "pending_manual")
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "user_refused_or_unknown")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "incomplete")

    def test_missing_knowledge_flow_starts_only_for_rag_fallback(self):
        should_start = should_start_missing_knowledge_flow(
            {
                "used_fallback": True,
                "fallback_reason": "no_relevant_knowledge",
                "quality_passed": False,
                "confidence": 0.2,
                "sources": [],
            }
        )
        should_skip = should_start_missing_knowledge_flow(
            {
                "used_fallback": False,
                "fallback_reason": None,
                "quality_passed": True,
                "confidence": 0.85,
                "sources": [{"id": "chunk-1"}],
            }
        )

        self.assertTrue(should_start)
        self.assertFalse(should_skip)
```

- [ ] **Step 2: Run the focused test file and verify it fails**

Run:

```powershell
python -m pytest backend/tests/test_service_ticket_service.py -q
```

Expected: FAIL with import errors for `INTENT_OPERATION`, `INTENT_MISSING_KNOWLEDGE`, `INTENT_AMBIGUOUS`, `_is_user_refusal`, and `should_start_missing_knowledge_flow`, plus behavior failures for direct no-SOP finalization.

- [ ] **Step 3: Add constants and helper functions**

In `backend/app/services/service_ticket_service.py`, insert this block after `ALL_TICKET_STATUSES`:

```python
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
```

Insert these helpers after `_is_knowledge_help_question()`:

```python
_REFUSAL_RE = re.compile(
    r"(不想回答|不用问|别问|不知道|不清楚|无法提供|没法提供|没有更多|没有其他|直接处理|直接提交|你们处理|人工处理)"
)


def _is_user_refusal(query: str) -> bool:
    text = " ".join((query or "").strip().split())
    return bool(text and _REFUSAL_RE.search(text))


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
    if rag_result.get("used_fallback") and (no_relevant_reason or rag_result.get("quality_passed") is False):
        return True
    if rag_result.get("quality_passed") is False and no_relevant_reason:
        return True
    if confidence is not None and float(confidence) < 0.35 and not sources:
        return True
    return False
```

- [ ] **Step 4: Replace manual answer helper**

Replace `_manual_ticket_answer()` with:

```python
def _manual_ticket_answer(ticket_id: Optional[str], missing: List[str], *, completion_status: str) -> str:
    ticket_label = ticket_id or "后台工单"
    base = f"已为您记录问题，工单号 {ticket_label}，后续将由专人处理。"
    if completion_status == COMPLETION_INCOMPLETE:
        if missing:
            labels = "、".join(_field_label(f) for f in missing)
            return f"当前信息未完全收集，仍缺少：{labels}。{base}"
        return f"当前信息未完全收集。{base}"
    return base
```

- [ ] **Step 5: Run focused tests and confirm new import failures are gone**

Run:

```powershell
python -m pytest backend/tests/test_service_ticket_service.py -q
```

Expected: FAIL only on behavior assertions for `process_clarification_turn()` until Task 2 updates the state machine.

- [ ] **Step 6: Commit Task 1**

```powershell
git add backend/tests/test_service_ticket_service.py backend/app/services/service_ticket_service.py
git commit -m "test: define ticket intent state expectations"
```

---

### Task 2: Implement A And C State Machine Behavior

**Files:**
- Modify: `backend/app/services/service_ticket_service.py`
- Modify: `backend/tests/test_service_ticket_service.py`

- [ ] **Step 1: Update `process_clarification_turn()` status decision**

Replace the block from `existing = active.get("clarification") if active else {}` through the `clarification = { ... }` assignment in `process_clarification_turn()` with:

```python
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
```

- [ ] **Step 2: Ensure newly created tickets get their ticket id in the final answer**

After the `repo.create_with_contexts(...)` call and before the `return` in `process_clarification_turn()`, insert:

```python
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
```

- [ ] **Step 3: Update legacy assertions to the new final copy**

In `backend/tests/test_service_ticket_service.py`, replace assertions that expect `"已转后台人工工单"` with assertions for `"已为您记录问题"` and `"工单号 ticket-1"`.

Use this exact replacement for the older assertion in `test_clarification_flow_reuses_ticket_and_turns_manual_after_enough_info`:

```python
        self.assertIn("已为您记录问题", third["answer"])
        self.assertIn("工单号 ticket-1", third["answer"])
```

- [ ] **Step 4: Run service ticket tests**

Run:

```powershell
python -m pytest backend/tests/test_service_ticket_service.py -q
```

Expected: PASS for service-ticket tests.

- [ ] **Step 5: Commit Task 2**

```powershell
git add backend/app/services/service_ticket_service.py backend/tests/test_service_ticket_service.py
git commit -m "feat: canonicalize ticket clarification state"
```

---

### Task 3: Add Post-RAG B-Class Non-Stream Conversion

**Files:**
- Modify: `backend/app/services/service_ticket_service.py`
- Modify: `backend/app/services/knowledge_service.py`
- Modify: `backend/app/api/v1/knowledge.py`
- Modify: `backend/tests/test_knowledge_service_tickets.py`
- Modify: `backend/tests/test_knowledge_api_llm_clarification.py`

- [ ] **Step 1: Add failing knowledge-service persistence test**

Append this test method to `KnowledgeServiceTicketTests` in `backend/tests/test_knowledge_service_tickets.py`:

```python
    def test_invoke_knowledge_qa_can_return_fallback_without_persisting_ticket(self):
        class FakeAgent:
            async def ainvoke(self, _initial_state, config=None):
                return {
                    "answer": "知识库未包含该问题的答案。",
                    "confidence": 0.2,
                    "sources": [],
                    "metrics": MagicMock(total_chunks_retrieved=0, chunks_after_rerank=0),
                    "used_fallback": True,
                    "fallback_reason": "no_relevant_knowledge",
                    "quality_passed": False,
                    "answer_quality": "low",
                }

        with patch("agents.knowledge.get_knowledge_agent", return_value=FakeAgent()), patch(
            "agents.knowledge.create_initial_state",
            return_value={"query": "一个知识库没有的问题"},
        ), patch(
            "app.services.knowledge_service._load_kb_retrieval",
            return_value=({"id": "kb-1", "name": "fushang", "kb_type": "standard"}, {}),
        ), patch(
            "app.services.knowledge_service._persist_conversation_messages"
        ) as mock_persist_messages:
            import asyncio

            result = asyncio.run(
                invoke_knowledge_qa(
                    query="一个知识库没有的问题",
                    model_name="doubao-seed-2-0-pro-260215",
                    session_id="session-1",
                    collection="fushang",
                    user_id="store-1",
                    persist=False,
                )
            )

        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["fallback_reason"], "no_relevant_knowledge")
        self.assertFalse(result["quality_passed"])
        self.assertEqual(result["quality_level"], "low")
        mock_persist_messages.assert_not_called()
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```powershell
python -m pytest backend/tests/test_knowledge_service_tickets.py::KnowledgeServiceTicketTests::test_invoke_knowledge_qa_can_return_fallback_without_persisting_ticket -q
```

Expected: FAIL because `invoke_knowledge_qa()` does not accept `persist`.

- [ ] **Step 3: Add `persist` support and raw quality fields**

In `backend/app/services/knowledge_service.py`, update `invoke_knowledge_qa()` signature:

```python
async def invoke_knowledge_qa(
    query: str,
    model_name: str,
    session_id: str,
    collection: Optional[str] = None,
    force_multi_doc: Optional[bool] = None,
    keyword_filter: Optional[str] = None,
    query_image_url: Optional[str] = None,
    query_image_oss_key: Optional[str] = None,
    user_id: str = "guest_default",
    user_name: Optional[str] = None,
    channel: str = "web",
    sender_id: Optional[str] = None,
    requester_name: Optional[str] = None,
    persist: bool = True,
) -> dict:
```

In `return_data`, add:

```python
        "used_fallback": result.get("used_fallback", False),
        "fallback_reason": result.get("fallback_reason"),
        "quality_passed": result.get("quality_passed"),
        "quality_level": quality_level,
        "manual_review_recommended": manual_review_recommended,
```

Wrap `_persist_conversation_messages(...)`:

```python
    if persist:
        _persist_conversation_messages(
            session_id,
            query,
            result.get("answer") or "",
            result.get("sources") or [],
            result.get("confidence"),
            query_image_oss_key,
            used_fallback=result.get("used_fallback", False),
            fallback_reason=result.get("fallback_reason"),
            quality_passed=result.get("quality_passed"),
            quality_level=quality_level,
            kb_name=kb.get("name") if kb else None,
            user_id=user_id,
            user_name=user_name,
            channel=channel,
            sender_id=sender_id,
            requester_name=requester_name or user_name,
        )
```

- [ ] **Step 4: Add B-class ticket creation helper**

In `backend/app/services/service_ticket_service.py`, insert this function before `record_ticket()`:

```python
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
```

- [ ] **Step 5: Add failing API test for B conversion**

Append this test method to `KnowledgeApiLlmClarificationTests` in `backend/tests/test_knowledge_api_llm_clarification.py`:

```python
    @patch("app.api.v1.knowledge.start_missing_knowledge_clarification")
    @patch("app.api.v1.knowledge.invoke_knowledge_qa", new_callable=AsyncMock)
    @patch("app.api.v1.knowledge.classify_operation_query_with_llm")
    @patch("app.api.v1.knowledge.persist_clarification_message")
    @patch("app.api.v1.knowledge._upload_query_image_if_needed")
    @patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session")
    async def test_rag_fallback_starts_missing_knowledge_clarification(
        self,
        mock_ensure_session,
        mock_upload_image,
        mock_persist_clarification,
        mock_llm_decision,
        mock_invoke_rag,
        mock_start_missing,
    ):
        mock_ensure_session.return_value = "session-1"
        mock_upload_image.return_value = (None, None)
        mock_persist_clarification.return_value = None
        mock_llm_decision.return_value = {
            "should_clarify": False,
            "reason": "knowledge_question",
            "message": "",
            "source": "llm",
        }
        mock_invoke_rag.return_value = {
            "request_id": "request-1",
            "session_id": "session-1",
            "answer": "知识库未包含该问题的答案。",
            "confidence": 0.2,
            "sources": [],
            "model": "doubao-seed-2-0-pro-260215",
            "thoughts": {"manual_review_recommended": True},
            "image_map": None,
            "finish_reason": "stop",
            "used_fallback": True,
            "fallback_reason": "no_relevant_knowledge",
            "quality_passed": False,
            "quality_level": "low",
        }
        mock_start_missing.return_value = {
            "ticket_id": "ticket-1",
            "answer": "请补充具体场景。",
            "finish_reason": "clarification",
            "clarification": {"intent_class": "B", "reason": "knowledge_missing"},
        }

        response = await knowledge.knowledge_qa(
            KnowledgeRequest(query="知识库没有的新问题", session_id="default", collection="fushang"),
            identity={"user_id": "store-1", "user_name": "张店长", "channel": "h5"},
        )

        self.assertEqual(response.finish_reason, "clarification")
        self.assertEqual(response.answer, "请补充具体场景。")
        self.assertEqual(response.thoughts["clarification"]["intent_class"], "B")
        mock_invoke_rag.assert_awaited_once()
        self.assertFalse(mock_invoke_rag.call_args.kwargs["persist"])
        mock_start_missing.assert_called_once()
```

- [ ] **Step 6: Wire non-stream API post-RAG conversion**

In `backend/app/api/v1/knowledge.py`, extend the import from `service_ticket_service`:

```python
from app.services.service_ticket_service import (
    classify_operation_query_with_llm,
    extract_operation_workflow_requirements,
    should_clarify_query,
    should_start_missing_knowledge_flow,
    start_missing_knowledge_clarification,
)
```

In the `invoke_knowledge_qa(...)` call inside `knowledge_qa()`, add `persist=False`.

Immediately after that call, insert:

```python
    if should_start_missing_knowledge_flow(result):
        missing_result = start_missing_knowledge_clarification(
            session_id=session_id,
            user_id=user_id,
            user_name=identity.get("user_name"),
            sender_id=identity.get("sender_id"),
            requester_name=identity.get("requester_name"),
            kb_name=request.collection,
            query=request.query,
            channel=identity.get("channel") or "web",
            rag_result=result,
            has_image=bool(query_image_oss_key),
            query_image_oss_key=query_image_oss_key,
        )
        return KnowledgeResponse(
            status_code=200,
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            answer=missing_result["answer"],
            confidence=0.0,
            sources=[],
            model=model_name,
            finish_reason=missing_result.get("finish_reason") or "clarification",
            thoughts={
                "clarification_required": True,
                "clarification": missing_result.get("clarification"),
                "missing_knowledge_started": True,
            },
            image_map=None,
        )

    from app.services.knowledge_service import _persist_conversation_messages
    _persist_conversation_messages(
        session_id,
        request.query,
        result.get("answer") or "",
        result.get("sources") or [],
        result.get("confidence"),
        query_image_oss_key,
        used_fallback=result.get("used_fallback", False),
        fallback_reason=result.get("fallback_reason"),
        quality_passed=result.get("quality_passed"),
        quality_level=result.get("quality_level"),
        kb_name=request.collection,
        user_id=user_id,
        user_name=identity.get("user_name"),
        channel=identity.get("channel") or "web",
        sender_id=identity.get("sender_id"),
        requester_name=identity.get("requester_name") or identity.get("user_name"),
    )
```

- [ ] **Step 7: Run affected non-stream tests**

Run:

```powershell
python -m pytest backend/tests/test_knowledge_service_tickets.py backend/tests/test_knowledge_api_llm_clarification.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```powershell
git add backend/app/services/service_ticket_service.py backend/app/services/knowledge_service.py backend/app/api/v1/knowledge.py backend/tests/test_knowledge_service_tickets.py backend/tests/test_knowledge_api_llm_clarification.py
git commit -m "feat: start missing knowledge clarification after rag fallback"
```

---

### Task 4: Add Stream B-Class Conversion

**Files:**
- Create: `backend/tests/test_knowledge_stream_ticket_flow.py`
- Modify: `backend/app/services/knowledge_service.py`
- Modify: `backend/app/api/v1/knowledge.py`

- [ ] **Step 1: Create failing stream tests**

Create `backend/tests/test_knowledge_stream_ticket_flow.py` with:

```python
# -*- coding: utf-8 -*-
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.knowledge_service import stream_knowledge_qa_sse


def _event_payloads(events):
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[len("data: "):]))
    return payloads


class StreamTicketFlowTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.services.knowledge_service.start_missing_knowledge_clarification")
    @patch("app.services.knowledge_service.should_start_missing_knowledge_flow")
    @patch("agents.knowledge.openai_stream.iter_openai_text_deltas")
    @patch("agents.knowledge.get_knowledge_stream_prep_agent")
    @patch("agents.knowledge.create_initial_state")
    @patch("app.services.knowledge_service._load_kb_retrieval")
    async def test_stream_converts_fallback_done_to_missing_knowledge_clarification(
        self,
        mock_load_kb,
        mock_create_state,
        mock_get_agent,
        mock_deltas,
        mock_should_start,
        mock_start_missing,
    ):
        class FakeSnapshot:
            def __init__(self, values):
                self.values = values

        class FakeAgent:
            def __init__(self):
                self.state_calls = 0

            async def ainvoke(self, _state, config=None):
                return None

            async def aget_state(self, config):
                self.state_calls += 1
                if self.state_calls == 1:
                    return FakeSnapshot(
                        {
                            "metrics": MagicMock(total_chunks_retrieved=0, chunks_after_rerank=0),
                            "reranked_chunks": [],
                            "config": MagicMock(model="doubao-seed-2-0-pro-260215", kb_type="standard"),
                            "query": "知识库没有的新问题",
                        }
                    )
                return FakeSnapshot(
                    {
                        "answer": "知识库未包含该问题的答案。",
                        "confidence": 0.2,
                        "sources": [],
                        "image_map": {},
                        "used_fallback": True,
                        "fallback_reason": "no_relevant_knowledge",
                        "quality_passed": False,
                        "answer_quality": "low",
                        "metrics": MagicMock(total_chunks_retrieved=0, chunks_after_rerank=0),
                    }
                )

            async def aupdate_state(self, config, values):
                return None

        async def fake_deltas(messages, model_name):
            yield "知识库未包含该问题的答案。"

        mock_load_kb.return_value = ({"id": "kb-1", "name": "fushang", "kb_type": "standard"}, {})
        mock_create_state.return_value = {"query": "知识库没有的新问题"}
        mock_get_agent.return_value = FakeAgent()
        mock_deltas.side_effect = fake_deltas
        mock_should_start.return_value = True
        mock_start_missing.return_value = {
            "ticket_id": "ticket-1",
            "answer": "请补充具体业务场景。",
            "finish_reason": "clarification",
            "clarification": {"intent_class": "B", "reason": "knowledge_missing"},
        }

        events = []
        async for event in stream_knowledge_qa_sse(
            query="知识库没有的新问题",
            model_name="doubao-seed-2-0-pro-260215",
            session_id="session-1",
            collection="fushang",
            user_id="store-1",
            user_name="张店长",
            channel="h5",
            convert_missing_knowledge_to_ticket=True,
        ):
            events.append(event)

        payloads = _event_payloads(events)
        done = payloads[-1]
        self.assertEqual(done["finish_reason"], "clarification")
        self.assertEqual(done["answer"], "请补充具体业务场景。")
        self.assertTrue(done["thoughts"]["clarification_required"])
        self.assertEqual(done["thoughts"]["clarification"]["intent_class"], "B")
        mock_start_missing.assert_called_once()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the stream test and verify it fails**

Run:

```powershell
python -m pytest backend/tests/test_knowledge_stream_ticket_flow.py -q
```

Expected: FAIL because `stream_knowledge_qa_sse()` does not accept `convert_missing_knowledge_to_ticket`.

- [ ] **Step 3: Import B helpers in `knowledge_service.py`**

Add this import near the other top-level imports in `backend/app/services/knowledge_service.py`:

```python
from app.services.service_ticket_service import (
    should_start_missing_knowledge_flow,
    start_missing_knowledge_clarification,
)
```

- [ ] **Step 4: Extend `stream_knowledge_qa_sse()` signature**

Add parameters to the signature:

```python
    convert_missing_knowledge_to_ticket: bool = False,
```

- [ ] **Step 5: Add stream conversion before yielding done**

Inside `stream_knowledge_qa_sse()`, after `done_thoughts` is built and before the `yield _sse("done", ...)` block, insert:

```python
    rag_result = {
        "request_id": request_id,
        "session_id": session_id,
        "answer": answer_final,
        "confidence": done_confidence,
        "sources": done_sources,
        "model": model_name,
        "thoughts": done_thoughts,
        "image_map": final.get("image_map") or ctx.get("image_map") or {},
        "finish_reason": finish_reason,
        "used_fallback": final.get("used_fallback", False),
        "fallback_reason": final.get("fallback_reason"),
        "quality_passed": final.get("quality_passed"),
        "quality_level": _extract_quality_level(final.get("answer_quality")),
    }
    missing_result = None
    if convert_missing_knowledge_to_ticket and should_start_missing_knowledge_flow(rag_result):
        missing_result = start_missing_knowledge_clarification(
            session_id=session_id,
            user_id=user_id,
            user_name=user_name,
            sender_id=sender_id,
            requester_name=requester_name or user_name,
            kb_name=collection,
            query=query,
            channel=channel,
            rag_result=rag_result,
            has_image=bool(query_image_oss_key),
            query_image_oss_key=query_image_oss_key,
        )
        answer_final = missing_result["answer"]
        done_sources = []
        done_confidence = 0.0
        finish_reason = missing_result.get("finish_reason") or "clarification"
        done_thoughts = {
            **done_thoughts,
            "clarification_required": True,
            "clarification": missing_result.get("clarification"),
            "missing_knowledge_started": True,
        }
```

Update the later `_persist_conversation_messages(...)` call:

```python
    if not missing_result:
        _persist_conversation_messages(
            session_id,
            query,
            answer_final,
            final.get("sources") or [],
            final.get("confidence"),
            query_image_oss_key,
            used_fallback=final.get("used_fallback", False),
            fallback_reason=final.get("fallback_reason"),
            quality_passed=final.get("quality_passed"),
            quality_level=_extract_quality_level(final.get("answer_quality")),
            kb_name=kb.get("name") if kb else None,
            user_id=user_id,
            user_name=user_name,
            channel=channel,
            sender_id=sender_id,
            requester_name=requester_name or user_name,
        )
```

- [ ] **Step 6: Pass stream conversion flag from API**

In `backend/app/api/v1/knowledge.py`, add this keyword argument to the `stream_knowledge_qa_sse(...)` call:

```python
            convert_missing_knowledge_to_ticket=True,
```

- [ ] **Step 7: Run stream tests**

Run:

```powershell
python -m pytest backend/tests/test_knowledge_stream_ticket_flow.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 4**

```powershell
git add backend/app/services/knowledge_service.py backend/app/api/v1/knowledge.py backend/tests/test_knowledge_stream_ticket_flow.py
git commit -m "feat: convert streaming rag fallback to clarification"
```

---

### Task 5: B-Class Active Follow-Up And Round Cap

**Files:**
- Modify: `backend/tests/test_service_ticket_service.py`
- Modify: `backend/app/services/service_ticket_service.py`

- [ ] **Step 1: Add failing tests for active B round cap**

Append this test method to `ServiceTicketServiceTests`:

```python
    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_missing_knowledge_flow_finalizes_after_five_rounds(self, mock_repo):
        repo = FakeClarificationRepo()
        repo.ticket = {
            "id": "ticket-1",
            "session_id": "session-1",
            "user_id": "store-1",
            "kb_name": "fushang",
            "status": "clarifying",
            "clarification_round": 4,
            "clarification": {
                "intent_class": INTENT_MISSING_KNOWLEDGE,
                "reason": "knowledge_missing",
                "original_query": "知识库没有的新问题",
                "required_fields": ["issue_detail", "phone"],
                "missing_fields": ["phone"],
                "collected": {"issue_detail": "知识库没有的新问题"},
                "turns": [
                    {"round": 1, "query": "知识库没有的新问题"},
                    {"round": 2, "query": "场景A"},
                    {"round": 3, "query": "影响门店"},
                    {"round": 4, "query": "仍然没有答案"},
                ],
                "kb_result": {"hit": False},
            },
        }
        mock_repo.return_value = repo

        result = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="手机号 13800138000",
            channel="h5",
            continue_only=True,
        )

        self.assertEqual(result["status"], "pending_manual")
        self.assertEqual(result["finish_reason"], "manual_ticket_created")
        self.assertEqual(repo.ticket["clarification_round"], 5)
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "max_rounds")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "incomplete")
```

- [ ] **Step 2: Run the focused test and verify behavior**

Run:

```powershell
python -m pytest backend/tests/test_service_ticket_service.py::ServiceTicketServiceTests::test_missing_knowledge_flow_finalizes_after_five_rounds -q
```

Expected: PASS if Task 2 round-cap logic already covers B; otherwise FAIL because B finalization still uses the old round rules.

- [ ] **Step 3: Adjust B completion status if the test fails**

If the test fails because B with all required fields is marked `complete`, update the completion decision in `process_clarification_turn()`:

```python
    elif fields_complete:
        exit_reason = EXIT_FIELDS_COMPLETE
        completion_status = COMPLETION_COMPLETE if intent_class == INTENT_OPERATION else COMPLETION_INCOMPLETE
```

- [ ] **Step 4: Run service tests**

Run:

```powershell
python -m pytest backend/tests/test_service_ticket_service.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```powershell
git add backend/app/services/service_ticket_service.py backend/tests/test_service_ticket_service.py
git commit -m "feat: cap missing knowledge clarification rounds"
```

---

### Task 6: Full Regression Verification

**Files:**
- No source edits expected.

- [ ] **Step 1: Run focused backend ticket and knowledge tests**

Run:

```powershell
python -m pytest backend/tests/test_service_ticket_service.py backend/tests/test_knowledge_service_tickets.py backend/tests/test_knowledge_api_llm_clarification.py backend/tests/test_knowledge_stream_ticket_flow.py -q
```

Expected: PASS.

- [ ] **Step 2: Run broader backend regression tests touched by the ticket path**

Run:

```powershell
python -m pytest backend/tests/test_service_ticket_repository.py backend/tests/test_knowledge_service_clarification.py backend/tests/test_quality_check_unanswered.py backend/tests/test_generate_sources.py -q
```

Expected: PASS.

- [ ] **Step 3: Run frontend tests only if backend API response shape changed**

Run:

```powershell
cd frontend
npm test -- --run
```

Expected: PASS, or if this project has no `test` script, document the exact npm error and skip frontend verification because this backend change keeps the request/response contract stable.

- [ ] **Step 4: Inspect git diff**

Run:

```powershell
git diff --stat
git diff -- backend/app/services/service_ticket_service.py backend/app/services/knowledge_service.py backend/app/api/v1/knowledge.py
```

Expected: only ticket-intent, knowledge-service persistence, API orchestration, and tests changed.

- [ ] **Step 5: Commit verification-only adjustments if any were needed**

If Step 1 or Step 2 required a small compatibility fix, commit it:

```powershell
git add backend/app/services/service_ticket_service.py backend/app/services/knowledge_service.py backend/app/api/v1/knowledge.py backend/tests
git commit -m "test: cover ticket intent flow regressions"
```

If no edits were needed, do not create an empty commit.

---

## Self-Review

Spec coverage:

- A/D boundary is covered by existing rule tests and new A no-SOP/SOP tests.
- A SOP field collection is covered by existing workflow tests plus the new field-complete test.
- B post-RAG start is covered by the new non-stream API test and stream test.
- B five-round cap is covered by the new service test.
- C three-round cap is covered by the new service test.
- Refusal is covered by helper and active-ticket finalization tests.
- Active recovery remains covered by existing `continue_only` tests.
- Canonical payload keys are asserted in A/B/C tests.

Placeholder scan:

- This plan contains concrete implementation steps and code blocks for every planned source edit.

Type consistency:

- New constants are strings exported by `service_ticket_service.py`.
- `invoke_knowledge_qa(..., persist=False)` is a backward-compatible optional parameter.
- `stream_knowledge_qa_sse(..., convert_missing_knowledge_to_ticket=True)` is a backward-compatible optional parameter.
- New B helper returns the same shape as existing clarification results: `ticket_id`, `status`, `answer`, `clarification_round`, `clarification`, `confidence`, and `finish_reason`.
