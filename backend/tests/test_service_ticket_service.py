# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch

from app.services.service_ticket_service import (
    INTENT_AMBIGUOUS,
    INTENT_MISSING_KNOWLEDGE,
    INTENT_OPERATION,
    _exit_reason_for_round_cap,
    _intent_class_for_reason,
    _is_complete,
    _is_user_refusal,
    _max_rounds_for_intent,
    build_context_snapshots,
    classify_operation_query_with_llm,
    classify_ticket_status,
    process_clarification_turn,
    should_clarify_query,
    should_start_missing_knowledge_flow,
)


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


class ServiceTicketServiceTests(unittest.TestCase):
    def test_short_ambiguous_query_requires_clarification(self):
        result = should_clarify_query("怎么处理")

        self.assertTrue(result["should_clarify"])
        self.assertIn("请补充", result["message"])

    def test_operation_query_requires_clarification(self):
        result = should_clarify_query("帮我开通直播权限")

        self.assertTrue(result["should_clarify"])
        self.assertEqual(result["reason"], "operation_required")

    def test_live_permission_query_requires_clarification(self):
        result = should_clarify_query("\u5f00\u901a\u76f4\u64ad\u6743\u9650")

        self.assertTrue(result["should_clarify"])
        self.assertEqual(result["reason"], "operation_required")

    def test_direct_account_binding_request_requires_clarification(self):
        result = should_clarify_query("\u5e2e\u6211\u7ed1\u5b9a\u4e00\u4e0b\u5bcc\u53cb\u8d26\u6237")

        self.assertTrue(result["should_clarify"])
        self.assertEqual(result["reason"], "operation_required")

    def test_permission_capability_questions_are_not_forced_into_ticket(self):
        examples = [
            "\u5e97\u9762\u4e4b\u524d\u5f00\u901a\u7684\u5458\u5de5\u7684\u76f4\u64ad\u6743\u9650\u53ef\u4ee5\u5173\u95ed\u5417",
            "\u76f4\u64ad\u6743\u9650\u53ef\u4ee5\u5173\u95ed\u5417",
            "\u600e\u4e48\u5173\u95ed\u76f4\u64ad\u6743\u9650",
        ]

        for query in examples:
            with self.subTest(query=query):
                result = should_clarify_query(query)
                self.assertFalse(result["should_clarify"])

    def test_delegate_permission_change_request_requires_clarification(self):
        result = should_clarify_query("\u5e2e\u6211\u5173\u95ed\u8fd9\u4e2a\u5458\u5de5\u7684\u76f4\u64ad\u6743\u9650")

        self.assertTrue(result["should_clarify"])
        self.assertEqual(result["reason"], "operation_required")

    def test_how_to_account_binding_is_not_forced_by_rules(self):
        result = should_clarify_query("\u5bcc\u53cb\u8d26\u6237\u600e\u4e48\u7ed1\u5b9a")

        self.assertFalse(result["should_clarify"])

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_llm_classifier_marks_semantic_operation_when_rules_do_not_match(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = '{"should_clarify": true, "reason": "operation_required", "message": "需要先补齐工单信息", "confidence": 0.86}'
        mock_get_llm.return_value = llm

        result = classify_operation_query_with_llm("帮我把富友收款弄一下")

        self.assertTrue(result["should_clarify"])
        self.assertEqual(result["reason"], "operation_required")
        self.assertEqual(result["source"], "llm")
        llm.responses_text.assert_called_once()

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_llm_classifier_failure_does_not_force_clarification(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.side_effect = RuntimeError("timeout")
        mock_get_llm.return_value = llm

        result = classify_operation_query_with_llm("帮我把富友收款弄一下")

        self.assertFalse(result["should_clarify"])
        self.assertEqual(result["source"], "llm_error")

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_llm_classifier_does_not_force_how_to_question_into_ticket(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = (
            '{"should_clarify": true, "reason": "operation_required", '
            '"message": "需要先补齐工单信息", "confidence": 0.9}'
        )
        mock_get_llm.return_value = llm

        result = classify_operation_query_with_llm("\u5fae\u4fe1\u51fa\u73b0\u5c01\u7981\u7684\u60c5\u51b5\u8981\u600e\u4e48\u529e")

        self.assertFalse(result["should_clarify"])
        self.assertEqual(result["reason"], "knowledge_question")
        self.assertEqual(result["source"], "llm_guard")

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_llm_classifier_does_not_force_capability_question_into_ticket(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = (
            '{"should_clarify": true, "reason": "operation_required", '
            '"message": "需要先补齐工单信息", "confidence": 0.9}'
        )
        mock_get_llm.return_value = llm

        result = classify_operation_query_with_llm(
            "\u5e97\u9762\u4e4b\u524d\u5f00\u901a\u7684\u5458\u5de5\u7684\u76f4\u64ad\u6743\u9650\u53ef\u4ee5\u5173\u95ed\u5417"
        )

        self.assertFalse(result["should_clarify"])
        self.assertEqual(result["reason"], "knowledge_question")
        self.assertEqual(result["source"], "llm_guard")

    def test_specific_query_does_not_require_clarification(self):
        result = should_clarify_query("苹果用户如何接受 21V 平台邀请？")

        self.assertFalse(result["should_clarify"])
        self.assertEqual(result["message"], "")

    def test_quality_passed_answer_is_resolved_by_ai(self):
        status = classify_ticket_status(
            used_fallback=False,
            quality_passed=True,
            confidence=0.78,
            fallback_reason=None,
        )

        self.assertEqual(status, "resolved_ai")

    def test_fallback_answer_requires_manual_work(self):
        status = classify_ticket_status(
            used_fallback=True,
            quality_passed=False,
            confidence=0.81,
            fallback_reason="Answer states knowledge base has no relevant content",
        )

        self.assertEqual(status, "pending_manual")

    def test_low_confidence_answer_requires_manual_work(self):
        status = classify_ticket_status(
            used_fallback=False,
            quality_passed=True,
            confidence=0.32,
            fallback_reason=None,
        )

        self.assertEqual(status, "pending_manual")

    def test_context_snapshots_preserve_chunk_references_and_content(self):
        sources = [
            {
                "id": "chunk-1",
                "job_id": "job-1",
                "file_name": "policy.docx",
                "content": "召回上下文",
                "score": 0.91,
                "metadata": {"chunk_index": 7, "page": 2},
            }
        ]

        contexts = build_context_snapshots(sources)

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["chunk_id"], "chunk-1")
        self.assertEqual(contexts[0]["job_id"], "job-1")
        self.assertEqual(contexts[0]["file_name"], "policy.docx")
        self.assertEqual(contexts[0]["chunk_index"], 7)
        self.assertEqual(contexts[0]["content"], "召回上下文")
        self.assertEqual(contexts[0]["sort_order"], 0)

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_continue_only_does_not_start_new_clarification(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo

        result = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="帮我开通直播权限",
            channel="h5",
            continue_only=True,
        )

        self.assertIsNone(result)
        self.assertIsNone(repo.ticket)

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_continue_only_does_not_hijack_independent_knowledge_question(self, mock_repo):
        repo = FakeClarificationRepo()
        repo.ticket = {
            "id": "ticket-1",
            "session_id": "session-1",
            "user_id": "store-1",
            "kb_name": "fushang",
            "status": "clarifying",
            "clarification_round": 1,
            "clarification": {
                "reason": "operation_required",
                "original_query": "帮我开通直播权限",
                "required_fields": ["issue_detail", "phone", "image"],
                "turns": [{"round": 1, "query": "帮我开通直播权限"}],
                "collected": {"issue_detail": "帮我开通直播权限"},
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
            query="\u5fae\u4fe1\u51fa\u73b0\u5c01\u7981\u7684\u60c5\u51b5\u8981\u600e\u4e48\u529e",
            channel="h5",
            continue_only=True,
        )

        self.assertIsNone(result)
        self.assertEqual(repo.ticket["clarification_round"], 1)

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_clarification_flow_reuses_ticket_and_turns_manual_after_enough_info(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo

        first = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="kb",
            query="帮我开通直播权限",
            channel="h5",
        )
        second = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="kb",
            query="手机号 13800138000，账号是门店A",
            channel="h5",
        )
        third = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="kb",
            query="身份证 110101199003074512，已经上传截图",
            channel="h5",
            has_image=True,
            query_image_oss_key="query_images/store-1/kb/session-1/a.jpg",
        )

        self.assertEqual(first["ticket_id"], "ticket-1")
        self.assertEqual(second["ticket_id"], "ticket-1")
        self.assertEqual(third["ticket_id"], "ticket-1")
        self.assertEqual(first["status"], "clarifying")
        self.assertEqual(second["status"], "clarifying")
        self.assertEqual(third["status"], "pending_manual")
        self.assertIn("已为您记录问题", third["answer"])
        self.assertIn("工单号 ticket-1", third["answer"])
        self.assertEqual(repo.ticket["clarification_round"], 3)
        self.assertEqual(repo.ticket["clarification"]["collected"]["phone"], "13800138000")
        self.assertEqual(repo.ticket["clarification"]["collected"]["id_card"], "110101199003074512")
        self.assertEqual(repo.ticket["clarification"]["collected"]["image_keys"], ["query_images/store-1/kb/session-1/a.jpg"])

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_clarification_uses_workflow_fields_and_contexts_when_available(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo
        workflow = {
            "workflow_found": True,
            "workflow_summary": "知识库流程：富友收款处理需要门店信息、手机号和收款页面截图。",
            "required_fields": ["issue_detail", "phone", "image"],
            "field_labels": {
                "issue_detail": "门店名称、富友账号或收款问题现象",
                "phone": "可联系手机号",
                "image": "富友收款页面或报错截图",
            },
            "fallback_used": False,
        }
        sources = [
            {
                "id": "chunk-1",
                "job_id": "job-1",
                "file_name": "富友流程.docx",
                "content": "富友收款处理需提供门店、手机号、截图。",
                "score": 0.88,
                "metadata": {"chunk_index": 2},
            }
        ]

        result = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="帮我把富友收款弄一下",
            channel="h5",
            reason="operation_required",
            force_start=True,
            workflow=workflow,
            workflow_sources=sources,
        )

        self.assertEqual(result["status"], "clarifying")
        self.assertIn("知识库流程", result["answer"])
        self.assertIn("可联系手机号", result["answer"])
        self.assertFalse(repo.ticket["clarification"]["fallback_used"])
        self.assertEqual(repo.ticket["clarification"]["required_fields"], ["issue_detail", "phone", "image"])
        self.assertEqual(repo.ticket["clarification"]["workflow_summary"], workflow["workflow_summary"])
        self.assertEqual(repo.ticket["contexts"][0]["chunk_id"], "chunk-1")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_active_clarification_reuses_existing_workflow_fields(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo
        workflow = {
            "workflow_found": True,
            "workflow_summary": "知识库流程：需要手机号。",
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
            query="富友那边帮忙安排一下",
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
        self.assertEqual(repo.ticket["clarification"]["required_fields"], ["phone"])
        self.assertEqual(repo.ticket["clarification"]["missing_fields"], [])
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "fields_complete")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "complete")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_workflow_without_summary_still_mentions_workflow(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo

        result = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="富友那边帮忙安排一下",
            channel="h5",
            reason="operation_required",
            force_start=True,
            workflow={
                "workflow_found": True,
                "workflow_summary": "",
                "required_fields": ["phone"],
                "field_labels": {"phone": "可联系手机号"},
                "fallback_used": False,
            },
            workflow_sources=[],
        )

        self.assertIn("查到相关流程", result["answer"])
        self.assertIn("可联系手机号", result["answer"])

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_workflow_mentions_flow_even_when_required_fields_present(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo

        result = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="张店长",
            sender_id=None,
            requester_name="张店长",
            kb_name="fushang",
            query="富友那边帮忙安排一下",
            channel="h5",
            reason="operation_required",
            force_start=True,
            workflow={
                "workflow_found": True,
                "workflow_summary": "富友处理需要先确认门店问题",
                "required_fields": ["issue_detail"],
                "field_labels": {"issue_detail": "门店问题描述"},
                "fallback_used": False,
            },
            workflow_sources=[],
        )

        self.assertEqual(result["status"], "pending_manual")
        self.assertIn("已为您记录问题", result["answer"])
        self.assertIn("工单号 ticket-1", result["answer"])
        self.assertEqual(result["clarification"]["exit_reason"], "fields_complete")
        self.assertEqual(result["clarification"]["completion_status"], "complete")


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
        self.assertFalse(_is_user_refusal("手机号不知道，门店是A店，截图已传"))

    def test_intent_helper_contracts_include_missing_knowledge_class(self):
        self.assertEqual(_intent_class_for_reason("knowledge_missing"), INTENT_MISSING_KNOWLEDGE)
        self.assertEqual(_intent_class_for_reason("ambiguous_query"), INTENT_AMBIGUOUS)
        self.assertEqual(_intent_class_for_reason("operation_required"), INTENT_OPERATION)
        self.assertEqual(_max_rounds_for_intent(INTENT_MISSING_KNOWLEDGE), 5)
        self.assertEqual(_max_rounds_for_intent(INTENT_AMBIGUOUS), 3)
        self.assertEqual(_max_rounds_for_intent(INTENT_OPERATION), 20)
        self.assertFalse(_is_complete(INTENT_MISSING_KNOWLEDGE, []))
        self.assertTrue(_is_complete(INTENT_OPERATION, []))
        self.assertEqual(_exit_reason_for_round_cap(INTENT_MISSING_KNOWLEDGE), "max_rounds")
        self.assertEqual(_exit_reason_for_round_cap(INTENT_AMBIGUOUS), "semantic_unresolved")

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
                "original_query": "\u77e5\u8bc6\u5e93\u6ca1\u6709\u7684\u65b0\u95ee\u9898",
                "required_fields": ["issue_detail", "phone"],
                "missing_fields": ["phone"],
                "collected": {"issue_detail": "\u77e5\u8bc6\u5e93\u6ca1\u6709\u7684\u65b0\u95ee\u9898"},
                "turns": [
                    {"round": 1, "query": "\u77e5\u8bc6\u5e93\u6ca1\u6709\u7684\u65b0\u95ee\u9898"},
                    {"round": 2, "query": "\u8fd8\u662f\u6ca1\u6709\u7b54\u6848"},
                    {"round": 3, "query": "\u9700\u8981\u540e\u53f0\u786e\u8ba4"},
                    {"round": 4, "query": "\u8bf7\u7ee7\u7eed\u8bb0\u5f55"},
                ],
                "kb_result": {"hit": False},
            },
        }
        mock_repo.return_value = repo

        result = process_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="\u5f20\u5e97\u957f",
            sender_id=None,
            requester_name="\u5f20\u5e97\u957f",
            kb_name="fushang",
            query="\u624b\u673a\u53f7 13800138000",
            channel="h5",
            continue_only=True,
        )

        self.assertEqual(result["status"], "pending_manual")
        self.assertEqual(result["finish_reason"], "manual_ticket_created")
        self.assertEqual(repo.ticket["clarification_round"], 5)
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "max_rounds")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "incomplete")

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
        self.assertIn("工单号 ticket-1", result["clarification"]["turns"][-1]["answer"])
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
        non_fallback_no_relevant = should_start_missing_knowledge_flow(
            {
                "used_fallback": False,
                "fallback_reason": "no_relevant_knowledge",
                "quality_passed": False,
                "confidence": 0.2,
                "sources": [],
            }
        )
        non_fallback_low_confidence = should_start_missing_knowledge_flow(
            {
                "used_fallback": False,
                "fallback_reason": None,
                "quality_passed": True,
                "confidence": 0.2,
                "sources": [],
            }
        )

        self.assertTrue(should_start)
        self.assertFalse(should_skip)
        self.assertFalse(non_fallback_no_relevant)
        self.assertFalse(non_fallback_low_confidence)

    def test_missing_knowledge_flow_is_defensive_for_malformed_inputs(self):
        self.assertFalse(should_start_missing_knowledge_flow(None))
        self.assertFalse(should_start_missing_knowledge_flow("no_relevant_knowledge"))
        self.assertFalse(should_start_missing_knowledge_flow({}))
        self.assertFalse(
            should_start_missing_knowledge_flow(
                {
                    "used_fallback": True,
                    "fallback_reason": "",
                    "quality_passed": True,
                    "confidence": "",
                    "sources": [],
                }
            )
        )
        self.assertFalse(
            should_start_missing_knowledge_flow(
                {
                    "used_fallback": True,
                    "fallback_reason": "",
                    "quality_passed": True,
                    "confidence": "low",
                    "sources": [],
                }
            )
        )
        self.assertTrue(
            should_start_missing_knowledge_flow(
                {
                    "used_fallback": True,
                    "fallback_reason": "no_relevant_knowledge",
                    "quality_passed": False,
                    "confidence": "low",
                    "sources": [],
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
