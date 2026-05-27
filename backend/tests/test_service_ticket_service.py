# -*- coding: utf-8 -*-
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from app.core.exceptions import NotFoundError

if "psycopg2" not in sys.modules:
    psycopg2_module = types.ModuleType("psycopg2")
    psycopg2_module.pool = types.SimpleNamespace(ThreadedConnectionPool=object)
    psycopg2_module.extras = types.SimpleNamespace(RealDictCursor=object)
    sys.modules["psycopg2"] = psycopg2_module
    sys.modules["psycopg2.pool"] = psycopg2_module.pool
    sys.modules["psycopg2.extras"] = psycopg2_module.extras

from app.services.service_ticket_service import (
    analyze_clarification_reply_with_llm,
    analyze_operation_workflow_with_llm,
    INTENT_AMBIGUOUS,
    INTENT_MISSING_KNOWLEDGE,
    INTENT_NORMAL,
    INTENT_OPERATION,
    classify_ticket_intent_with_llm,
    process_ticket_clarification_turn,
)
from app.services import service_ticket_service


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
        self.assertGreaterEqual(kwargs["timeout"], 2.5)
        self.assertLessEqual(kwargs["timeout"], 4.0)
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

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_llm_b_without_rag_miss_does_not_start_ticket_flow(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = json.dumps(
            {
                "intent_class": "B",
                "reason": "knowledge_missing",
                "needs_ticket_flow": True,
                "confidence": 0.82,
                "fields": {},
                "missing_fields": ["issue_detail"],
                "question": "请补充问题细节。",
                "rationale_brief": "LLM thinks knowledge is missing",
            },
            ensure_ascii=False,
        )
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(
            query="一个新问题",
            history=[],
            has_image=False,
            rag_result=None,
        )

        self.assertEqual(result["intent_class"], INTENT_MISSING_KNOWLEDGE)
        self.assertFalse(result["needs_ticket_flow"])

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_llm_b_with_rag_no_relevant_fallback_starts_ticket_flow(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = json.dumps(
            {
                "intent_class": "B",
                "reason": "knowledge_missing",
                "needs_ticket_flow": False,
                "confidence": 0.82,
                "fields": {},
                "missing_fields": ["issue_detail"],
                "question": "请补充问题细节。",
                "rationale_brief": "RAG missed",
            },
            ensure_ascii=False,
        )
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(
            query="一个新问题",
            history=[],
            has_image=False,
            rag_result={"used_fallback": True, "fallback_reason": "no_relevant_documents"},
        )

        self.assertEqual(result["intent_class"], INTENT_MISSING_KNOWLEDGE)
        self.assertTrue(result["needs_ticket_flow"])

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_llm_b_with_real_shaped_missing_reason_starts_ticket_flow(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = json.dumps(
            {
                "intent_class": "B",
                "reason": "knowledge_missing",
                "needs_ticket_flow": False,
                "confidence": 0.82,
                "fields": {},
                "missing_fields": ["issue_detail"],
                "question": "请补充问题细节。",
                "rationale_brief": "RAG missed",
            },
            ensure_ascii=False,
        )
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(
            query="一个新问题",
            history=[],
            has_image=False,
            rag_result={
                "used_fallback": True,
                "fallback_reason": "Answer states knowledge base has no relevant content",
            },
        )

        self.assertEqual(result["intent_class"], INTENT_MISSING_KNOWLEDGE)
        self.assertTrue(result["needs_ticket_flow"])

    def test_classifier_prompt_requires_llm_question_self_check_for_error_queries(self):
        prompt = service_ticket_service._classifier_prompt(
            query="企业微信发不了红包，提示该单已被其他账号发起支付，你无权再发起",
            history=[],
            has_image=False,
            rag_result={
                "used_fallback": True,
                "fallback_reason": "Answer states knowledge base has no relevant content",
                "quality_passed": False,
                "quality_level": "low",
                "sources": [],
            },
        )[0]["content"]

        self.assertIn("明确业务对象/系统名 + 现象/报错/提示语，就不是 C", prompt)
        self.assertIn("严禁复制、改写、复述 query", prompt)
        self.assertIn("发生入口、涉及账号/门店、截图，以及影响范围", prompt)
        self.assertIn("操作需求", prompt)

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_classifier_llm_review_corrects_ambiguous_error_query_to_rag(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.side_effect = [
            json.dumps(
                {
                    "intent_class": "C",
                    "reason": "ambiguous_query",
                    "needs_ticket_flow": True,
                    "confidence": 0.82,
                    "fields": {},
                    "missing_fields": ["issue_detail"],
                    "question": "请您描述一下具体遇到的问题，以及涉及的业务系统或功能",
                    "rationale_brief": "candidate incorrectly wants clarification",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "action": "allow_rag",
                    "reason": "query already has system object and error text",
                    "question": "",
                },
                ensure_ascii=False,
            ),
        ]
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(
            query="企业微信发不了红包，提示该单已被其他账号发起支付，你无权再发起",
            history=[],
            has_image=False,
            rag_result=None,
        )

        self.assertEqual(result["intent_class"], INTENT_NORMAL)
        self.assertFalse(result["needs_ticket_flow"])
        self.assertEqual(result["question"], "")
        self.assertEqual(llm.responses_text.call_count, 2)
        review_prompt = llm.responses_text.call_args_list[1].args[0][0]["content"]
        self.assertIn("是否应该在 RAG 之前拦截", review_prompt)

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_classifier_review_prompt_uses_chinese_action_values(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.side_effect = [
            json.dumps(
                {
                    "intent_class": "C",
                    "reason": "ambiguous_query",
                    "needs_ticket_flow": True,
                    "confidence": 0.82,
                    "fields": {},
                    "missing_fields": ["issue_detail"],
                    "question": "请补充业务系统和报错原文。",
                    "rationale_brief": "candidate incorrectly wants clarification",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "action": "放行RAG",
                    "reason": "已有明确业务对象和报错原文",
                    "question": "",
                },
                ensure_ascii=False,
            ),
        ]
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(
            query="企业微信发不了红包，提示该单已被其他账号发起支付，你无权再发起",
            history=[],
            has_image=False,
            rag_result=None,
        )

        self.assertEqual(result["intent_class"], INTENT_NORMAL)
        review_prompt = llm.responses_text.call_args_list[1].args[0][0]["content"]
        self.assertIn("放行RAG", review_prompt)
        self.assertIn("进入工单流", review_prompt)
        self.assertNotIn("allow_rag", review_prompt)
        self.assertNotIn("ticket_flow", review_prompt)

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_llm_b_with_generic_quality_failure_does_not_start_ticket_flow(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = json.dumps(
            {
                "intent_class": "B",
                "reason": "knowledge_missing",
                "needs_ticket_flow": True,
                "confidence": 0.82,
                "fields": {},
                "missing_fields": ["issue_detail"],
                "question": "请补充问题细节。",
                "rationale_brief": "RAG answer quality failed",
            },
            ensure_ascii=False,
        )
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(
            query="一个新问题",
            history=[],
            has_image=False,
            rag_result={
                "used_fallback": True,
                "quality_passed": False,
                "fallback_reason": "Answer too short; Confidence below threshold",
            },
        )

        self.assertEqual(result["intent_class"], INTENT_MISSING_KNOWLEDGE)
        self.assertFalse(result["needs_ticket_flow"])

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_malformed_classifier_json_defaults_empty_text_without_image_to_c(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = "not json"
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(query="   ", history=[], has_image=False)

        self.assertEqual(result["intent_class"], INTENT_AMBIGUOUS)
        self.assertTrue(result["needs_ticket_flow"])
        self.assertEqual(result["source"], "fallback")

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_invalid_intent_falls_back_by_query_safety(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = json.dumps(
            {
                "intent_class": "Z",
                "reason": "unknown",
                "needs_ticket_flow": True,
                "confidence": 0.5,
                "fields": {},
                "missing_fields": [],
                "question": "",
                "rationale_brief": "",
            }
        )
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(query="正常知识问题", history=[], has_image=False)

        self.assertEqual(result["intent_class"], INTENT_NORMAL)
        self.assertFalse(result["needs_ticket_flow"])
        self.assertEqual(result["source"], "fallback")

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_fenced_json_is_parsed(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = (
            "```json\n"
            + json.dumps(
                {
                    "intent_class": "D",
                    "reason": "knowledge_question",
                    "needs_ticket_flow": False,
                    "confidence": 0.7,
                    "fields": {},
                    "missing_fields": [],
                    "question": "",
                    "rationale_brief": "tutorial",
                }
            )
            + "\n```"
        )
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(query="如何查看余额", history=[], has_image=False)

        self.assertEqual(result["intent_class"], INTENT_NORMAL)
        self.assertFalse(result["needs_ticket_flow"])
        self.assertEqual(result["source"], "llm")

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_non_dict_fields_and_non_list_missing_fields_are_normalized_empty(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = json.dumps(
            {
                "intent_class": "A",
                "reason": "operation_required",
                "needs_ticket_flow": True,
                "confidence": 0.9,
                "fields": ["bad"],
                "missing_fields": "phone",
                "question": "请补充手机号。",
                "rationale_brief": "operation",
            }
        )
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(query="帮我处理账户", history=[], has_image=False)

        self.assertEqual(result["intent_class"], INTENT_OPERATION)
        self.assertEqual(result["fields"], {})
        self.assertEqual(result["missing_fields"], [])

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_operation_workflow_analysis_uses_mini_llm_and_strict_json_contract(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = json.dumps(
            {
                "workflow_found": True,
                "confidence": 0.93,
                "workflow_summary": "微信子商户号开户 SOP",
                "required_fields": ["business_license", "legal_person", "bank_account"],
                "required_field_details": [
                    {"key": "business_license", "label": "营业执照", "reason": "SOP 要求准备营业执照"},
                    {"key": "legal_person", "label": "法人信息", "reason": "SOP 要求填写法人信息"},
                    {"key": "bank_account", "label": "银行账户", "reason": "SOP 要求填写结算账户"},
                ],
                "question": "请补充营业执照、法人信息和银行账户信息。",
                "workflow_sources": [{"file_name": "开通微信子商户号.docx", "chunk_index": 0}],
                "rationale_brief": "候选内容包含准备资料和开户流程",
            },
            ensure_ascii=False,
        )
        mock_get_llm.return_value = llm

        result = analyze_operation_workflow_with_llm(
            query="帮我开通微信子商户号",
            decision={"intent_class": "A", "fields": {"issue_detail": "开通微信子商户号"}},
            rag_result={
                "answer": "开通微信子商户号需要营业执照、法人信息和银行账户。",
                "sources": [
                    {
                        "file_name": "开通微信子商户号.docx",
                        "chunk_index": 0,
                        "content": "开户流程：准备营业执照、法人信息、银行账户，进入门店收款码管理提交。",
                    }
                ],
            },
        )

        self.assertTrue(result["workflow_found"])
        self.assertIn("business_license", result["required_fields"])
        self.assertEqual(result["question"], "请补充营业执照、法人信息和银行账户信息。")
        kwargs = llm.responses_text.call_args.kwargs
        self.assertEqual(kwargs["model"], "doubao-seed-2-0-mini-260428")
        self.assertEqual(kwargs["timeout"], service_ticket_service.WORKFLOW_ANALYSIS_TIMEOUT)
        self.assertEqual(kwargs["max_retries"], 0)
        prompt = llm.responses_text.call_args.args[0][0]["content"]
        self.assertIn("只能输出一个合法 JSON 对象", prompt)
        self.assertIn("不能根据关键词、常识或用户意图自行猜测", prompt)

    def test_operation_workflow_analysis_without_candidates_returns_no_workflow(self):
        result = analyze_operation_workflow_with_llm(
            query="帮我开通微信子商户号",
            decision={"intent_class": "A"},
            rag_result=None,
        )

        self.assertFalse(result["workflow_found"])
        self.assertEqual(result["required_fields"], [])

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_operation_workflow_analysis_sends_parent_context_to_llm(self, mock_get_llm):
        llm = MagicMock()

        def fake_responses_text(messages, **_kwargs):
            payload = json.loads(messages[1]["content"])
            source = payload["knowledge_candidates"]["sources"][0]
            parent_context = source.get("parent_content") or ""
            if "准备资料" not in parent_context or "法人身份证" not in parent_context:
                return json.dumps(
                    {
                        "workflow_found": False,
                        "confidence": 0.2,
                        "workflow_summary": "",
                        "required_fields": [],
                        "required_field_details": [],
                        "question": "",
                        "workflow_sources": [],
                        "rationale_brief": "缺少父块 SOP 上下文",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "workflow_found": True,
                    "confidence": 0.92,
                    "workflow_summary": "开通微信子商户号 SOP",
                    "required_fields": ["business_license", "legal_person_id_card", "bank_account"],
                    "required_field_details": [
                        {"key": "business_license", "label": "营业执照", "reason": "SOP 准备资料要求"},
                        {"key": "legal_person_id_card", "label": "法人身份证", "reason": "SOP 准备资料要求"},
                        {"key": "bank_account", "label": "基本存款账户信息", "reason": "SOP 准备资料要求"},
                    ],
                    "question": "请补充营业执照、法人身份证和基本存款账户信息。",
                    "workflow_sources": [{"file_name": "开通微信子商户号.docx", "chunk_index": 2}],
                    "rationale_brief": "父块包含准备资料和开户流程",
                },
                ensure_ascii=False,
            )

        llm.responses_text.side_effect = fake_responses_text
        mock_get_llm.return_value = llm

        result = analyze_operation_workflow_with_llm(
            query="帮我开通微信子商户号",
            decision={"intent_class": "A", "fields": {"issue_detail": "开通微信子商户号"}},
            rag_result={
                "answer": "",
                "sources": [
                    {
                        "file_name": "开通微信子商户号.docx",
                        "chunk_index": 2,
                        "content": "2开户流程在【新平台店长端-门店收款码管理-微信子商户号】菜单中进行提交资料开户。",
                        "metadata": {
                            "parent_content": (
                                "【培训】开通微信子商户号\n"
                                "开户流程\n"
                                "2.1准备资料-法人的信息\n"
                                "营业执照、法人身份证、法人手机号邮箱、基本存款账户信息。\n"
                                "2.2开户流程：在新平台店长端提交资料开户。"
                            )
                        },
                    }
                ],
            },
        )

        self.assertTrue(result["workflow_found"])
        self.assertIn("business_license", result["required_fields"])

    def test_operation_workflow_prompt_does_not_treat_empty_answer_as_no_sop(self):
        messages = service_ticket_service._workflow_analysis_prompt(
            "帮我开通微信子商户号",
            {"intent_class": "A", "fields": {"issue_detail": "开通微信子商户号"}},
            {
                "answer": "",
                "sources": [
                    {
                        "file_name": "开通微信子商户号.docx",
                        "chunk_index": 2,
                        "content": "开户流程在新平台店长端提交资料开户。",
                        "metadata": {"parent_content": "准备资料包含营业执照、法人身份证和基本存款账户信息。"},
                    }
                ],
            },
        )

        prompt = messages[0]["content"]
        self.assertIn("answer 为空不代表没有 SOP", prompt)
        self.assertIn("parent_content", prompt)
        self.assertIn("正例", prompt)
        self.assertIn("反例", prompt)

    def test_operation_workflow_prompt_flattens_parent_context_as_candidate_text(self):
        messages = service_ticket_service._workflow_analysis_prompt(
            "帮我开通微信子商户号",
            {"intent_class": "A", "fields": {"issue_detail": "开通微信子商户号"}},
            {
                "answer": "",
                "sources": [
                    {
                        "file_name": "开通微信子商户号.docx",
                        "chunk_index": 2,
                        "content": "开户流程在新平台店长端提交资料开户。",
                        "metadata": {"parent_content": "准备资料包含营业执照、法人身份证和基本存款账户信息。"},
                    }
                ],
            },
        )

        payload = json.loads(messages[1]["content"])
        candidate_text = payload["knowledge_candidates"].get("candidate_text", "")
        self.assertIn("开通微信子商户号.docx", candidate_text)
        self.assertIn("准备资料", candidate_text)
        self.assertIn("法人身份证", candidate_text)

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_clarification_reply_analysis_merges_context_with_mini_llm(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.return_value = json.dumps(
            {
                "intent_class": "A",
                "needs_ticket_flow": True,
                "fields": {"phone": "13800138000"},
                "missing_fields": ["store"],
                "question": "请继续补充门店名称。",
                "ready_for_manual": False,
                "user_refused": False,
                "resolved_query": "帮我开通权限，手机号 13800138000",
                "rationale_brief": "用户补充了手机号但缺门店",
            },
            ensure_ascii=False,
        )
        mock_get_llm.return_value = llm

        result = analyze_clarification_reply_with_llm(
            query="手机号 13800138000",
            active_ticket={
                "clarification": {
                    "intent_class": "A",
                    "original_query": "帮我开通权限",
                    "required_fields": ["issue_detail", "phone", "store"],
                    "missing_fields": ["phone", "store"],
                    "collected": {"issue_detail": "开通权限"},
                    "turns": [{"round": 1, "query": "帮我开通权限", "answer": "请补充手机号和门店。"}],
                },
                "clarification_round": 1,
            },
            has_image=False,
        )

        self.assertEqual(result["intent_class"], INTENT_OPERATION)
        self.assertTrue(result["needs_ticket_flow"])
        self.assertEqual(result["fields"]["phone"], "13800138000")
        self.assertEqual(result["missing_fields"], ["store"])
        kwargs = llm.responses_text.call_args.kwargs
        self.assertEqual(kwargs["model"], "doubao-seed-2-0-mini-260428")
        self.assertLessEqual(kwargs["max_tokens"], 500)
        self.assertEqual(kwargs["max_retries"], 0)
        prompt = llm.responses_text.call_args.args[0][0]["content"]
        self.assertIn("合并上下文", prompt)
        self.assertIn("ready_for_manual", prompt)


class FakeClarificationRepo:
    def __init__(self):
        self.ticket = None
        self.replaced_contexts = None

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

    def replace_contexts(self, ticket_id, contexts):
        self.replaced_contexts = (ticket_id, contexts)
        self.ticket["contexts"] = contexts


class FakeConversationRepo:
    def __init__(self, messages=None, error=None):
        self.messages = messages or []
        self.error = error
        self.calls = []

    def list_messages(self, session_id, limit=100):
        self.calls.append((session_id, limit))
        if self.error:
            raise self.error
        return self.messages


class FakeAdminTicketRepo:
    def __init__(self):
        self.list_kwargs = None
        self.count_kwargs = None
        self.stats_kwargs = None
        self.deleted_id = None
        self.delete_result = True

    def list(self, **kwargs):
        self.list_kwargs = kwargs
        return [{"id": "ticket-1", "status": "pending_manual"}]

    def count(self, **kwargs):
        self.count_kwargs = kwargs
        return 1

    def stats(self, **kwargs):
        self.stats_kwargs = kwargs
        return {"total": 1, "by_status": {"pending_manual": 1}, "daily": []}

    def delete(self, ticket_id):
        self.deleted_id = ticket_id
        return self.delete_result


class ServiceTicketAdminServiceTests(unittest.TestCase):
    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_list_tickets_returns_total_and_items_from_repository(self, mock_repo):
        repo = FakeAdminTicketRepo()
        mock_repo.return_value = repo

        result = service_ticket_service.list_tickets(
            status="pending_manual",
            kb_name="kb",
            user_id="store-1",
            start_date="2026-05-01",
            end_date="2026-05-27",
            limit=20,
            offset=0,
        )

        self.assertEqual(result, {"total": 1, "items": [{"id": "ticket-1", "status": "pending_manual"}]})
        self.assertEqual(repo.count_kwargs["status"], "pending_manual")
        self.assertEqual(repo.list_kwargs["kb_name"], "kb")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_ticket_stats_delegates_to_repository(self, mock_repo):
        repo = FakeAdminTicketRepo()
        mock_repo.return_value = repo

        result = service_ticket_service.ticket_stats(kb_name="kb", user_id="store-1")

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["by_status"]["pending_manual"], 1)
        self.assertEqual(repo.stats_kwargs["kb_name"], "kb")
        self.assertEqual(repo.stats_kwargs["user_id"], "store-1")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_delete_ticket_delegates_to_repository(self, mock_repo):
        repo = FakeAdminTicketRepo()
        mock_repo.return_value = repo

        result = service_ticket_service.delete_ticket("ticket-1")

        self.assertEqual(result, {"id": "ticket-1", "deleted": True})
        self.assertEqual(repo.deleted_id, "ticket-1")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_delete_ticket_raises_not_found_when_missing(self, mock_repo):
        repo = FakeAdminTicketRepo()
        repo.delete_result = False
        mock_repo.return_value = repo

        with self.assertRaises(NotFoundError):
            service_ticket_service.delete_ticket("missing-ticket")


class ServiceTicketStateTests(unittest.TestCase):
    def test_new_ticket_snapshots_conversation_history_and_current_turn(self):
        repo = FakeClarificationRepo()
        conv_repo = FakeConversationRepo(
            [
                {
                    "id": "msg-1",
                    "role": "user",
                    "content": "previous question",
                    "created_at": "2026-05-27 10:00:00",
                },
                {
                    "id": "msg-2",
                    "role": "assistant",
                    "content": "previous answer",
                    "sources": [{"file_name": "faq.docx"}],
                    "confidence": 0.82,
                    "created_at": "2026-05-27 10:00:01",
                },
            ]
        )
        decision = {
            "intent_class": "B",
            "reason": "knowledge_missing",
            "needs_ticket_flow": True,
            "fields": {"issue_detail": "payment cannot recharge"},
            "missing_fields": ["phone"],
            "question": "please provide phone",
            "source": "llm",
        }

        with patch("app.services.service_ticket_service.get_service_ticket_repository", return_value=repo), patch(
            "app.db.get_conversation_repository", return_value=conv_repo
        ):
            process_ticket_clarification_turn(
                session_id="session-1",
                user_id="store-1",
                user_name="store owner",
                sender_id=None,
                requester_name="store owner",
                kb_name="fushang",
                query="payment cannot recharge",
                channel="h5",
                decision=decision,
                query_image_oss_key="query_images/session-1/current.jpg",
                has_image=True,
            )

        chat_history = repo.ticket["clarification"]["chat_history"]
        self.assertEqual([item["role"] for item in chat_history], ["user", "assistant", "user", "assistant"])
        self.assertEqual(chat_history[0]["content"], "previous question")
        self.assertEqual(chat_history[1]["sources"], [{"file_name": "faq.docx"}])
        self.assertEqual(chat_history[2]["content"], "payment cannot recharge")
        self.assertEqual(chat_history[2]["query_image_oss_key"], "query_images/session-1/current.jpg")
        self.assertEqual(chat_history[3]["content"], "please provide phone")

    def test_active_ticket_keeps_existing_chat_history_when_conversation_read_fails(self):
        repo = FakeClarificationRepo()
        repo.ticket = {
            "id": "ticket-1",
            "session_id": "session-1",
            "user_id": "store-1",
            "kb_name": "fushang",
            "status": "clarifying",
            "clarification_round": 1,
            "clarification": {
                "intent_class": "C",
                "reason": "ambiguous_query",
                "original_query": "that issue",
                "required_fields": ["issue_detail"],
                "missing_fields": ["issue_detail"],
                "collected": {},
                "workflow_found": False,
                "workflow_summary": "",
                "workflow_sources": [],
                "kb_result": {},
                "image_analysis": {"has_image": False},
                "turns": [{"round": 1, "query": "that issue", "answer": "clarify please"}],
                "chat_history": [
                    {"role": "user", "content": "that issue"},
                    {"role": "assistant", "content": "clarify please"},
                ],
            },
            "contexts": [],
        }
        conv_repo = FakeConversationRepo(error=RuntimeError("db unavailable"))
        decision = {
            "intent_class": "C",
            "reason": "ambiguous_query",
            "needs_ticket_flow": True,
            "fields": {},
            "missing_fields": ["issue_detail"],
            "question": "please describe the issue",
            "source": "llm",
        }

        with patch("app.services.service_ticket_service.get_service_ticket_repository", return_value=repo), patch(
            "app.db.get_conversation_repository", return_value=conv_repo
        ):
            process_ticket_clarification_turn(
                session_id="session-1",
                user_id="store-1",
                user_name="store owner",
                sender_id=None,
                requester_name="store owner",
                kb_name="fushang",
                query="new detail",
                channel="h5",
                decision=decision,
                continue_only=True,
            )

        chat_history = repo.ticket["clarification"]["chat_history"]
        self.assertEqual([item["content"] for item in chat_history], [
            "that issue",
            "clarify please",
            "new detail",
            "please describe the issue",
        ])

    def test_active_ticket_backfills_chat_history_from_existing_turns(self):
        repo = FakeClarificationRepo()
        repo.ticket = {
            "id": "ticket-1",
            "session_id": "session-1",
            "user_id": "store-1",
            "kb_name": "fushang",
            "status": "clarifying",
            "clarification_round": 1,
            "clarification": {
                "intent_class": "C",
                "reason": "ambiguous_query",
                "original_query": "that issue",
                "required_fields": ["issue_detail"],
                "missing_fields": ["issue_detail"],
                "collected": {},
                "workflow_found": False,
                "workflow_summary": "",
                "workflow_sources": [],
                "kb_result": {},
                "image_analysis": {"has_image": False},
                "turns": [{"round": 1, "query": "that issue", "answer": "clarify please"}],
            },
            "contexts": [],
        }
        conv_repo = FakeConversationRepo(error=RuntimeError("db unavailable"))
        decision = {
            "intent_class": "C",
            "reason": "ambiguous_query",
            "needs_ticket_flow": True,
            "fields": {},
            "missing_fields": ["issue_detail"],
            "question": "please describe the issue",
            "source": "llm",
        }

        with patch("app.services.service_ticket_service.get_service_ticket_repository", return_value=repo), patch(
            "app.db.get_conversation_repository", return_value=conv_repo
        ):
            process_ticket_clarification_turn(
                session_id="session-1",
                user_id="store-1",
                user_name="store owner",
                sender_id=None,
                requester_name="store owner",
                kb_name="fushang",
                query="new detail",
                channel="h5",
                decision=decision,
                continue_only=True,
            )

        chat_history = repo.ticket["clarification"]["chat_history"]
        self.assertEqual([item["content"] for item in chat_history], [
            "that issue",
            "clarify please",
            "new detail",
            "please describe the issue",
        ])

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_ambiguous_active_ticket_can_reclassify_after_user_reply(self, mock_repo):
        repo = FakeClarificationRepo()
        repo.ticket = {
            "id": "ticket-1",
            "session_id": "session-1",
            "user_id": "store-1",
            "kb_name": "fushang",
            "status": "clarifying",
            "clarification_round": 1,
            "clarification": {
                "intent_class": "C",
                "reason": "ambiguous_query",
                "original_query": "这个怎么弄",
                "required_fields": ["issue_detail"],
                "missing_fields": ["issue_detail"],
                "collected": {},
                "workflow_found": False,
                "turns": [{"round": 1, "query": "这个怎么弄", "answer": "请补充具体问题。"}],
            },
            "contexts": [],
        }
        mock_repo.return_value = repo

        result = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="是一个新接口报错",
            channel="h5",
            decision={
                "intent_class": "B",
                "reason": "knowledge_missing",
                "needs_ticket_flow": True,
                "fields": {"issue_detail": "新接口报错"},
                "missing_fields": ["impact"],
                "question": "请补充影响范围。",
                "source": "llm",
            },
            continue_only=True,
        )

        self.assertEqual(result["status"], "clarifying")
        self.assertEqual(repo.ticket["clarification"]["intent_class"], "B")
        self.assertEqual(repo.ticket["clarification"]["collected"]["issue_detail"], "新接口报错")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_user_refusal_finalizes_active_ticket_for_manual_followup(self, mock_repo):
        repo = FakeClarificationRepo()
        repo.ticket = {
            "id": "ticket-1",
            "session_id": "session-1",
            "user_id": "store-1",
            "kb_name": "fushang",
            "status": "clarifying",
            "clarification_round": 1,
            "clarification": {
                "intent_class": "B",
                "reason": "knowledge_missing",
                "original_query": "新问题",
                "required_fields": ["issue_detail", "impact"],
                "missing_fields": ["impact"],
                "collected": {"issue_detail": "新问题"},
                "workflow_found": False,
                "turns": [{"round": 1, "query": "新问题", "answer": "请补充影响范围。"}],
            },
            "contexts": [],
        }
        mock_repo.return_value = repo

        result = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="不知道，不想填了",
            channel="h5",
            decision={
                "intent_class": "B",
                "reason": "user_refused",
                "needs_ticket_flow": True,
                "fields": {},
                "missing_fields": ["impact"],
                "question": "",
                "user_refused": True,
                "ready_for_manual": True,
                "source": "llm",
            },
            continue_only=True,
        )

        self.assertEqual(result["status"], "pending_manual")
        self.assertEqual(result["finish_reason"], "manual_ticket_created")
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "user_refused")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "incomplete")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_new_ticket_persists_rag_sources_as_context_snapshots(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo
        decision = {
            "intent_class": "B",
            "reason": "knowledge_missing",
            "needs_ticket_flow": True,
            "fields": {"issue_detail": "payment cannot recharge"},
            "missing_fields": ["phone"],
            "question": "please provide phone",
            "source": "llm",
        }
        source = {
            "chunk_id": "11111111-1111-1111-1111-111111111111",
            "job_id": "22222222-2222-2222-2222-222222222222",
            "file_name": "payment-guide.docx",
            "chunk_index": 7,
            "content": "child chunk about recharge failure",
            "score": 0.91,
            "metadata": {
                "chunk_strategy": "parent_child",
                "parent_id": "job-1-parent-0",
                "parent_content": "parent chunk full recharge troubleshooting context",
            },
        }

        process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="payment cannot recharge",
            channel="h5",
            decision=decision,
            rag_result={"sources": [source]},
        )

        self.assertEqual(len(repo.ticket["contexts"]), 1)
        context = repo.ticket["contexts"][0]
        self.assertEqual(context["chunk_id"], "11111111-1111-1111-1111-111111111111")
        self.assertEqual(context["job_id"], "22222222-2222-2222-2222-222222222222")
        self.assertEqual(context["file_name"], "payment-guide.docx")
        self.assertEqual(context["chunk_index"], 7)
        self.assertEqual(context["content"], "child chunk about recharge failure")
        self.assertEqual(context["metadata"]["parent_id"], "job-1-parent-0")
        self.assertIn("parent chunk full recharge", context["metadata"]["parent_content"])

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_active_ticket_backfills_contexts_from_later_rag_result(self, mock_repo):
        repo = FakeClarificationRepo()
        repo.ticket = {
            "id": "ticket-1",
            "session_id": "session-1",
            "user_id": "store-1",
            "kb_name": "fushang",
            "status": "clarifying",
            "clarification_round": 1,
            "clarification": {
                "intent_class": "C",
                "reason": "ambiguous_query",
                "original_query": "that issue",
                "required_fields": ["issue_detail"],
                "missing_fields": ["issue_detail"],
                "collected": {},
                "workflow_found": False,
                "workflow_summary": "",
                "workflow_sources": [],
                "kb_result": {},
                "image_analysis": {"has_image": False},
                "turns": [{"round": 1, "query": "that issue"}],
            },
            "contexts": [],
        }
        mock_repo.return_value = repo

        process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="it is recharge failure",
            channel="h5",
            decision={
                "intent_class": "B",
                "reason": "knowledge_missing",
                "needs_ticket_flow": True,
                "fields": {"issue_detail": "recharge failure"},
                "missing_fields": [],
                "question": "",
                "source": "llm",
            },
            rag_result={
                "sources": [
                    {
                        "chunk_id": "33333333-3333-3333-3333-333333333333",
                        "job_id": "44444444-4444-4444-4444-444444444444",
                        "file_name": "faq.docx",
                        "chunk_index": 2,
                        "content": "later retrieved child chunk",
                        "score": 0.8,
                        "metadata": {"chunk_strategy": "parent_child", "parent_content": "later parent context"},
                    }
                ]
            },
            continue_only=True,
        )

        self.assertEqual(repo.replaced_contexts[0], "ticket-1")
        self.assertEqual(repo.replaced_contexts[1][0]["chunk_id"], "33333333-3333-3333-3333-333333333333")
        self.assertEqual(repo.ticket["contexts"][0]["metadata"]["parent_content"], "later parent context")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_operation_without_sop_finalizes_manual_ticket(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo
        decision = {
            "intent_class": "A",
            "reason": "operation_required",
            "needs_ticket_flow": True,
            "fields": {"issue_detail": "bind account"},
            "missing_fields": [],
            "question": "",
            "source": "llm",
        }

        result = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="help me bind account",
            channel="h5",
            decision=decision,
            workflow={"workflow_found": False, "required_fields": []},
        )

        self.assertEqual(result["status"], "pending_manual")
        self.assertEqual(result["finish_reason"], "manual_ticket_created")
        self.assertIn("已为您记录问题，工单号 ticket-1，后续将由专人处理。", result["answer"])
        self.assertEqual(repo.ticket["clarification"]["intent_class"], "A")
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "no_standard_workflow")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "incomplete")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_operation_without_workflow_finalizes_as_no_standard_workflow(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo
        decision = {
            "intent_class": "A",
            "reason": "operation_required",
            "needs_ticket_flow": True,
            "fields": {"issue_detail": "bind account"},
            "missing_fields": [],
            "question": "",
            "source": "llm",
        }

        result = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="help me bind account",
            channel="h5",
            decision=decision,
        )

        self.assertEqual(result["status"], "pending_manual")
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "no_standard_workflow")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "incomplete")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_operation_workflow_analysis_failure_keeps_clarifying_instead_of_manual_ticket(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo
        decision = {
            "intent_class": "A",
            "reason": "operation_required",
            "needs_ticket_flow": True,
            "fields": {},
            "missing_fields": [],
            "question": "请补充企业主体信息、门店和联系方式。",
            "source": "llm",
        }

        result = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="帮我开通微信子商户号",
            channel="h5",
            decision=decision,
            workflow={
                "workflow_found": False,
                "confidence": 0.0,
                "required_fields": [],
                "question": "",
                "rationale_brief": "workflow_analysis_failed",
                "source": "fallback",
            },
        )

        self.assertEqual(result["status"], "clarifying")
        self.assertEqual(result["finish_reason"], "clarification")
        self.assertEqual(result["answer"], "请补充企业主体信息、门店和联系方式。")
        self.assertEqual(repo.ticket["clarification"]["workflow_source"], "fallback")
        self.assertEqual(repo.ticket["clarification"]["workflow_rationale"], "workflow_analysis_failed")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_operation_with_workflow_asks_workflow_question_for_missing_fields(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo
        decision = {
            "intent_class": "A",
            "reason": "operation_required",
            "needs_ticket_flow": True,
            "fields": {"issue_detail": "open wechat sub merchant"},
            "missing_fields": [],
            "question": "open wechat sub merchant",
            "source": "llm",
        }

        result = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="open wechat sub merchant",
            channel="h5",
            decision=decision,
            workflow={
                "workflow_found": True,
                "workflow_summary": "wechat sub merchant SOP",
                "workflow_sources": [{"title": "SOP"}],
                "required_fields": ["issue_detail", "business_license", "legal_person"],
                "question": "Please provide business license and legal person info.",
            },
        )

        self.assertEqual(result["status"], "clarifying")
        self.assertEqual(result["answer"], "Please provide business license and legal person info.")
        self.assertEqual(repo.ticket["answer"], "Please provide business license and legal person info.")
        self.assertEqual(repo.ticket["clarification"]["missing_fields"], ["business_license", "legal_person"])

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_operation_uses_original_query_as_issue_detail_when_llm_fields_empty(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo
        decision = {
            "intent_class": "A",
            "reason": "operation_required",
            "needs_ticket_flow": True,
            "fields": {},
            "missing_fields": [],
            "question": "",
            "source": "llm",
        }

        result = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="帮我开通微信子商户号",
            channel="h5",
            decision=decision,
            workflow={
                "workflow_found": True,
                "workflow_summary": "开通微信子商户号 SOP",
                "required_fields": ["issue_detail", "business_license", "legal_person_id_card"],
                "question": "请补充营业执照和法人身份证。",
            },
        )

        self.assertEqual(result["status"], "clarifying")
        self.assertEqual(repo.ticket["clarification"]["collected"]["issue_detail"], "帮我开通微信子商户号")
        self.assertEqual(repo.ticket["clarification"]["missing_fields"], ["business_license", "legal_person_id_card"])

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_operation_with_complete_required_fields_marks_complete(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo
        decision = {
            "intent_class": "A",
            "reason": "operation_required",
            "needs_ticket_flow": True,
            "fields": {"issue_detail": "bind account", "phone": "13800138000"},
            "missing_fields": [],
            "question": "",
            "source": "llm",
        }

        result = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="help me bind account, phone 13800138000",
            channel="h5",
            decision=decision,
            workflow={
                "workflow_found": True,
                "workflow_summary": "standard account binding SOP",
                "workflow_sources": [{"title": "SOP"}],
                "required_fields": ["issue_detail", "phone"],
            },
        )

        self.assertEqual(result["status"], "pending_manual")
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "fields_complete")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "complete")
        self.assertFalse(result["answer"].startswith("当前信息未完全收集。"))

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
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="this issue",
            channel="h5",
            decision=decision,
        )
        second = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="still unclear",
            channel="h5",
            decision=decision,
            continue_only=True,
        )
        third = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="that one",
            channel="h5",
            decision=decision,
            continue_only=True,
        )

        self.assertEqual(first["status"], "clarifying")
        self.assertEqual(second["status"], "clarifying")
        self.assertEqual(third["status"], "pending_manual")
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "semantic_unresolved")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "incomplete")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_missing_knowledge_finalizes_after_five_rounds_without_complete(self, mock_repo):
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
                "original_query": "new knowledge base gap",
                "required_fields": ["issue_detail", "phone"],
                "missing_fields": ["phone"],
                "collected": {"issue_detail": "new knowledge base gap"},
                "workflow_found": False,
                "workflow_summary": "",
                "workflow_sources": [],
                "kb_result": {"hit": False},
                "image_analysis": {"has_image": False, "categories": []},
                "turns": [{"round": i, "query": f"turn-{i}"} for i in range(1, 5)],
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
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="phone 13800138000",
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


if __name__ == "__main__":
    unittest.main()
