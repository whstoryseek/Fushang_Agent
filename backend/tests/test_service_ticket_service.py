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
    load_ticket_history_context,
    process_ticket_clarification_turn,
    record_unanswered_normal_ticket,
    resolve_active_clarification_as_unanswered_normal,
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

        self.assertIn("只要 query 已经包含以下任意两类信息，就视为语义清晰，不得判 C", prompt)
        self.assertIn("完整问句但缺少账号、门店、订单号、手机号、截图，不算 C", prompt)
        self.assertIn("线下办理字段是否收齐", prompt)
        self.assertIn("严禁复制、改写、复述 query", prompt)
        self.assertIn("发生入口、涉及账号/门店、截图、影响范围、订单/红包类型等具体字段", prompt)
        self.assertIn("操作需求", prompt)

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_llm_b_with_rag_no_relevant_fallback_starts_ticket_flow(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.side_effect = [
            json.dumps(
                {
                    "intent_class": "B",
                    "reason": "knowledge_missing",
                    "needs_ticket_flow": False,
                    "confidence": 0.82,
                    "fields": {},
                    "missing_fields": ["issue_detail"],
                    "question": "please provide more detail",
                    "rationale_brief": "RAG missed",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "action": "进入工单流",
                    "reason": "确认是知识库未覆盖问题，需要继续收集信息",
                    "question": "please provide more detail",
                },
                ensure_ascii=False,
            ),
        ]
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
        llm.responses_text.side_effect = [
            json.dumps(
                {
                    "intent_class": "B",
                    "reason": "knowledge_missing",
                    "needs_ticket_flow": False,
                    "confidence": 0.82,
                    "fields": {},
                    "missing_fields": ["issue_detail"],
                    "question": "please provide more detail",
                    "rationale_brief": "RAG missed",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "action": "进入工单流",
                    "reason": "确认是知识库未覆盖问题，需要继续收集信息",
                    "question": "please provide more detail",
                },
                ensure_ascii=False,
            ),
        ]
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
    def test_classifier_review_failure_allows_rag_for_specific_cause_questions(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.side_effect = [
            json.dumps(
                {
                    "intent_class": "C",
                    "reason": "ambiguous_query",
                    "needs_ticket_flow": True,
                    "confidence": 0.74,
                    "fields": {},
                    "missing_fields": ["issue_detail"],
                    "question": "请补充发生入口、涉及账号/门店、截图，以及影响范围（单个账号还是多名用户）。",
                    "rationale_brief": "误判为需要澄清",
                },
                ensure_ascii=False,
            ),
            RuntimeError("review timeout"),
        ]
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(
            query="客户积分一直在扣是什么原因？",
            history=[],
            has_image=False,
            rag_result=None,
        )

        self.assertEqual(result["intent_class"], INTENT_NORMAL)
        self.assertFalse(result["needs_ticket_flow"])
        self.assertEqual(result["question"], "")
        self.assertEqual(result["source"], "llm_review_fallback")

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_classifier_review_failure_allows_rag_for_missing_knowledge_candidate(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.side_effect = [
            json.dumps(
                {
                    "intent_class": "B",
                    "reason": "knowledge_missing",
                    "needs_ticket_flow": True,
                    "confidence": 0.81,
                    "fields": {},
                    "missing_fields": ["issue_detail"],
                    "question": "请补充问题现象、场景和影响范围。",
                    "rationale_brief": "candidate wants clarification",
                },
                ensure_ascii=False,
            ),
            RuntimeError("review timeout"),
        ]
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(
            query="顾客能进直播，但是直播画面显示没有画面怎么办？",
            history=[],
            has_image=False,
            rag_result={
                "used_fallback": True,
                "fallback_reason": "no_relevant_documents",
                "quality_passed": False,
                "sources": [],
            },
        )

        self.assertEqual(result["intent_class"], INTENT_NORMAL)
        self.assertFalse(result["needs_ticket_flow"])
        self.assertEqual(result["question"], "")
        self.assertEqual(result["source"], "llm_review_fallback")

    def test_classifier_prompt_protects_specific_reason_and_solution_questions(self):
        prompt = service_ticket_service._classifier_prompt(
            query="企业微信无法发视频是什么原因？",
            history=[],
            has_image=False,
            rag_result=None,
        )[0]["content"]

        self.assertIn("企业微信无法发视频是什么原因", prompt)
        self.assertIn("客户积分一直在扣是什么原因", prompt)
        self.assertIn("富友账户有钱，但下单时显示余额不足怎么办", prompt)
        self.assertIn("必须判 D", prompt)
        self.assertIn("即使 rag_result.used_fallback=true", prompt)
        self.assertIn("最高优先级", prompt)
        self.assertIn("只要 query 自身已经足够让 RAG 去检索，就判 D", prompt)

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_classifier_llm_review_corrects_missing_knowledge_reason_question_to_rag(self, mock_get_llm):
        llm = MagicMock()
        llm.responses_text.side_effect = [
            json.dumps(
                {
                    "intent_class": "B",
                    "reason": "knowledge_missing",
                    "needs_ticket_flow": True,
                    "confidence": 0.79,
                    "fields": {},
                    "missing_fields": ["issue_detail"],
                    "question": "请补充发生入口、涉及账号/门店、截图，以及影响范围（单个账号还是多名用户）。",
                    "rationale_brief": "误判为知识库未覆盖需追问",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "action": "放行RAG",
                    "reason": "清楚的原因类问题，即使 RAG fallback 也不应前台追问",
                    "question": "",
                },
                ensure_ascii=False,
            ),
        ]
        mock_get_llm.return_value = llm

        result = classify_ticket_intent_with_llm(
            query="富友账户有钱，但下单时显示余额不足怎么办",
            history=[],
            has_image=False,
            rag_result={
                "used_fallback": True,
                "fallback_reason": "no_relevant_documents",
                "answer": "可先排查账户类型、冻结金额或支付通道余额。",
                "sources": [],
            },
        )

        self.assertEqual(result["intent_class"], INTENT_NORMAL)
        self.assertFalse(result["needs_ticket_flow"])
        self.assertEqual(result["question"], "")
        self.assertEqual(result["source"], "llm_review")
        self.assertEqual(llm.responses_text.call_count, 2)
        review_prompt = llm.responses_text.call_args_list[1].args[0][0]["content"]
        self.assertIn("RAG 之后把用户问题转入前台工单追问", review_prompt)
        self.assertIn("即使 rag_result.used_fallback=true", review_prompt)
        self.assertIn("最高优先级", review_prompt)
        self.assertIn("缺少这些字段，不得作为拦成 C/B 的理由", review_prompt)
        self.assertIn("只要 query 自身已经足够让 RAG 检索，就放行RAG", review_prompt)

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
        self.assertIn("不要因为仍然缺少账号、门店、订单号、截图，就阻止转成 D", prompt)


class FakeClarificationRepo:
    def __init__(self):
        self.ticket = None
        self.replaced_contexts = None
        self.transaction_calls = []

    def find_active_clarification(self, **kwargs):
        return self.ticket

    def run_clarification_transaction(self, *, session_id, user_id, kb_name, callback):
        self.transaction_calls.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "kb_name": kb_name,
            }
        )
        return callback(self.ticket, None)

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

    def replace_contexts(self, ticket_id, contexts, conn=None):
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
    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_process_ticket_clarification_turn_uses_locked_transaction(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo

        result = process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="help me bind account",
            channel="h5",
            decision={
                "intent_class": "A",
                "reason": "operation_required",
                "needs_ticket_flow": True,
                "fields": {"issue_detail": "bind account"},
                "missing_fields": [],
                "question": "",
                "source": "llm",
            },
        )

        self.assertEqual(result["status"], "clarifying")
        self.assertEqual(len(repo.transaction_calls), 1)
        self.assertEqual(repo.transaction_calls[0]["session_id"], "session-1")

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_ticket_image_field_analyzer_uses_dedicated_multimodal_model(self, mock_get_llm):
        llm = MagicMock()
        llm.chat_with_images.return_value = json.dumps(
            {
                "fields": {
                    "business_license": {
                        "value": "已上传营业执照图片",
                        "document_type": "营业执照",
                        "confidence": 0.91,
                        "extracted": {"company_name": "示例公司"},
                    }
                },
                "document_types": ["营业执照"],
                "image_summary": "图片中包含营业执照。",
                "rationale_brief": "识别到营业执照版式和企业名称。",
            },
            ensure_ascii=False,
        )
        mock_get_llm.return_value = llm

        result = service_ticket_service.analyze_ticket_image_fields_with_llm(
            query="营业执照见图",
            active_ticket={
                "clarification": {
                    "intent_class": "A",
                    "required_fields": ["issue_detail", "business_license"],
                    "missing_fields": ["business_license"],
                    "collected": {"issue_detail": "开通微信子商户号"},
                }
            },
            image_url="https://example.test/license.jpg",
            query_image_oss_key="query_images/session-1/license.jpg",
        )

        self.assertIn("business_license", result["fields"])
        self.assertEqual(result["fields"]["business_license"]["source"], "image")
        self.assertEqual(result["fields"]["business_license"]["oss_key"], "query_images/session-1/license.jpg")
        llm.chat_with_images.assert_called_once()
        kwargs = llm.chat_with_images.call_args.kwargs
        self.assertEqual(kwargs["model"], "doubao-seed-2-0-pro-260215")
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertTrue(kwargs["disable_thinking"])
        messages = llm.chat_with_images.call_args.args[0]
        self.assertIn("字段名不要翻译", messages[0]["content"])
        self.assertEqual(messages[1]["content"][1]["image"], "https://example.test/license.jpg")

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_ticket_image_field_analyzer_accepts_multiple_images(self, mock_get_llm):
        llm = MagicMock()
        llm.chat_with_images.return_value = json.dumps(
            {
                "fields": {
                    "business_license": {"value": "已上传营业执照图片", "document_type": "营业执照"},
                    "legal_person_id_card": {"value": "已上传法人身份证图片", "document_type": "身份证"},
                },
                "document_types": ["营业执照", "身份证"],
                "image_summary": "两张图片分别是营业执照和身份证。",
                "rationale_brief": "图片资料与缺失字段匹配。",
            },
            ensure_ascii=False,
        )
        mock_get_llm.return_value = llm

        result = service_ticket_service.analyze_ticket_image_fields_with_llm(
            query="资料见图",
            active_ticket={
                "clarification": {
                    "intent_class": "A",
                    "required_fields": ["business_license", "legal_person_id_card"],
                    "missing_fields": ["business_license", "legal_person_id_card"],
                    "collected": {},
                }
            },
            image_urls=["https://example.test/license.jpg", "https://example.test/id-card.jpg"],
            query_image_oss_keys=["query_images/license.jpg", "query_images/id-card.jpg"],
        )

        self.assertEqual(set(result["fields"].keys()), {"business_license", "legal_person_id_card"})
        self.assertEqual(result["fields"]["business_license"]["oss_keys"], ["query_images/license.jpg", "query_images/id-card.jpg"])
        messages = llm.chat_with_images.call_args.args[0]
        image_parts = [part for part in messages[1]["content"] if "image" in part]
        self.assertEqual([part["image"] for part in image_parts], [
            "https://example.test/license.jpg",
            "https://example.test/id-card.jpg",
        ])

    @patch("app.services.service_ticket_service.get_llm_service")
    def test_ticket_image_field_analyzer_includes_required_field_details_for_authorization(self, mock_get_llm):
        llm = MagicMock()
        llm.chat_with_images.return_value = json.dumps(
            {
                "fields": {
                    "legal_authorization": {
                        "value": "已上传法人授权材料图片",
                        "document_type": "法人授权材料",
                    }
                },
                "document_types": ["法人授权材料"],
                "image_summary": "图片中包含法人授权材料。",
                "rationale_brief": "识别到授权委托书样式。",
            },
            ensure_ascii=False,
        )
        mock_get_llm.return_value = llm

        result = service_ticket_service.analyze_ticket_image_fields_with_llm(
            query="资料见图",
            active_ticket={
                "clarification": {
                    "intent_class": "A",
                    "required_fields": ["legal_authorization"],
                    "required_field_details": [
                        {"key": "legal_authorization", "label": "法人授权材料", "reason": "绑定流程要求"}
                    ],
                    "missing_fields": ["legal_authorization"],
                    "collected": {},
                }
            },
            image_url="https://example.test/auth.jpg",
            query_image_oss_key="query_images/session-1/auth.jpg",
        )

        self.assertIn("legal_authorization", result["fields"])
        self.assertEqual(result["fields"]["legal_authorization"]["document_type"], "法人授权材料")
        self.assertEqual(result["fields"]["legal_authorization"]["oss_key"], "query_images/session-1/auth.jpg")
        messages = llm.chat_with_images.call_args.args[0]
        self.assertIn("法人授权材料", messages[1]["content"][0]["text"])

    def test_image_field_analysis_can_complete_operation_ticket(self):
        image_analysis = {
            "fields": {
                "business_license": {
                    "value": "已上传营业执照图片",
                    "document_type": "营业执照",
                    "source": "image",
                    "oss_key": "query_images/session-1/license.jpg",
                    "confidence": 0.91,
                }
            },
            "image_summary": "图片中包含营业执照。",
        }
        decision = service_ticket_service.merge_ticket_image_analysis_into_decision(
            {
                "intent_class": "A",
                "reason": "clarification_reply",
                "needs_ticket_flow": True,
                "fields": {},
                "missing_fields": ["business_license"],
                "question": "请上传营业执照。",
                "source": "llm",
            },
            image_analysis,
            {
                "clarification": {
                    "intent_class": "A",
                    "required_fields": ["issue_detail", "business_license"],
                    "missing_fields": ["business_license"],
                    "collected": {"issue_detail": "开通微信子商户号"},
                }
            },
        )
        repo = FakeClarificationRepo()
        repo.ticket = {
            "id": "ticket-1",
            "session_id": "session-1",
            "user_id": "store-1",
            "kb_name": "fushang",
            "status": "clarifying",
            "clarification_round": 1,
            "clarification": {
                "intent_class": "A",
                "reason": "operation_required",
                "original_query": "帮我开通微信子商户号",
                "required_fields": ["issue_detail", "business_license"],
                "missing_fields": ["business_license"],
                "collected": {"issue_detail": "开通微信子商户号"},
                "workflow_found": True,
                "workflow_summary": "微信子商户号开通 SOP",
                "workflow_sources": [],
                "turns": [{"round": 1, "query": "帮我开通微信子商户号", "answer": "请上传营业执照。"}],
            },
            "contexts": [],
        }

        with patch("app.services.service_ticket_service.get_service_ticket_repository", return_value=repo):
            result = process_ticket_clarification_turn(
                session_id="session-1",
                user_id="store-1",
                user_name="store owner",
                sender_id=None,
                requester_name="store owner",
                kb_name="fushang",
                query="营业执照见图",
                channel="web",
                decision=decision,
                has_image=True,
                query_image_oss_key="query_images/session-1/license.jpg",
                continue_only=True,
            )

        self.assertEqual(result["status"], "pending_manual")
        self.assertEqual(result["clarification"]["completion_status"], "complete")
        self.assertEqual(result["clarification"]["missing_fields"], [])
        self.assertEqual(result["clarification"]["collected"]["business_license"]["source"], "image")
        self.assertEqual(result["clarification"]["image_analysis"]["field_analysis"]["image_summary"], "图片中包含营业执照。")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_operation_ticket_persists_required_field_details_for_followup_image_analysis(self, mock_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo

        process_ticket_clarification_turn(
            session_id="session-1",
            user_id="store-1",
            user_name="store owner",
            sender_id=None,
            requester_name="store owner",
            kb_name="fushang",
            query="帮我绑定一下富友账户和门店",
            channel="web",
            decision={
                "intent_class": "A",
                "reason": "operation_required",
                "needs_ticket_flow": True,
                "fields": {"issue_detail": "绑定富友账户和门店"},
                "missing_fields": [],
                "question": "",
            },
            workflow={
                "workflow_found": True,
                "required_fields": ["issue_detail", "legal_authorization"],
                "required_field_details": [
                    {"key": "legal_authorization", "label": "法人授权材料", "reason": "绑定流程要求"}
                ],
                "question": "请提供法人授权材料。",
            },
        )

        self.assertEqual(
            repo.ticket["clarification"]["required_field_details"],
            [{"key": "legal_authorization", "label": "法人授权材料", "reason": "绑定流程要求"}],
        )

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_active_ticket_records_multiple_uploaded_image_keys(self, mock_repo):
        repo = FakeClarificationRepo()
        repo.ticket = {
            "id": "ticket-1",
            "session_id": "session-1",
            "user_id": "store-1",
            "kb_name": "fushang",
            "status": "clarifying",
            "clarification_round": 1,
            "clarification": {
                "intent_class": "A",
                "reason": "operation_required",
                "original_query": "帮我开通微信子商户号",
                "required_fields": ["issue_detail", "business_license", "legal_person_id_card"],
                "missing_fields": ["business_license", "legal_person_id_card"],
                "collected": {"issue_detail": "开通微信子商户号"},
                "workflow_found": True,
                "turns": [{"round": 1, "query": "帮我开通微信子商户号", "answer": "请上传资料。"}],
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
            query="资料见图",
            channel="web",
            decision={
                "intent_class": "A",
                "reason": "clarification_reply",
                "needs_ticket_flow": True,
                "fields": {},
                "missing_fields": ["business_license", "legal_person_id_card"],
                "question": "请继续补充资料。",
            },
            has_image=True,
            query_image_oss_key="query_images/license.jpg",
            query_image_oss_keys=["query_images/license.jpg", "query_images/id-card.jpg"],
            continue_only=True,
        )

        clarification = repo.ticket["clarification"]
        self.assertEqual(clarification["collected"]["image_keys"], ["query_images/license.jpg", "query_images/id-card.jpg"])
        self.assertEqual(clarification["image_analysis"]["image_keys"], ["query_images/license.jpg", "query_images/id-card.jpg"])
        self.assertEqual(clarification["turns"][-1]["image_keys"], ["query_images/license.jpg", "query_images/id-card.jpg"])

    def test_load_ticket_history_context_merges_session_and_active_ticket_history(self):
        conv_repo = FakeConversationRepo(
            [
                {"role": "user", "content": "历史问题", "created_at": "2026-05-27 10:00:00"},
                {"role": "assistant", "content": "历史回答", "created_at": "2026-05-27 10:00:01"},
            ]
        )
        active_ticket = {
            "clarification": {
                "chat_history": [
                    {"role": "user", "content": "请继续补充门店"},
                    {"role": "assistant", "content": "还需要营业执照"},
                ]
            }
        }

        with patch("app.db.get_conversation_repository", return_value=conv_repo):
            history = load_ticket_history_context("session-1", active_ticket=active_ticket, limit=8)

        self.assertEqual(
            history,
            [
                {"role": "user", "content": "历史问题", "created_at": "2026-05-27 10:00:00"},
                {"role": "assistant", "content": "历史回答", "created_at": "2026-05-27 10:00:01"},
                {"role": "user", "content": "请继续补充门店"},
                {"role": "assistant", "content": "还需要营业执照"},
            ],
        )

    def test_record_unanswered_normal_ticket_persists_multi_images_and_contexts(self):
        repo = FakeClarificationRepo()
        conv_repo = FakeConversationRepo(
            [
                {"role": "user", "content": "为什么新活动没显示"},
                {"role": "assistant", "content": "知识库暂未命中"},
            ]
        )

        with patch("app.services.service_ticket_service.get_service_ticket_repository", return_value=repo), patch(
            "app.db.get_conversation_repository", return_value=conv_repo
        ):
            ticket = record_unanswered_normal_ticket(
                session_id="session-1",
                user_id="store-1",
                user_name="store owner",
                kb_name="fushang",
                query="为什么新活动没显示",
                answer="当前知识库暂无相关信息",
                confidence=0.21,
                rag_result={
                    "fallback_reason": "no_relevant_documents",
                    "quality_level": "low",
                    "sources": [
                        {
                            "chunk_id": "chunk-1",
                            "job_id": "job-1",
                            "file_name": "campaign.docx",
                            "chunk_index": 2,
                            "content": "child chunk",
                            "score": 0.56,
                            "metadata": {"parent_content": "parent context"},
                        }
                    ],
                },
                channel="web",
                requester_name="store owner",
                entry_user_id="wangqizhi_l8el",
                entry_user_name="农资店王麒麟",
                entry_source="renruikeji_sso",
                has_image=True,
                query_image_oss_key="query_images/a.jpg",
                query_image_oss_keys=["query_images/a.jpg", "query_images/b.jpg"],
            )

        self.assertEqual(ticket["status"], "unanswered_normal")
        self.assertEqual(ticket["entry_user_id"], "wangqizhi_l8el")
        self.assertEqual(ticket["entry_user_name"], "农资店王麒麟")
        self.assertEqual(ticket["entry_source"], "renruikeji_sso")
        self.assertEqual(ticket["clarification"]["image_analysis"]["query_image_oss_keys"], ["query_images/a.jpg", "query_images/b.jpg"])
        self.assertEqual(ticket["clarification"]["turns"][0]["image_keys"], ["query_images/a.jpg", "query_images/b.jpg"])
        self.assertEqual(ticket["clarification"]["chat_history"][-1]["content"], "当前知识库暂无相关信息")
        self.assertEqual(ticket["contexts"][0]["metadata"]["parent_content"], "parent context")

    def test_resolve_active_clarification_as_unanswered_normal_updates_ticket_and_contexts(self):
        repo = FakeClarificationRepo()
        repo.ticket = {
            "id": "ticket-1",
            "session_id": "session-1",
            "user_id": "store-1",
            "kb_name": "fushang",
            "status": "clarifying",
            "clarification_round": 2,
            "clarification": {
                "intent_class": "C",
                "turns": [{"round": 1, "query": "这个怎么弄", "answer": "请补充场景"}],
            },
        }
        conv_repo = FakeConversationRepo(
            [
                {"role": "user", "content": "这个怎么弄"},
                {"role": "assistant", "content": "请补充场景"},
            ]
        )

        with patch("app.services.service_ticket_service.get_service_ticket_repository", return_value=repo), patch(
            "app.db.get_conversation_repository", return_value=conv_repo
        ):
            ticket = resolve_active_clarification_as_unanswered_normal(
                active_ticket=repo.ticket,
                query="是活动配置刷新后没生效",
                answer="当前知识库暂无相关信息",
                rag_result={
                    "fallback_reason": "no_relevant_documents",
                    "quality_level": "low",
                    "sources": [
                        {
                            "chunk_id": "chunk-2",
                            "job_id": "job-2",
                            "file_name": "activity.docx",
                            "chunk_index": 1,
                            "content": "retrieved child chunk",
                            "score": 0.48,
                            "metadata": {"parent_content": "retrieved parent context"},
                        }
                    ],
                },
                has_image=True,
                query_image_oss_key="query_images/activity-a.jpg",
                query_image_oss_keys=["query_images/activity-a.jpg", "query_images/activity-b.jpg"],
                requester_name="store owner",
                user_name="store owner",
                entry_user_id="wangqizhi_l8el",
                entry_user_name="农资店王麒麟",
                entry_source="renruikeji_sso",
            )

        self.assertEqual(ticket["status"], "unanswered_normal")
        self.assertEqual(ticket["clarification_round"], 3)
        self.assertEqual(ticket["clarification"]["exit_reason"], "returned_to_rag_but_unanswered")
        self.assertEqual(ticket["clarification"]["chat_history"][-2]["content"], "是活动配置刷新后没生效")
        self.assertEqual(ticket["clarification"]["chat_history"][-2]["query_image_oss_keys"], ["query_images/activity-a.jpg", "query_images/activity-b.jpg"])
        self.assertEqual(repo.replaced_contexts[0], "ticket-1")
        self.assertEqual(repo.replaced_contexts[1][0]["metadata"]["parent_content"], "retrieved parent context")

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
                entry_user_id="wangqizhi_l8el",
                entry_user_name="农资店王麒麟",
                entry_source="renruikeji_sso",
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
        self.assertEqual(repo.ticket["entry_user_id"], "wangqizhi_l8el")
        self.assertEqual(repo.ticket["entry_user_name"], "农资店王麒麟")
        self.assertEqual(repo.ticket["entry_source"], "renruikeji_sso")

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

    @patch("app.db.get_chunk_repository")
    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_new_ticket_backfills_parent_child_context_from_chunk_repository(self, mock_repo, mock_get_chunk_repo):
        repo = FakeClarificationRepo()
        mock_repo.return_value = repo
        chunk_repo = MagicMock()
        chunk_repo.get_by_ids_with_file_names.return_value = [
            {
                "chunk_id": "11111111-1111-1111-1111-111111111111",
                "job_id": "22222222-2222-2222-2222-222222222222",
                "file_name": "payment-guide.docx",
                "chunk_index": 7,
                "content": "child chunk about recharge failure",
                "metadata": {
                    "chunk_strategy": "parent_child",
                    "parent_id": "job-1-parent-0",
                    "parent_content": "parent chunk full recharge troubleshooting context",
                },
            }
        ]
        mock_get_chunk_repo.return_value = chunk_repo
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
            "chunk_index": 7,
            "content": "child chunk about recharge failure",
            "score": 0.91,
            "metadata": {
                "chunk_strategy": "parent_child",
                "parent_id": "job-1-parent-0",
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

        chunk_repo.get_by_ids_with_file_names.assert_called_once_with(["11111111-1111-1111-1111-111111111111"])
        context = repo.ticket["contexts"][0]
        self.assertEqual(context["file_name"], "payment-guide.docx")
        self.assertEqual(context["metadata"]["parent_content"], "parent chunk full recharge troubleshooting context")

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
    def xest_operation_without_sop_finalizes_manual_ticket(self, mock_repo):
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
    def xest_operation_without_workflow_finalizes_as_no_standard_workflow(self, mock_repo):
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
    def test_operation_without_sop_keeps_clarifying_first(self, mock_repo):
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

        self.assertEqual(result["status"], "clarifying")
        self.assertEqual(result["finish_reason"], "clarification")
        self.assertEqual(repo.ticket["clarification"]["intent_class"], "A")
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "incomplete")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_operation_without_workflow_keeps_clarifying(self, mock_repo):
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

        self.assertEqual(result["status"], "clarifying")
        self.assertEqual(result["finish_reason"], "clarification")
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "incomplete")

    @patch("app.services.service_ticket_service.get_service_ticket_repository")
    def test_operation_without_workflow_finalizes_after_round_limit(self, mock_repo):
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

        first = process_ticket_clarification_turn(
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
        follow_up = None
        for idx in range(1, service_ticket_service.MAX_CLARIFICATION_ROUNDS[INTENT_OPERATION]):
            follow_up = process_ticket_clarification_turn(
                session_id="session-1",
                user_id="store-1",
                user_name="store owner",
                sender_id=None,
                requester_name="store owner",
                kb_name="fushang",
                query=f"follow-up-{idx}",
                channel="h5",
                decision=decision,
                continue_only=True,
            )

        self.assertEqual(first["status"], "clarifying")
        self.assertIsNotNone(follow_up)
        self.assertEqual(follow_up["status"], "pending_manual")
        self.assertEqual(follow_up["finish_reason"], "manual_ticket_created")
        self.assertEqual(repo.ticket["clarification"]["exit_reason"], "max_rounds")
        self.assertEqual(repo.ticket["clarification"]["completion_status"], "incomplete")

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
