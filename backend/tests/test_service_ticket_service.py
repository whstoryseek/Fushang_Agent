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


if __name__ == "__main__":
    unittest.main()
