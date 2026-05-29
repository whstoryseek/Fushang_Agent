# -*- coding: utf-8 -*-
import asyncio
import sys
import threading
import time
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

if "dashscope" not in sys.modules:
    dashscope_module = types.ModuleType("dashscope")
    dashscope_module.TextEmbedding = types.SimpleNamespace(call=lambda *args, **kwargs: None)
    dashscope_module.api_key = ""
    sys.modules["dashscope"] = dashscope_module

if "pymilvus" not in sys.modules:
    pymilvus_module = types.ModuleType("pymilvus")
    pymilvus_module.MilvusClient = object
    pymilvus_module.DataType = types.SimpleNamespace(
        VARCHAR="VARCHAR",
        INT64="INT64",
        SPARSE_FLOAT_VECTOR="SPARSE_FLOAT_VECTOR",
        FLOAT_VECTOR="FLOAT_VECTOR",
    )
    pymilvus_module.Function = object
    pymilvus_module.FunctionType = types.SimpleNamespace(BM25="BM25")
    pymilvus_module.AnnSearchRequest = object
    pymilvus_module.RRFRanker = object
    pymilvus_module.WeightedRanker = object
    sys.modules["pymilvus"] = pymilvus_module

from app.api.v1 import knowledge
from app.models.requests import KnowledgeRequest


def _rag_result(**overrides):
    result = {
        "request_id": "req-1",
        "session_id": "session-1",
        "answer": "normal rag answer",
        "confidence": 0.86,
        "sources": [{"file_name": "policy.md"}],
        "model": "doubao-seed-2-0-pro-260215",
        "thoughts": {"query_analysis": {}},
        "image_map": None,
        "finish_reason": "stop",
        "used_fallback": False,
        "fallback_reason": None,
        "quality_passed": True,
        "quality_level": "high",
    }
    result.update(overrides)
    return result


class KnowledgeApiTicketRaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_text_returns_rag_when_classifier_is_pending(self):
        release_classifier = threading.Event()

        def slow_classifier(*args, **kwargs):
            release_classifier.wait(0.25)
            return {
                "intent_class": "D",
                "needs_ticket_flow": False,
                "fields": {},
                "missing_fields": [],
            }

        async def fake_rag(**kwargs):
            self.assertFalse(kwargs["persist"])
            return _rag_result()

        with patch("app.api.v1.knowledge.TICKET_CLASSIFIER_INTERCEPT_TIMEOUT", 0.01), patch(
            "app.api.v1.knowledge.conversation_service.ensure_knowledge_session",
            return_value="session-1",
        ), patch("app.api.v1.knowledge.invoke_knowledge_qa", side_effect=fake_rag), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            side_effect=slow_classifier,
        ), patch("app.api.v1.knowledge.persist_knowledge_result") as persist, patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn"
        ) as process_ticket:
            start = time.monotonic()
            try:
                response = await knowledge.knowledge_qa(
                    KnowledgeRequest(query="how to configure", session_id="session-1"),
                    user_id="store-1",
                )
            finally:
                release_classifier.set()
            elapsed = time.monotonic() - start

        self.assertLess(elapsed, 0.2)
        self.assertEqual(response.answer, "normal rag answer")
        self.assertEqual(response.finish_reason, "stop")
        persist.assert_called_once()
        process_ticket.assert_not_called()

    async def test_normal_rag_answer_records_resolved_ai_ticket(self):
        async def fake_rag(**kwargs):
            return _rag_result(
                answer="只有门店负责人才有开户权限。",
                sources=[{"file_name": "wechat-submerchant.docx", "content": "开户步骤"}],
            )

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            side_effect=fake_rag,
        ), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            return_value={"intent_class": "D", "needs_ticket_flow": False, "fields": {}, "missing_fields": []},
        ), patch("app.api.v1.knowledge.persist_knowledge_result") as persist, patch(
            "app.api.v1.knowledge.record_ai_resolved_ticket",
            create=True,
        ) as record_ticket:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="微信子商户号开户资料是什么？", session_id="session-1", collection="kb"),
                user_id="store-1",
            )

        self.assertEqual(response.finish_reason, "stop")
        persist.assert_called_once()
        record_ticket.assert_called_once()
        self.assertEqual(record_ticket.call_args.kwargs["status"], "resolved_ai")
        self.assertEqual(record_ticket.call_args.kwargs["query"], "微信子商户号开户资料是什么？")
        self.assertEqual(record_ticket.call_args.kwargs["user_id"], "store-1")

    async def test_a_classifier_before_deadline_uses_rag_candidates_for_llm_sop_analysis(self):
        rag_started = asyncio.Event()

        async def fake_rag(**kwargs):
            rag_started.set()
            return _rag_result(
                answer="SOP says prepare permission applicant and store.",
                sources=[{"file_name": "permission-sop.docx", "content": "prepare applicant and store"}],
            )

        decision = {
            "intent_class": "A",
            "needs_ticket_flow": True,
            "fields": {"issue_detail": "open permission"},
            "missing_fields": [],
            "question": "",
        }
        ticket = {
            "answer": "please provide applicant and store",
            "confidence": 0.0,
            "finish_reason": "clarification",
            "clarification": {"intent_class": "A", "workflow_found": True},
        }

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            side_effect=fake_rag,
        ), patch("app.api.v1.knowledge.classify_ticket_intent_with_llm", return_value=decision), patch(
            "app.api.v1.knowledge.lookup_operation_workflow",
            return_value={
                "workflow_found": True,
                "required_fields": ["issue_detail", "store"],
                "question": "please provide applicant and store",
                "workflow_sources": [{"file_name": "permission-sop.docx"}],
            },
        ) as lookup, patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn",
            return_value=ticket,
        ) as process_ticket, patch("app.api.v1.knowledge.persist_knowledge_result") as persist:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="please open permission", session_id="session-1", collection="kb"),
                user_id="store-1",
            )

        self.assertTrue(rag_started.is_set())
        self.assertEqual(response.answer, "please provide applicant and store")
        self.assertEqual(response.finish_reason, "clarification")
        self.assertTrue(response.thoughts["clarification_required"])
        lookup.assert_called_once()
        self.assertEqual(lookup.call_args.kwargs["rag_result"]["sources"][0]["file_name"], "permission-sop.docx")
        process_ticket.assert_called_once()
        self.assertTrue(process_ticket.call_args.kwargs["workflow"]["workflow_found"])
        persist.assert_not_called()

    async def test_classifier_finishing_after_deadline_but_before_rag_can_still_intercept(self):
        def slightly_slow_classifier(*args, **kwargs):
            time.sleep(0.03)
            return {
                "intent_class": "C",
                "needs_ticket_flow": True,
                "fields": {},
                "missing_fields": ["issue_detail"],
                "question": "Please clarify the business scenario.",
            }

        async def slower_rag(**kwargs):
            await asyncio.sleep(0.08)
            return _rag_result()

        ticket = {
            "answer": "Please clarify the business scenario.",
            "confidence": 0.0,
            "finish_reason": "clarification",
            "clarification": {"intent_class": "C"},
        }

        with patch("app.api.v1.knowledge.TICKET_CLASSIFIER_INTERCEPT_TIMEOUT", 0.01), patch(
            "app.api.v1.knowledge.conversation_service.ensure_knowledge_session",
            return_value="session-1",
        ), patch("app.api.v1.knowledge.invoke_knowledge_qa", side_effect=slower_rag), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            side_effect=slightly_slow_classifier,
        ), patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn",
            return_value=ticket,
        ) as process_ticket, patch("app.api.v1.knowledge.persist_knowledge_result") as persist:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="unclear thing", session_id="session-1"),
                user_id="store-1",
            )

        self.assertEqual(response.finish_reason, "clarification")
        self.assertEqual(response.thoughts["clarification"]["intent_class"], "C")
        process_ticket.assert_called_once()
        persist.assert_not_called()

    async def test_rag_miss_triggers_b_ticket_after_proven_miss(self):
        decisions = [
            {"intent_class": "D", "needs_ticket_flow": False, "fields": {}, "missing_fields": []},
            {
                "intent_class": "B",
                "needs_ticket_flow": True,
                "fields": {"issue_detail": "new issue"},
                "missing_fields": ["impact"],
                "question": "What is the business impact?",
            },
        ]

        async def fake_rag(**kwargs):
            return _rag_result(
                answer="No relevant knowledge found",
                confidence=0.2,
                used_fallback=True,
                fallback_reason="no_relevant_documents",
                quality_passed=False,
                quality_level="low",
            )

        ticket = {
            "answer": "What is the business impact?",
            "confidence": 0.0,
            "finish_reason": "clarification",
            "clarification": {"intent_class": "B"},
        }

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            side_effect=fake_rag,
        ), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            side_effect=decisions,
        ) as classify, patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn",
            return_value=ticket,
        ) as process_ticket, patch("app.api.v1.knowledge.persist_knowledge_result") as persist:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="brand new edge case", session_id="session-1", collection="kb"),
                user_id="store-1",
            )

        self.assertEqual(response.answer, "What is the business impact?")
        self.assertEqual(response.finish_reason, "clarification")
        self.assertEqual(classify.call_count, 2)
        self.assertEqual(classify.call_args.kwargs["rag_result"]["fallback_reason"], "no_relevant_documents")
        process_ticket.assert_called_once()
        persist.assert_not_called()

    async def test_unproven_fallback_stays_normal_rag(self):
        async def fake_rag(**kwargs):
            return _rag_result(
                used_fallback=True,
                fallback_reason="low_confidence_answer",
                quality_passed=False,
                quality_level="low",
            )

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            side_effect=fake_rag,
        ), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            return_value={"intent_class": "D", "needs_ticket_flow": False, "fields": {}, "missing_fields": []},
        ) as classify, patch("app.api.v1.knowledge.process_ticket_clarification_turn") as process_ticket, patch(
            "app.api.v1.knowledge.persist_knowledge_result"
        ) as persist:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="normal but weak", session_id="session-1"),
                user_id="store-1",
            )

        self.assertEqual(response.answer, "normal rag answer")
        self.assertEqual(classify.call_count, 1)
        process_ticket.assert_not_called()
        persist.assert_called_once()

    async def test_active_clarification_reply_updates_ticket_before_rag(self):
        active = {
            "id": "ticket-1",
            "clarification_round": 1,
            "clarification": {
                "intent_class": "A",
                "original_query": "帮我开通权限",
                "required_fields": ["issue_detail", "phone", "store"],
                "missing_fields": ["phone", "store"],
                "collected": {"issue_detail": "开通权限"},
            },
        }
        continuation = {
            "intent_class": "A",
            "needs_ticket_flow": True,
            "fields": {"phone": "13800138000"},
            "missing_fields": ["store"],
            "question": "请继续补充门店名称。",
        }
        ticket = {
            "answer": "请继续补充门店名称。",
            "confidence": 0.0,
            "finish_reason": "clarification",
            "clarification": {"intent_class": "A", "missing_fields": ["store"]},
        }

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.find_active_clarification_ticket",
            return_value=active,
            create=True,
        ) as find_active, patch(
            "app.api.v1.knowledge.analyze_clarification_reply_with_llm",
            return_value=continuation,
            create=True,
        ) as analyze, patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn",
            return_value=ticket,
        ) as process_ticket, patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            return_value=_rag_result(),
        ) as rag, patch("app.api.v1.knowledge.persist_knowledge_result") as persist:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="手机号 13800138000", session_id="session-1", collection="kb"),
                user_id="store-1",
            )

        self.assertEqual(response.finish_reason, "clarification")
        find_active.assert_called_once()
        analyze.assert_called_once()
        process_ticket.assert_called_once()
        self.assertTrue(process_ticket.call_args.kwargs["continue_only"])
        self.assertEqual(process_ticket.call_args.kwargs["decision"]["fields"]["phone"], "13800138000")
        rag.assert_not_called()
        persist.assert_not_called()

    async def test_active_clarification_reply_can_return_to_rag_and_resolve_ticket(self):
        active = {
            "id": "ticket-1",
            "clarification_round": 1,
            "clarification": {
                "intent_class": "C",
                "original_query": "这个怎么弄",
                "turns": [{"round": 1, "query": "这个怎么弄", "answer": "请补充具体场景。"}],
            },
        }
        continuation = {
            "intent_class": "D",
            "needs_ticket_flow": False,
            "fields": {},
            "missing_fields": [],
            "question": "",
            "resolved_query": "微信子商户号开户需要什么资料？",
        }

        async def fake_rag(**kwargs):
            self.assertEqual(kwargs["query"], "微信子商户号开户需要什么资料？")
            return _rag_result(answer="需要营业执照、法人信息和银行卡。")

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.find_active_clarification_ticket",
            return_value=active,
            create=True,
        ), patch(
            "app.api.v1.knowledge.analyze_clarification_reply_with_llm",
            return_value=continuation,
            create=True,
        ), patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            side_effect=fake_rag,
        ), patch(
            "app.api.v1.knowledge.resolve_active_clarification_with_rag",
            create=True,
        ) as resolve_ticket, patch("app.api.v1.knowledge.persist_knowledge_result") as persist, patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn"
        ) as process_ticket:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="微信子商户号开户需要什么资料？", session_id="session-1", collection="kb"),
                user_id="store-1",
            )

        self.assertEqual(response.finish_reason, "stop")
        self.assertIn("营业执照", response.answer)
        resolve_ticket.assert_called_once()
        process_ticket.assert_not_called()
        persist.assert_called_once()


    async def test_active_clarification_reply_rag_miss_resolves_as_unanswered_normal_without_reentering_ticket_flow(self):
        active = {
            "id": "ticket-1",
            "clarification_round": 1,
            "clarification": {
                "intent_class": "C",
                "original_query": "这个怎么弄",
                "turns": [{"round": 1, "query": "这个怎么弄", "answer": "请补充具体场景。"}],
            },
        }
        continuation = {
            "intent_class": "D",
            "needs_ticket_flow": False,
            "fields": {},
            "missing_fields": [],
            "question": "",
            "resolved_query": "顾客能进直播，但是直播画面显示没有画面怎么办？",
        }

        async def fake_rag(**kwargs):
            self.assertEqual(kwargs["query"], "顾客能进直播，但是直播画面显示没有画面怎么办？")
            return _rag_result(
                answer="No relevant knowledge found",
                confidence=0.2,
                used_fallback=True,
                fallback_reason="no_relevant_documents",
                quality_passed=False,
                quality_level="low",
                sources=[],
            )

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.find_active_clarification_ticket",
            return_value=active,
            create=True,
        ), patch(
            "app.api.v1.knowledge.analyze_clarification_reply_with_llm",
            return_value=continuation,
            create=True,
        ), patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            side_effect=fake_rag,
        ), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            return_value={"intent_class": "D", "needs_ticket_flow": False, "fields": {}, "missing_fields": []},
        ) as classify, patch(
            "app.api.v1.knowledge.resolve_active_clarification_as_unanswered_normal",
            create=True,
        ) as resolve_unanswered, patch(
            "app.api.v1.knowledge.persist_knowledge_result"
        ) as persist, patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn"
        ) as process_ticket:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="直播没有画面", session_id="session-1", collection="kb"),
                user_id="store-1",
            )

        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.answer, "No relevant knowledge found")
        classify.assert_called_once()
        resolve_unanswered.assert_called_once()
        process_ticket.assert_not_called()
        persist.assert_called_once()


if __name__ == "__main__":
    unittest.main()
