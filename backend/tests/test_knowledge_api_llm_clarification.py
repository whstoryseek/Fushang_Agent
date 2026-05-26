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

    async def test_a_classifier_before_deadline_returns_ticket_response(self):
        rag_started = asyncio.Event()
        rag_cancelled = asyncio.Event()

        async def slow_rag(**kwargs):
            rag_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                rag_cancelled.set()
                raise

        decision = {
            "intent_class": "A",
            "needs_ticket_flow": True,
            "fields": {"issue_detail": "open permission"},
            "missing_fields": [],
            "question": "",
        }
        ticket = {
            "answer": "ticket created",
            "confidence": 0.0,
            "finish_reason": "manual_ticket_created",
            "clarification": {"intent_class": "A"},
        }

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            side_effect=slow_rag,
        ), patch("app.api.v1.knowledge.classify_ticket_intent_with_llm", return_value=decision), patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn",
            return_value=ticket,
        ) as process_ticket, patch("app.api.v1.knowledge.persist_knowledge_result") as persist:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="please open permission", session_id="session-1", collection="kb"),
                user_id="store-1",
            )

        self.assertTrue(rag_started.is_set())
        self.assertTrue(rag_cancelled.is_set())
        self.assertEqual(response.answer, "ticket created")
        self.assertEqual(response.finish_reason, "manual_ticket_created")
        self.assertTrue(response.thoughts["clarification_required"])
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


if __name__ == "__main__":
    unittest.main()
