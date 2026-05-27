# -*- coding: utf-8 -*-
import asyncio
import sys
import types
import unittest
from unittest.mock import patch

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


class FakeTask:
    def __init__(self, result):
        self._result = result

    def done(self):
        return False

    def cancel(self):
        return None

    def __await__(self):
        async def _result():
            return self._result

        return _result().__await__()


def _kb_rag_result():
    return {
        "request_id": "req-kb",
        "session_id": "session-kb",
        "answer": "开通微信子商户号需要营业执照、法人信息、银行卡等资料。",
        "confidence": 0.88,
        "sources": [{"file_name": "开通微信子商户号.docx"}],
        "model": "doubao-seed-2-0-pro-260215",
        "thoughts": {"retrieval": {"chunks_retrieved": 3}},
        "image_map": {},
        "finish_reason": "stop",
        "used_fallback": False,
        "fallback_reason": None,
        "quality_passed": True,
        "quality_level": "high",
        "kb_name": "fushang",
    }


class KnowledgeFullFlowFromKbTests(unittest.IsolatedAsyncioTestCase):
    async def test_kb_tutorial_question_stays_normal_rag_d_flow(self):
        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-kb"), patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            return_value=_kb_rag_result(),
        ), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            return_value={"intent_class": "D", "needs_ticket_flow": False, "fields": {}, "missing_fields": []},
        ), patch("app.api.v1.knowledge.persist_knowledge_result") as persist, patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn"
        ) as ticket:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="微信子商户号开户需要准备哪些资料？", session_id="session-kb", collection="fushang"),
                user_id="store-1",
            )

        self.assertEqual(response.finish_reason, "stop")
        self.assertIn("营业执照", response.answer)
        self.assertEqual(response.sources[0]["file_name"], "开通微信子商户号.docx")
        persist.assert_called_once()
        ticket.assert_not_called()

    async def test_operation_with_kb_sop_asks_for_required_fields_before_manual_ticket(self):
        decision = {
            "intent_class": "A",
            "needs_ticket_flow": True,
            "fields": {"issue_detail": "开通微信子商户号"},
            "missing_fields": ["business_license", "legal_person", "bank_card"],
            "question": "请补充营业执照、法人信息和银行卡。",
        }
        ticket_result = {
            "answer": "请补充营业执照、法人信息和银行卡。",
            "confidence": 0.0,
            "finish_reason": "clarification",
            "clarification": {
                "intent_class": "A",
                "workflow_found": True,
                "required_fields": ["business_license", "legal_person", "bank_card"],
            },
        }

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-kb"), patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            return_value=_kb_rag_result(),
        ), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            return_value=decision,
        ), patch(
            "app.api.v1.knowledge.lookup_operation_workflow",
            return_value={
                "workflow_found": True,
                "required_fields": ["business_license", "legal_person", "bank_card"],
                "question": "请补充营业执照、法人信息和银行卡。",
                "workflow_summary": "开通微信子商户号 SOP",
                "workflow_sources": [{"file_name": "开通微信子商户号.docx"}],
            },
        ) as lookup, patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn",
            return_value=ticket_result,
        ) as ticket:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="帮我开通微信子商户号", session_id="session-kb", collection="fushang"),
                user_id="store-1",
            )

        self.assertEqual(response.finish_reason, "clarification")
        self.assertIn("营业执照", response.answer)
        lookup.assert_called_once()
        self.assertEqual(lookup.call_args.kwargs["rag_result"]["sources"][0]["file_name"], "开通微信子商户号.docx")
        ticket.assert_called_once()
        self.assertTrue(ticket.call_args.kwargs["workflow"]["workflow_found"])
        self.assertEqual(ticket.call_args.kwargs["workflow"]["workflow_sources"][0]["file_name"], "开通微信子商户号.docx")

    async def test_c_ambiguous_question_enters_clarification_flow(self):
        decision = {
            "intent_class": "C",
            "needs_ticket_flow": True,
            "fields": {},
            "missing_fields": ["issue_detail"],
            "question": "请补充具体业务场景或报错信息。",
        }
        ticket_result = {
            "answer": "请补充具体业务场景或报错信息。",
            "confidence": 0.0,
            "finish_reason": "clarification",
            "clarification": {"intent_class": "C"},
        }

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-kb"), patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            return_value=_kb_rag_result(),
        ), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            return_value=decision,
        ), patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn",
            return_value=ticket_result,
        ) as ticket:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="这个怎么弄", session_id="session-kb", collection="fushang"),
                user_id="store-1",
            )

        self.assertEqual(response.finish_reason, "clarification")
        self.assertEqual(response.thoughts["clarification"]["intent_class"], "C")
        ticket.assert_called_once()

    async def test_b_missing_knowledge_starts_only_after_rag_miss(self):
        rag_miss = {
            **_kb_rag_result(),
            "request_id": "req-miss",
            "answer": "知识库中没有找到相关内容。",
            "confidence": 0.1,
            "sources": [],
            "used_fallback": True,
            "fallback_reason": "no_relevant_documents",
            "quality_passed": False,
            "quality_level": "low",
        }
        decisions = [
            {"intent_class": "D", "needs_ticket_flow": False, "fields": {}, "missing_fields": []},
            {
                "intent_class": "B",
                "needs_ticket_flow": True,
                "fields": {"issue_detail": "量子积分税率配置"},
                "missing_fields": ["scenario"],
                "question": "请补充业务场景和影响范围。",
            },
        ]
        ticket_result = {
            "answer": "请补充业务场景和影响范围。",
            "confidence": 0.0,
            "finish_reason": "clarification",
            "clarification": {"intent_class": "B"},
        }

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-kb"), patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            return_value=rag_miss,
        ), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            side_effect=decisions,
        ) as classify, patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn",
            return_value=ticket_result,
        ) as ticket:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="量子积分税率怎么配置？", session_id="session-kb", collection="fushang"),
                user_id="store-1",
            )

        self.assertEqual(response.finish_reason, "clarification")
        self.assertEqual(response.thoughts["clarification"]["intent_class"], "B")
        self.assertEqual(classify.call_count, 2)
        ticket.assert_called_once()


if __name__ == "__main__":
    unittest.main()
