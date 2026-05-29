# -*- coding: utf-8 -*-
import json
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
from app.services.knowledge_service import stream_knowledge_qa_sse


def _rag_result(**overrides):
    payload = {
        "request_id": "req-1",
        "session_id": "session-1",
        "answer": "No relevant knowledge found",
        "confidence": 0.2,
        "sources": [],
        "model": "doubao-seed-2-0-pro-260215",
        "thoughts": {},
        "image_map": {},
        "finish_reason": "stop",
        "used_fallback": True,
        "fallback_reason": "no_relevant_documents",
        "quality_passed": False,
        "quality_level": "low",
        "kb_name": "kb",
    }
    payload.update(overrides)
    return payload


def _done_payload(chunk: str) -> dict:
    line = next(item for item in chunk.splitlines() if item.startswith("data: "))
    return json.loads(line.removeprefix("data: "))


class FakeSnapshot:
    def __init__(self, values):
        self.values = values


class FakeStreamAgent:
    def __init__(self, final_values):
        self.final_values = final_values
        self.state_reads = 0

    async def ainvoke(self, initial_state, config=None):
        return {}

    async def aupdate_state(self, config, values):
        return None

    async def aget_state(self, config):
        self.state_reads += 1
        if self.state_reads == 1:
            return FakeSnapshot(
                {
                    "query_intent": "qa",
                    "query_complexity": "simple",
                    "query_keywords": [],
                    "metrics": types.SimpleNamespace(total_chunks_retrieved=1, chunks_after_rerank=1),
                }
            )
        return FakeSnapshot(self.final_values)


async def _one_delta(_messages, _model_name):
    yield "No relevant knowledge found"


BACKGROUND_DECISION = {
    "intent_class": "D",
    "needs_ticket_flow": False,
    "fields": {},
    "missing_fields": [],
    "question": "",
}


class MissingKnowledgeTicketConversionTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_route_keeps_normal_stop_and_records_unanswered_normal_when_classifier_allows_rag(self):
        decisions = [dict(BACKGROUND_DECISION), dict(BACKGROUND_DECISION)]

        async def fake_rag(**kwargs):
            return _rag_result()

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            side_effect=fake_rag,
        ), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            side_effect=decisions,
        ) as classify, patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn"
        ) as process_ticket, patch(
            "app.api.v1.knowledge.persist_knowledge_result"
        ) as persist, patch(
            "app.api.v1.knowledge.record_unanswered_normal_ticket"
        ) as record_unanswered, patch(
            "app.api.v1.knowledge.record_ai_resolved_ticket"
        ) as record_resolved:
            response = await knowledge.knowledge_qa(
                KnowledgeRequest(query="new gap", session_id="session-1", collection="kb"),
                user_id="store-1",
            )

        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.answer, "No relevant knowledge found")
        self.assertEqual(classify.call_count, 2)
        process_ticket.assert_not_called()
        persist.assert_called_once()
        record_unanswered.assert_called_once()
        record_resolved.assert_not_called()

    async def test_stream_keeps_normal_stop_and_records_unanswered_normal_when_classifier_allows_rag(self):
        final_values = {
            "answer": "No relevant knowledge found",
            "confidence": 0.2,
            "sources": [],
            "query_intent": "qa",
            "query_complexity": "simple",
            "query_keywords": [],
            "metrics": types.SimpleNamespace(total_chunks_retrieved=1, chunks_after_rerank=1),
            "used_fallback": True,
            "fallback_reason": "no_relevant_documents",
            "quality_passed": False,
            "answer_quality": "low",
            "image_map": {},
        }
        ctx = {
            "reranked_chunks": [],
            "messages": [],
            "model_name": "doubao-seed-2-0-pro-260215",
            "image_map": {},
            "is_image_mode": False,
            "is_multimodal_kb": False,
        }

        with patch("agents.knowledge.get_knowledge_stream_prep_agent", return_value=FakeStreamAgent(final_values)), patch(
            "agents.knowledge.create_initial_state",
            return_value={"query": "new gap"},
        ), patch(
            "agents.knowledge.nodes.generate.prepare_generation_context",
            return_value=ctx,
        ), patch(
            "agents.knowledge.nodes.generate.build_sources_from_reranked",
            return_value=[],
        ), patch(
            "agents.knowledge.openai_stream.iter_openai_text_deltas",
            side_effect=lambda messages, model_name: _one_delta(messages, model_name),
        ), patch(
            "app.services.knowledge_service.classify_ticket_intent_with_llm",
            return_value=dict(BACKGROUND_DECISION),
        ) as classify, patch(
            "app.services.knowledge_service.process_ticket_clarification_turn"
        ) as process_ticket, patch(
            "app.services.knowledge_service._persist_conversation_messages"
        ) as persist, patch(
            "app.services.knowledge_service.record_unanswered_normal_ticket"
        ) as record_unanswered, patch(
            "app.services.knowledge_service.record_ai_resolved_ticket"
        ) as record_resolved:
            chunks = [
                chunk
                async for chunk in stream_knowledge_qa_sse(
                    query="new gap",
                    model_name="doubao-seed-2-0-pro-260215",
                    session_id="session-1",
                    collection="kb",
                    convert_missing_knowledge_to_ticket=True,
                    user_id="store-1",
                    user_name="Store",
                    channel="h5",
                    sender_id="sender-1",
                    requester_name="Manager",
                )
            ]

        done_payload = _done_payload(chunks[-1])
        self.assertEqual(done_payload["finish_reason"], "stop")
        self.assertEqual(done_payload["answer"], "No relevant knowledge found")
        classify.assert_called_once()
        process_ticket.assert_not_called()
        persist.assert_called_once()
        record_unanswered.assert_called_once()
        record_resolved.assert_not_called()


if __name__ == "__main__":
    unittest.main()
