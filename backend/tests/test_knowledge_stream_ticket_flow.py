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


def _event_payload(chunk: str) -> dict:
    data_line = next(line for line in chunk.splitlines() if line.startswith("data: "))
    return json.loads(data_line.removeprefix("data: "))


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
    yield "normal streamed answer"


class KnowledgeStreamTicketFlowTests(unittest.IsolatedAsyncioTestCase):
    async def _collect_stream(self, *, convert=False, classifier=None, ticket=None, process_side_effect=None):
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
        classifier = classifier or {
            "intent_class": "B",
            "needs_ticket_flow": True,
            "fields": {"issue_detail": "new gap"},
            "missing_fields": ["impact"],
            "question": "Please clarify impact",
        }
        ticket = ticket or {
            "answer": "Please clarify impact",
            "confidence": 0.0,
            "finish_reason": "clarification",
            "clarification": {"intent_class": "B"},
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
            return_value=classifier,
        ) as classify, patch(
            "app.services.knowledge_service.process_ticket_clarification_turn",
            return_value=ticket,
            side_effect=process_side_effect,
        ) as process_ticket, patch(
            "app.services.knowledge_service._persist_conversation_messages"
        ) as persist:
            chunks = [
                chunk
                async for chunk in stream_knowledge_qa_sse(
                    query="new gap",
                    model_name="doubao-seed-2-0-pro-260215",
                    session_id="session-1",
                    collection="kb",
                    convert_missing_knowledge_to_ticket=convert,
                    user_id="store-1",
                    user_name="Store",
                    channel="h5",
                    sender_id="sender-1",
                    requester_name="Manager",
                )
            ]
        return chunks, classify, process_ticket, persist

    async def test_stream_conversion_disabled_by_default_keeps_normal_done_and_persists(self):
        chunks, classify, process_ticket, persist = await self._collect_stream(convert=False)
        done_payload = _event_payload(chunks[-1])

        self.assertEqual(done_payload["answer"], "No relevant knowledge found")
        self.assertEqual(done_payload["finish_reason"], "stop")
        classify.assert_not_called()
        process_ticket.assert_not_called()
        persist.assert_called_once()

    async def test_stream_conversion_enabled_converts_done_event_and_skips_persist(self):
        chunks, classify, process_ticket, persist = await self._collect_stream(convert=True)
        done_payload = _event_payload(chunks[-1])

        self.assertEqual(done_payload["answer"], "Please clarify impact")
        self.assertEqual(done_payload["finish_reason"], "clarification")
        self.assertTrue(done_payload["thoughts"]["clarification_required"])
        classify.assert_called_once()
        process_ticket.assert_called_once()
        persist.assert_not_called()

    async def test_stream_conversion_error_falls_back_to_normal_done_and_persists(self):
        chunks, _classify, _process_ticket, persist = await self._collect_stream(
            convert=True,
            process_side_effect=RuntimeError("ticket write failed"),
        )
        done_payload = _event_payload(chunks[-1])

        self.assertEqual(done_payload["answer"], "No relevant knowledge found")
        self.assertEqual(done_payload["finish_reason"], "stop")
        persist.assert_called_once()

    async def test_stream_api_enables_missing_knowledge_conversion(self):
        captured = {}

        async def fake_stream(**kwargs):
            captured.update(kwargs)
            yield "event: done\ndata: {}\n\n"

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.stream_knowledge_qa_sse",
            side_effect=fake_stream,
        ):
            response = await knowledge.knowledge_qa_stream(
                KnowledgeRequest(query="new gap", session_id="session-1", collection="kb"),
                user_id="store-1",
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        self.assertEqual(chunks, ["event: done\ndata: {}\n\n"])
        self.assertTrue(captured["convert_missing_knowledge_to_ticket"])
        self.assertEqual(captured["user_id"], "store-1")
        self.assertEqual(captured["channel"], "web")


if __name__ == "__main__":
    unittest.main()
