# -*- coding: utf-8 -*-
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from app.services.knowledge_service import stream_knowledge_qa_sse


def _event_payloads(events):
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[len("data: "):]))
    return payloads


class FakeRAGConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeSnapshot:
    def __init__(self, values):
        self.values = values


class FakeAgent:
    def __init__(self, *, answer="Knowledge base has no answer."):
        self.answer = answer
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
                    "query": "missing query",
                }
            )
        return FakeSnapshot(
            {
                "answer": self.answer,
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


def _fake_agent_modules(*, agent, delta_text="Knowledge base has no answer."):
    fake_agents = types.ModuleType("agents")
    fake_knowledge = types.ModuleType("agents.knowledge")
    fake_nodes = types.ModuleType("agents.knowledge.nodes")
    fake_generate = types.ModuleType("agents.knowledge.nodes.generate")
    fake_openai_stream = types.ModuleType("agents.knowledge.openai_stream")

    async def fake_deltas(_messages, _model_name):
        yield delta_text

    fake_knowledge.RAGConfig = FakeRAGConfig
    fake_knowledge.create_initial_state = MagicMock(return_value={"query": "missing query"})
    fake_knowledge.get_knowledge_stream_prep_agent = MagicMock(return_value=agent)
    fake_generate.build_sources_from_reranked = MagicMock(return_value=[])
    fake_generate._sanitize_image_placeholders = lambda text, _image_map: text
    fake_generate.prepare_generation_context = MagicMock(
        return_value={
            "messages": [],
            "model_name": "doubao-seed-2-0-pro-260215",
            "reranked_chunks": [],
            "image_map": {},
            "is_image_mode": False,
            "is_multimodal_kb": False,
        }
    )
    fake_openai_stream.iter_openai_text_deltas = fake_deltas
    fake_agents.knowledge = fake_knowledge
    fake_knowledge.nodes = fake_nodes
    fake_knowledge.openai_stream = fake_openai_stream
    fake_nodes.generate = fake_generate

    return {
        "agents": fake_agents,
        "agents.knowledge": fake_knowledge,
        "agents.knowledge.nodes": fake_nodes,
        "agents.knowledge.nodes.generate": fake_generate,
        "agents.knowledge.openai_stream": fake_openai_stream,
    }


async def _collect_stream_events(**kwargs):
    events = []
    async for event in stream_knowledge_qa_sse(
        query="missing query",
        model_name="doubao-seed-2-0-pro-260215",
        session_id="session-1",
        collection="fushang",
        user_id="store-1",
        user_name="Store Manager",
        channel="h5",
        sender_id="sender-1",
        requester_name="Requester",
        query_image_oss_key="oss://query-image.jpg",
        convert_missing_knowledge_to_ticket=True,
        **kwargs,
    ):
        events.append(event)
    return events


class StreamTicketFlowTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.services.knowledge_service._persist_conversation_messages")
    @patch("app.services.knowledge_service.start_missing_knowledge_clarification")
    @patch("app.services.knowledge_service.should_start_missing_knowledge_flow")
    @patch("app.services.knowledge_service._load_kb_retrieval")
    async def test_stream_converts_fallback_done_to_missing_knowledge_clarification(
        self,
        mock_load_kb,
        mock_should_start,
        mock_start_missing,
        mock_persist,
    ):
        mock_load_kb.return_value = (
            {"id": "kb-1", "name": "Fushang KB", "kb_type": "standard"},
            {},
        )
        mock_should_start.return_value = True
        mock_start_missing.return_value = {
            "ticket_id": "ticket-1",
            "answer": "Please add business context.",
            "finish_reason": "clarification",
            "clarification": {"intent_class": "B", "reason": "knowledge_missing"},
        }

        with patch.dict(sys.modules, _fake_agent_modules(agent=FakeAgent()), clear=False):
            events = await _collect_stream_events()

        payloads = _event_payloads(events)
        done = payloads[-1]
        self.assertEqual(done["finish_reason"], "clarification")
        self.assertEqual(done["answer"], "Please add business context.")
        self.assertTrue(done["thoughts"]["clarification_required"])
        self.assertTrue(done["thoughts"]["missing_knowledge_started"])
        self.assertEqual(done["thoughts"]["clarification"]["intent_class"], "B")
        mock_persist.assert_not_called()
        mock_start_missing.assert_called_once()

        missing_kwargs = mock_start_missing.call_args.kwargs
        self.assertEqual(missing_kwargs["session_id"], "session-1")
        self.assertEqual(missing_kwargs["user_id"], "store-1")
        self.assertEqual(missing_kwargs["user_name"], "Store Manager")
        self.assertEqual(missing_kwargs["channel"], "h5")
        self.assertEqual(missing_kwargs["sender_id"], "sender-1")
        self.assertEqual(missing_kwargs["requester_name"], "Requester")
        self.assertEqual(missing_kwargs["query"], "missing query")
        self.assertTrue(missing_kwargs["has_image"])
        self.assertEqual(missing_kwargs["query_image_oss_key"], "oss://query-image.jpg")
        self.assertEqual(missing_kwargs["kb_name"], "Fushang KB")

        rag_result = missing_kwargs["rag_result"]
        self.assertTrue(rag_result["used_fallback"])
        self.assertEqual(rag_result["fallback_reason"], "no_relevant_knowledge")
        self.assertFalse(rag_result["quality_passed"])
        self.assertEqual(rag_result["quality_level"], "low")
        self.assertEqual(rag_result["sources"], [])

    @patch("app.services.knowledge_service._persist_conversation_messages")
    @patch("app.services.knowledge_service.start_missing_knowledge_clarification")
    @patch("app.services.knowledge_service.should_start_missing_knowledge_flow")
    @patch("app.services.knowledge_service._load_kb_retrieval")
    async def test_stream_falls_back_to_normal_done_when_ticket_conversion_fails(
        self,
        mock_load_kb,
        mock_should_start,
        mock_start_missing,
        mock_persist,
    ):
        mock_load_kb.return_value = (
            {"id": "kb-1", "name": "Fushang KB", "kb_type": "standard"},
            {},
        )
        mock_should_start.return_value = True
        mock_start_missing.side_effect = RuntimeError("ticket service unavailable")

        with patch.dict(sys.modules, _fake_agent_modules(agent=FakeAgent()), clear=False):
            events = await _collect_stream_events()

        payloads = _event_payloads(events)
        done = payloads[-1]
        self.assertEqual(done["finish_reason"], "stop")
        self.assertEqual(done["answer"], "Knowledge base has no answer.")
        self.assertFalse(done["thoughts"]["clarification_required"])
        self.assertNotIn("missing_knowledge_started", done["thoughts"])
        mock_start_missing.assert_called_once()
        mock_persist.assert_called_once()


if __name__ == "__main__":
    unittest.main()
