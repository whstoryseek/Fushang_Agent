# -*- coding: utf-8 -*-
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


fake_agents = types.ModuleType("agents")
fake_knowledge = types.ModuleType("agents.knowledge")
fake_nodes = types.ModuleType("agents.knowledge.nodes")
fake_generate = types.ModuleType("agents.knowledge.nodes.generate")
fake_openai_stream = types.ModuleType("agents.knowledge.openai_stream")

class FakeRAGConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


fake_knowledge.RAGConfig = FakeRAGConfig
fake_knowledge.create_initial_state = MagicMock()
fake_knowledge.get_knowledge_stream_prep_agent = MagicMock()
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


async def _empty_deltas(_messages, _model_name):
    if False:
        yield ""


fake_openai_stream.iter_openai_text_deltas = _empty_deltas
fake_agents.knowledge = fake_knowledge
fake_knowledge.nodes = fake_nodes
fake_knowledge.openai_stream = fake_openai_stream
fake_nodes.generate = fake_generate

sys.modules.setdefault("agents", fake_agents)
sys.modules.setdefault("agents.knowledge", fake_knowledge)
sys.modules.setdefault("agents.knowledge.nodes", fake_nodes)
sys.modules.setdefault("agents.knowledge.nodes.generate", fake_generate)
sys.modules.setdefault("agents.knowledge.openai_stream", fake_openai_stream)

from app.services.knowledge_service import stream_knowledge_qa_sse


def _event_payloads(events):
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[len("data: "):]))
    return payloads


class StreamTicketFlowTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.services.knowledge_service._persist_conversation_messages")
    @patch("app.services.knowledge_service.start_missing_knowledge_clarification")
    @patch("app.services.knowledge_service.should_start_missing_knowledge_flow")
    @patch("agents.knowledge.openai_stream.iter_openai_text_deltas")
    @patch("agents.knowledge.get_knowledge_stream_prep_agent")
    @patch("agents.knowledge.create_initial_state")
    @patch("app.services.knowledge_service._load_kb_retrieval")
    async def test_stream_converts_fallback_done_to_missing_knowledge_clarification(
        self,
        mock_load_kb,
        mock_create_state,
        mock_get_agent,
        mock_deltas,
        mock_should_start,
        mock_start_missing,
        mock_persist,
    ):
        class FakeSnapshot:
            def __init__(self, values):
                self.values = values

        class FakeAgent:
            def __init__(self):
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
                            "query": "知识库没有的新问题",
                        }
                    )
                return FakeSnapshot(
                    {
                        "answer": "知识库未包含该问题的答案。",
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

        async def fake_deltas(messages, model_name):
            yield "知识库未包含该问题的答案。"

        mock_load_kb.return_value = (
            {"id": "kb-1", "name": "富商知识库", "kb_type": "standard"},
            {},
        )
        mock_create_state.return_value = {"query": "知识库没有的新问题"}
        mock_get_agent.return_value = FakeAgent()
        mock_deltas.side_effect = fake_deltas
        mock_should_start.return_value = True
        mock_start_missing.return_value = {
            "ticket_id": "ticket-1",
            "answer": "请补充具体业务场景。",
            "finish_reason": "clarification",
            "clarification": {"intent_class": "B", "reason": "knowledge_missing"},
        }

        events = []
        async for event in stream_knowledge_qa_sse(
            query="知识库没有的新问题",
            model_name="doubao-seed-2-0-pro-260215",
            session_id="session-1",
            collection="fushang",
            user_id="store-1",
            user_name="张店长",
            channel="h5",
            sender_id="sender-1",
            requester_name="王经理",
            query_image_oss_key="oss://query-image.jpg",
            convert_missing_knowledge_to_ticket=True,
        ):
            events.append(event)

        payloads = _event_payloads(events)
        done = payloads[-1]
        self.assertEqual(done["finish_reason"], "clarification")
        self.assertEqual(done["answer"], "请补充具体业务场景。")
        self.assertTrue(done["thoughts"]["clarification_required"])
        self.assertTrue(done["thoughts"]["missing_knowledge_started"])
        self.assertEqual(done["thoughts"]["clarification"]["intent_class"], "B")
        mock_persist.assert_not_called()
        mock_start_missing.assert_called_once()

        missing_kwargs = mock_start_missing.call_args.kwargs
        self.assertEqual(missing_kwargs["session_id"], "session-1")
        self.assertEqual(missing_kwargs["user_id"], "store-1")
        self.assertEqual(missing_kwargs["user_name"], "张店长")
        self.assertEqual(missing_kwargs["channel"], "h5")
        self.assertEqual(missing_kwargs["sender_id"], "sender-1")
        self.assertEqual(missing_kwargs["requester_name"], "王经理")
        self.assertEqual(missing_kwargs["query"], "知识库没有的新问题")
        self.assertTrue(missing_kwargs["has_image"])
        self.assertEqual(missing_kwargs["query_image_oss_key"], "oss://query-image.jpg")
        self.assertEqual(missing_kwargs["kb_name"], "富商知识库")

        rag_result = missing_kwargs["rag_result"]
        self.assertTrue(rag_result["used_fallback"])
        self.assertEqual(rag_result["fallback_reason"], "no_relevant_knowledge")
        self.assertFalse(rag_result["quality_passed"])
        self.assertEqual(rag_result["quality_level"], "low")
        self.assertEqual(rag_result["sources"], [])


if __name__ == "__main__":
    unittest.main()
