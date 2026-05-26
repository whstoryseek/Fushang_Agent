# -*- coding: utf-8 -*-
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.knowledge_service import stream_knowledge_qa_sse


def _event_payloads(events):
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[len("data: "):]))
    return payloads


class StreamTicketFlowTests(unittest.IsolatedAsyncioTestCase):
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

        mock_load_kb.return_value = ({"id": "kb-1", "name": "fushang", "kb_type": "standard"}, {})
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
            convert_missing_knowledge_to_ticket=True,
        ):
            events.append(event)

        payloads = _event_payloads(events)
        done = payloads[-1]
        self.assertEqual(done["finish_reason"], "clarification")
        self.assertEqual(done["answer"], "请补充具体业务场景。")
        self.assertTrue(done["thoughts"]["clarification_required"])
        self.assertEqual(done["thoughts"]["clarification"]["intent_class"], "B")
        mock_start_missing.assert_called_once()


if __name__ == "__main__":
    unittest.main()
