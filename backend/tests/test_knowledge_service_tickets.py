# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch

from app.services.knowledge_service import _persist_conversation_messages, invoke_knowledge_qa


class KnowledgeServiceTicketTests(unittest.TestCase):
    @patch("app.services.service_ticket_service.record_qa_ticket")
    @patch("app.db.get_conversation_repository")
    def test_persist_conversation_also_records_service_ticket(self, mock_get_conv_repo, mock_record_ticket):
        conv_repo = MagicMock()
        conv_repo.get_session.return_value = {"id": "session-1"}
        mock_get_conv_repo.return_value = conv_repo

        _persist_conversation_messages(
            "session-1",
            "店长问题",
            "AI 答案",
            [{"id": "chunk-1", "job_id": "job-1", "content": "召回片段"}],
            0.83,
            None,
            used_fallback=False,
            fallback_reason=None,
            quality_passed=True,
            quality_level="high",
            kb_name="kb",
            user_id="store-1",
            user_name="张店长",
            channel="h5",
        )

        mock_record_ticket.assert_called_once()
        kwargs = mock_record_ticket.call_args.kwargs
        self.assertEqual(kwargs["user_id"], "store-1")
        self.assertEqual(kwargs["user_name"], "张店长")
        self.assertEqual(kwargs["channel"], "h5")
        self.assertEqual(kwargs["kb_name"], "kb")
        self.assertEqual(kwargs["query"], "店长问题")
        self.assertEqual(kwargs["answer"], "AI 答案")
        self.assertEqual(kwargs["sources"][0]["id"], "chunk-1")


    def test_low_confidence_knowledge_answer_is_not_converted_to_clarification(self):
        class FakeAgent:
            async def ainvoke(self, _initial_state, config=None):
                return {
                    "answer": "知识库中没有找到足够明确的处理办法，请先参考已有资料排查。",
                    "confidence": 0.2,
                    "sources": [{"id": "chunk-1"}],
                    "metrics": MagicMock(total_chunks_retrieved=7, chunks_after_rerank=7),
                    "used_fallback": True,
                    "fallback_reason": "no_relevant_knowledge",
                    "quality_passed": False,
                }

        with patch("agents.knowledge.get_knowledge_agent", return_value=FakeAgent()), patch(
            "agents.knowledge.create_initial_state",
            return_value={"query": "微信出现封禁的情况要怎么办"},
        ), patch(
            "app.services.knowledge_service._load_kb_retrieval",
            return_value=({"id": "kb-1", "name": "fushang", "kb_type": "standard"}, {}),
        ), patch(
            "app.services.knowledge_service.persist_clarification_message"
        ) as mock_persist_clarification, patch(
            "app.services.knowledge_service._persist_conversation_messages"
        ) as mock_persist_messages:
            import asyncio

            result = asyncio.run(
                invoke_knowledge_qa(
                    query="微信出现封禁的情况要怎么办",
                    model_name="doubao-seed-2-0-pro-260215",
                    session_id="session-1",
                    collection="fushang",
                    user_id="store-1",
                )
            )

        self.assertEqual(result["finish_reason"], "stop")
        self.assertIn("知识库", result["answer"])
        mock_persist_clarification.assert_not_called()
        self.assertTrue(mock_persist_messages.call_args.kwargs["used_fallback"])

    def test_invoke_knowledge_qa_can_return_fallback_without_persisting_ticket(self):
        class FakeAgent:
            async def ainvoke(self, _initial_state, config=None):
                return {
                    "answer": "知识库未包含该问题的答案。",
                    "confidence": 0.2,
                    "sources": [],
                    "metrics": MagicMock(total_chunks_retrieved=0, chunks_after_rerank=0),
                    "used_fallback": True,
                    "fallback_reason": "no_relevant_knowledge",
                    "quality_passed": False,
                    "answer_quality": "low",
                }

        with patch("agents.knowledge.get_knowledge_agent", return_value=FakeAgent()), patch(
            "agents.knowledge.create_initial_state",
            return_value={"query": "一个知识库没有的问题"},
        ), patch(
            "app.services.knowledge_service._load_kb_retrieval",
            return_value=({"id": "kb-1", "name": "fushang", "kb_type": "standard"}, {}),
        ), patch(
            "app.services.knowledge_service._persist_conversation_messages"
        ) as mock_persist_messages:
            import asyncio

            result = asyncio.run(
                invoke_knowledge_qa(
                    query="一个知识库没有的问题",
                    model_name="doubao-seed-2-0-pro-260215",
                    session_id="session-1",
                    collection="fushang",
                    user_id="store-1",
                    persist=False,
                )
            )

        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["fallback_reason"], "no_relevant_knowledge")
        self.assertFalse(result["quality_passed"])
        self.assertEqual(result["quality_level"], "low")
        mock_persist_messages.assert_not_called()


if __name__ == "__main__":
    unittest.main()
