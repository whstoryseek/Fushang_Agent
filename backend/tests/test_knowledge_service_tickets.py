# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch

from app.services.knowledge_service import _persist_conversation_messages, invoke_knowledge_qa
from app.services.service_ticket_service import start_missing_knowledge_clarification


class KnowledgeServiceTicketTests(unittest.TestCase):
    @patch("app.services.service_ticket_service.record_qa_ticket")
    @patch("app.db.get_conversation_repository")
    def test_persist_conversation_also_records_service_ticket(self, mock_get_conv_repo, mock_record_ticket):
        conv_repo = MagicMock()
        conv_repo.get_session.return_value = {"id": "session-1"}
        mock_get_conv_repo.return_value = conv_repo

        _persist_conversation_messages(
            "session-1",
            "\u5e97\u957f\u95ee\u9898",
            "AI \u7b54\u6848",
            [{"id": "chunk-1", "job_id": "job-1", "content": "\u53ec\u56de\u7247\u6bb5"}],
            0.83,
            None,
            used_fallback=False,
            fallback_reason=None,
            quality_passed=True,
            quality_level="high",
            kb_name="kb",
            user_id="store-1",
            user_name="\u5f20\u5e97\u957f",
            channel="h5",
        )

        mock_record_ticket.assert_called_once()
        kwargs = mock_record_ticket.call_args.kwargs
        self.assertEqual(kwargs["user_id"], "store-1")
        self.assertEqual(kwargs["user_name"], "\u5f20\u5e97\u957f")
        self.assertEqual(kwargs["channel"], "h5")
        self.assertEqual(kwargs["kb_name"], "kb")
        self.assertEqual(kwargs["query"], "\u5e97\u957f\u95ee\u9898")
        self.assertEqual(kwargs["answer"], "AI \u7b54\u6848")
        self.assertEqual(kwargs["sources"][0]["id"], "chunk-1")

    def test_low_confidence_knowledge_answer_is_not_converted_to_clarification(self):
        class FakeAgent:
            async def ainvoke(self, _initial_state, config=None):
                return {
                    "answer": "\u77e5\u8bc6\u5e93\u4e2d\u6ca1\u6709\u627e\u5230\u8db3\u591f\u660e\u786e\u7684\u5904\u7406\u529e\u6cd5\uff0c\u8bf7\u5148\u53c2\u8003\u5df2\u6709\u8d44\u6599\u6392\u67e5\u3002",
                    "confidence": 0.2,
                    "sources": [{"id": "chunk-1"}],
                    "metrics": MagicMock(total_chunks_retrieved=7, chunks_after_rerank=7),
                    "used_fallback": True,
                    "fallback_reason": "no_relevant_knowledge",
                    "quality_passed": False,
                }

        with patch("agents.knowledge.get_knowledge_agent", return_value=FakeAgent()), patch(
            "agents.knowledge.create_initial_state",
            return_value={"query": "\u5fae\u4fe1\u51fa\u73b0\u5c01\u7981\u7684\u60c5\u51b5\u8981\u600e\u4e48\u529e"},
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
                    query="\u5fae\u4fe1\u51fa\u73b0\u5c01\u7981\u7684\u60c5\u51b5\u8981\u600e\u4e48\u529e",
                    model_name="doubao-seed-2-0-pro-260215",
                    session_id="session-1",
                    collection="fushang",
                    user_id="store-1",
                )
            )

        self.assertEqual(result["finish_reason"], "stop")
        self.assertIn("\u77e5\u8bc6\u5e93", result["answer"])
        mock_persist_clarification.assert_not_called()
        self.assertTrue(mock_persist_messages.call_args.kwargs["used_fallback"])

    def test_invoke_knowledge_qa_can_return_fallback_without_persisting_ticket(self):
        class FakeAgent:
            async def ainvoke(self, _initial_state, config=None):
                return {
                    "answer": "\u77e5\u8bc6\u5e93\u672a\u5305\u542b\u8be5\u95ee\u9898\u7684\u7b54\u6848\u3002",
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
            return_value={"query": "\u4e00\u4e2a\u77e5\u8bc6\u5e93\u6ca1\u6709\u7684\u95ee\u9898"},
        ), patch(
            "app.services.knowledge_service._load_kb_retrieval",
            return_value=({"id": "kb-1", "name": "fushang", "kb_type": "standard"}, {}),
        ), patch(
            "app.services.knowledge_service._persist_conversation_messages"
        ) as mock_persist_messages:
            import asyncio

            result = asyncio.run(
                invoke_knowledge_qa(
                    query="\u4e00\u4e2a\u77e5\u8bc6\u5e93\u6ca1\u6709\u7684\u95ee\u9898",
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
        self.assertEqual(result["kb_name"], "fushang")
        mock_persist_messages.assert_not_called()

    def test_start_missing_knowledge_clarification_preserves_rag_audit_payload(self):
        class FakeClarificationRepo:
            def __init__(self):
                self.kwargs = None

            def create_with_contexts(self, **kwargs):
                self.kwargs = kwargs
                return {"id": "ticket-1"}

        repo = FakeClarificationRepo()
        rag_result = {
            "request_id": "request-1",
            "answer": "\u77e5\u8bc6\u5e93\u672a\u5305\u542b\u8be5\u95ee\u9898\u7684\u7b54\u6848\u3002",
            "confidence": 0.2,
            "sources": [{"id": "chunk-1", "content": "context"}],
            "model": "doubao-seed-2-0-pro-260215",
            "thoughts": {"manual_review_recommended": True},
            "used_fallback": True,
            "fallback_reason": "no_relevant_knowledge",
            "quality_passed": False,
            "quality_level": "low",
            "manual_review_recommended": True,
        }

        with patch(
            "app.services.service_ticket_service.get_service_ticket_repository",
            return_value=repo,
        ):
            result = start_missing_knowledge_clarification(
                session_id="session-1",
                user_id="store-1",
                user_name="\u5f20\u5e97\u957f",
                sender_id="sender-1",
                requester_name="\u5f20\u5e97\u957f",
                kb_name="\u5bcc\u53cb",
                query="\u77e5\u8bc6\u5e93\u6ca1\u6709\u7684\u65b0\u95ee\u9898",
                channel="h5",
                rag_result=rag_result,
            )

        self.assertEqual(result["ticket_id"], "ticket-1")
        kb_result = repo.kwargs["clarification"]["kb_result"]
        self.assertEqual(kb_result["request_id"], "request-1")
        self.assertFalse(kb_result["hit"])
        self.assertEqual(kb_result["fallback_reason"], "no_relevant_knowledge")
        self.assertEqual(kb_result["confidence"], 0.2)
        self.assertEqual(kb_result["answer"], rag_result["answer"])
        self.assertEqual(kb_result["sources"], rag_result["sources"])
        self.assertEqual(kb_result["model"], "doubao-seed-2-0-pro-260215")
        self.assertEqual(kb_result["thoughts"], rag_result["thoughts"])
        self.assertFalse(kb_result["quality_passed"])
        self.assertEqual(kb_result["quality_level"], "low")
        self.assertTrue(kb_result["manual_review_recommended"])


if __name__ == "__main__":
    unittest.main()
