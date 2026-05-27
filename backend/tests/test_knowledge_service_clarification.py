# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch

from app.services.knowledge_service import persist_clarification_message


class KnowledgeServiceClarificationTests(unittest.TestCase):
    @patch("app.services.service_ticket_service.process_clarification_turn")
    @patch("app.db.get_conversation_repository")
    def test_persist_clarification_records_conversation_and_ticket(self, mock_get_conv_repo, mock_process):
        conv_repo = MagicMock()
        conv_repo.get_session.return_value = {"id": "session-1"}
        mock_get_conv_repo.return_value = conv_repo
        mock_process.return_value = {
            "ticket_id": "ticket-1",
            "status": "clarifying",
            "answer": "请补充一下具体对象或场景",
            "finish_reason": "clarification",
        }

        result = persist_clarification_message(
            session_id="session-1",
            query="怎么处理",
            answer="请补充一下具体对象或场景",
            kb_name="kb",
            user_id="store-1",
            user_name="张店长",
            channel="h5",
            sender_id="sender-1",
        )

        self.assertEqual(result["ticket_id"], "ticket-1")
        self.assertEqual(conv_repo.add_message.call_count, 2)
        mock_process.assert_called_once()
        self.assertEqual(mock_process.call_args.kwargs["user_id"], "store-1")
        self.assertEqual(mock_process.call_args.kwargs["sender_id"], "sender-1")


if __name__ == "__main__":
    unittest.main()
