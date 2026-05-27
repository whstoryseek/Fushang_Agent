import os
import sys
import types
import unittest
from unittest.mock import patch

os.environ.setdefault("PG_HOST", "localhost")
os.environ.setdefault("PG_USER", "tester")
os.environ.setdefault("PG_PASSWORD", "tester")
os.environ.setdefault("MILVUS_HOST", "localhost")
os.environ.setdefault("VOLCES_API_KEY", "test-key")

if "psycopg2" not in sys.modules:
    psycopg2_module = types.ModuleType("psycopg2")
    psycopg2_module.pool = types.SimpleNamespace(ThreadedConnectionPool=object)
    psycopg2_module.extras = types.SimpleNamespace(RealDictCursor=object)
    sys.modules["psycopg2"] = psycopg2_module
    sys.modules["psycopg2.pool"] = psycopg2_module.pool
    sys.modules["psycopg2.extras"] = psycopg2_module.extras

from app.core.exceptions import ForbiddenError
from app.services import conversation_service


class ConversationHistoryTests(unittest.TestCase):
    def test_list_sessions_supports_optional_kb_filter(self):
        with patch("app.services.conversation_service.get_conversation_repository") as mock_get_repo:
            repo = mock_get_repo.return_value
            repo.list_sessions.return_value = [{"id": "s1"}]

            result = conversation_service.list_sessions(kb_name=None, user_id="tester")

        self.assertEqual(result["total"], 1)
        repo.list_sessions.assert_called_once_with(kb_name=None, user_id="tester")

    def test_build_session_title_trims_whitespace_and_length(self):
        title = conversation_service.build_session_title("   这是一个很长的历史对话标题，需要被裁剪一下   ", max_length=10)
        self.assertEqual(title, "这是一个很长的历史对...")

    @patch("app.services.conversation_service.create_session")
    def test_ensure_knowledge_session_creates_session_for_default_placeholder(self, mock_create_session):
        mock_create_session.return_value = {"id": "session-123"}

        session_id = conversation_service.ensure_knowledge_session(
            collection="kb-demo",
            session_id="default",
            query="请解释一下报销流程的适用范围",
        )

        self.assertEqual(session_id, "session-123")
        mock_create_session.assert_called_once()
        self.assertEqual(mock_create_session.call_args.kwargs["kb_name"], "kb-demo")
        self.assertIn("请解释一下报销流程的适用范围", mock_create_session.call_args.kwargs["title"])

    @patch("app.services.conversation_service.create_session")
    def test_ensure_knowledge_session_keeps_existing_session(self, mock_create_session):
        with patch("app.services.conversation_service.get_conversation_repository") as mock_get_repo:
            repo = mock_get_repo.return_value
            repo.get_session.return_value = {"id": "session-existing", "user_id": "default"}

            session_id = conversation_service.ensure_knowledge_session(
                collection="kb-demo",
                session_id="session-existing",
                query="继续追问",
            )

        self.assertEqual(session_id, "session-existing")
        mock_create_session.assert_not_called()

    @patch("app.services.conversation_service.create_session")
    def test_ensure_knowledge_session_creates_session_when_lookup_rejects_id(self, mock_create_session):
        mock_create_session.return_value = {"id": "session-new"}
        with patch("app.services.conversation_service.get_conversation_repository") as mock_get_repo:
            repo = mock_get_repo.return_value
            repo.get_session.side_effect = Exception("invalid uuid")

            session_id = conversation_service.ensure_knowledge_session(
                collection="kb-demo",
                session_id="daily-session-title",
                query="继续追问",
                user_id="guest_local",
            )

        self.assertEqual(session_id, "session-new")
        mock_create_session.assert_called_once()

    def test_ensure_knowledge_session_rejects_foreign_session(self):
        with patch("app.services.conversation_service.get_conversation_repository") as mock_get_repo:
            repo = mock_get_repo.return_value
            repo.get_session.return_value = {"id": "session-existing", "user_id": "other-guest"}

            with self.assertRaises(ForbiddenError):
                conversation_service.ensure_knowledge_session(
                    collection="kb-demo",
                    session_id="session-existing",
                    query="继续追问",
                    user_id="guest_local",
                )

    def test_get_session_messages_rejects_foreign_session(self):
        with patch("app.services.conversation_service.get_conversation_repository") as mock_get_repo:
            repo = mock_get_repo.return_value
            repo.get_session.return_value = {"id": "session-existing", "user_id": "other-guest"}

            with self.assertRaises(ForbiddenError):
                conversation_service.get_session_messages(
                    "session-existing",
                    user_id="guest_local",
                )


if __name__ == "__main__":
    unittest.main()
