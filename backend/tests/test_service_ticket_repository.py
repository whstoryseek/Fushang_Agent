# -*- coding: utf-8 -*-
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

from app.db.service_ticket_repository import ServiceTicketRepository


class ServiceTicketRepositoryTests(unittest.TestCase):
    def test_create_with_contexts_persists_ticket_and_context_rows(self):
        repo = ServiceTicketRepository()

        with patch.object(
            repo,
            "_execute_returning",
            return_value=[
                {
                    "id": "ticket-1",
                    "session_id": None,
                    "user_id": "store-1",
                    "user_name": "店长",
                    "kb_name": "kb",
                    "query": "问题",
                    "answer": "答案",
                    "status": "resolved_ai",
                    "confidence": 0.9,
                    "fallback_reason": None,
                    "quality_level": "high",
                    "sources": "[]",
                    "channel": "web",
                    "processing_ms": 123,
                    "sender_id": None,
                    "requester_name": "店长",
                    "clarification_round": 0,
                    "clarification": "{}",
                    "created_at": "2026-05-26 10:00:00+08",
                    "updated_at": "2026-05-26 10:00:00+08",
                    "resolved_at": None,
                }
            ],
        ) as returning, patch.object(repo, "_execute_many") as many:
            ticket = repo.create_with_contexts(
                user_id="store-1",
                user_name="店长",
                kb_name="kb",
                query="问题",
                answer="答案",
                status="resolved_ai",
                confidence=0.9,
                fallback_reason=None,
                quality_level="high",
                sources=[],
                channel="web",
                processing_ms=123,
                contexts=[
                    {
                        "chunk_id": "chunk-1",
                        "job_id": "job-1",
                        "file_name": "policy.docx",
                        "chunk_index": 1,
                        "score": 0.87,
                        "content": "召回内容",
                        "metadata": {"page": 2},
                        "sort_order": 0,
                    }
                ],
            )

        self.assertEqual(ticket["id"], "ticket-1")
        self.assertEqual(ticket["status"], "resolved_ai")
        returning.assert_called_once()
        many.assert_called_once()
        self.assertEqual(many.call_args.args[1][0][0], "ticket-1")
        self.assertEqual(many.call_args.args[1][0][1], "chunk-1")

    def test_update_clarification_updates_status_and_collected_payload(self):
        repo = ServiceTicketRepository()

        with patch.object(
            repo,
            "_execute_returning",
            return_value=[
                {
                    "id": "ticket-1",
                    "session_id": "session-1",
                    "user_id": "store-1",
                    "user_name": "店长",
                    "kb_name": "kb",
                    "query": "开通直播权限",
                    "answer": "已转后台人工工单",
                    "status": "pending_manual",
                    "confidence": 0.0,
                    "fallback_reason": "operation_required",
                    "quality_level": "clarifying",
                    "sources": "[]",
                    "channel": "h5",
                    "processing_ms": None,
                    "sender_id": "sender-1",
                    "requester_name": "店长",
                    "clarification_round": 3,
                    "clarification": '{"collected":{"phone":"13800138000"}}',
                    "note": None,
                    "created_at": "2026-05-26 10:00:00+08",
                    "updated_at": "2026-05-26 10:02:00+08",
                    "resolved_at": None,
                }
            ],
        ) as returning:
            ticket = repo.update_clarification(
                "ticket-1",
                status="pending_manual",
                answer="已转后台人工工单",
                clarification_round=3,
                clarification={"collected": {"phone": "13800138000"}},
                sender_id="sender-1",
                requester_name="店长",
            )

        self.assertEqual(ticket["status"], "pending_manual")
        self.assertEqual(ticket["clarification_round"], 3)
        self.assertEqual(ticket["clarification"]["collected"]["phone"], "13800138000")
        self.assertIn("clarification = %s", returning.call_args.args[0])

    def test_stats_counts_statuses_and_daily_rows(self):
        repo = ServiceTicketRepository()

        with patch.object(
            repo,
            "_execute_select",
            side_effect=[
                [{"total": 3}],
                [
                    {"status": "resolved_ai", "count": 2},
                    {"status": "pending_manual", "count": 1},
                ],
                [{"date": "2026-05-26", "count": 3, "pending_manual": 1}],
            ],
        ):
            stats = repo.stats()

        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["by_status"]["resolved_ai"], 2)
        self.assertEqual(stats["by_status"]["pending_manual"], 1)
        self.assertEqual(stats["daily"][0]["pending_manual"], 1)


if __name__ == "__main__":
    unittest.main()
