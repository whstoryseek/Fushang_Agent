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
from app.db import init_db


class ServiceTicketRepositoryTests(unittest.TestCase):
    def test_init_db_includes_service_ticket_tables_and_indexes(self):
        schema = "\n".join(init_db._TABLES)

        self.assertIn("CREATE TABLE IF NOT EXISTS service_ticket", schema)
        self.assertIn("CREATE TABLE IF NOT EXISTS service_ticket_context", schema)
        self.assertIn("idx_service_ticket_status", schema)
        self.assertIn("idx_service_ticket_context_ticket", schema)
        self.assertIn("clarification JSONB NOT NULL DEFAULT '{}'", schema)
        self.assertIn("entry_user_id", schema)
        self.assertIn("entry_user_name", schema)
        self.assertIn("entry_source", schema)
        self.assertIn("idx_service_ticket_active_clarifying_unique", schema)
        self.assertIn("COALESCE(session_id, '')", schema)
        self.assertIn("COALESCE(kb_name, '')", schema)
        self.assertIn("WHERE status = 'clarifying'", schema)

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
                    "entry_user_id": "wangqizhi_l8el",
                    "entry_user_name": "农资店王麒麟",
                    "entry_source": "renruikeji_sso",
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
                entry_user_id="wangqizhi_l8el",
                entry_user_name="农资店王麒麟",
                entry_source="renruikeji_sso",
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
        self.assertEqual(ticket["entry_user_id"], "wangqizhi_l8el")
        self.assertEqual(ticket["entry_user_name"], "农资店王麒麟")
        self.assertEqual(ticket["entry_source"], "renruikeji_sso")
        returning.assert_called_once()
        many.assert_called_once()
        self.assertEqual(many.call_args.args[1][0][0], "ticket-1")
        self.assertEqual(many.call_args.args[1][0][1], "chunk-1")
        self.assertIn("entry_user_id", returning.call_args.args[0])

    def test_find_active_clarification_supports_row_locking(self):
        repo = ServiceTicketRepository()

        with patch.object(repo, "_execute_select", return_value=[]) as select:
            ticket = repo.find_active_clarification(
                session_id="session-1",
                user_id="store-1",
                kb_name="kb",
                for_update=True,
            )

        self.assertIsNone(ticket)
        self.assertIn("FOR UPDATE", select.call_args.args[0])

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
                    "entry_user_id": "wangqizhi_l8el",
                    "entry_user_name": "农资店王麒麟",
                    "entry_source": "renruikeji_sso",
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

    def test_update_clarification_unanswered_normal_marks_resolved_at(self):
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
                    "query": "新活动政策刷新后没有显示是什么原因？",
                    "answer": "当前知识库暂无相关信息",
                    "status": "unanswered_normal",
                    "confidence": 0.2,
                    "fallback_reason": "no_relevant_documents",
                    "quality_level": "low",
                    "sources": "[]",
                    "channel": "web",
                    "processing_ms": None,
                    "sender_id": None,
                    "requester_name": "店长",
                    "entry_user_id": "wangqizhi_l8el",
                    "entry_user_name": "农资店王麒麟",
                    "entry_source": "renruikeji_sso",
                    "clarification_round": 2,
                    "clarification": '{"intent_class":"D"}',
                    "note": None,
                    "created_at": "2026-05-26 10:00:00+08",
                    "updated_at": "2026-05-26 10:02:00+08",
                    "resolved_at": "2026-05-26 10:02:00+08",
                }
            ],
        ) as returning:
            ticket = repo.update_clarification(
                "ticket-1",
                status="unanswered_normal",
                answer="当前知识库暂无相关信息",
                clarification_round=2,
                clarification={"intent_class": "D"},
            )

        self.assertEqual(ticket["status"], "unanswered_normal")
        self.assertIn("resolved_at = COALESCE(resolved_at, NOW())", returning.call_args.args[0])

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

    def test_delete_removes_ticket_by_id_and_reports_success(self):
        repo = ServiceTicketRepository()

        with patch.object(
            repo,
            "_execute_returning",
            return_value=[{"id": "ticket-1"}],
        ) as returning:
            deleted = repo.delete("ticket-1")

        self.assertTrue(deleted)
        self.assertIn("DELETE FROM service_ticket", returning.call_args.args[0])
        self.assertEqual(returning.call_args.args[1], ("ticket-1",))

    def test_delete_returns_false_when_ticket_missing(self):
        repo = ServiceTicketRepository()

        with patch.object(repo, "_execute_returning", return_value=[]):
            deleted = repo.delete("missing-ticket")

        self.assertFalse(deleted)


if __name__ == "__main__":
    unittest.main()
