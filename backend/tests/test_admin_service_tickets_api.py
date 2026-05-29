# -*- coding: utf-8 -*-
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

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

from app.api.v1.admin import service_tickets


class AdminServiceTicketsApiTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.api.v1.admin.service_tickets.service_ticket_service.list_tickets")
    async def test_list_service_tickets_returns_success_payload(self, mock_list):
        mock_list.return_value = {"total": 1, "items": [{"id": "ticket-1"}]}

        response = await service_tickets.list_service_tickets(status="pending_manual")

        self.assertEqual(response.body.decode("utf-8"), '{"success":true,"data":{"total":1,"items":[{"id":"ticket-1"}]}}')
        self.assertEqual(mock_list.call_args.kwargs["status"], "pending_manual")

    @patch("app.api.v1.admin.service_tickets.service_ticket_service.update_original_chunk_from_ticket")
    async def test_update_ticket_chunk_delegates_to_service(self, mock_update):
        mock_update.return_value = {"ticket_id": "ticket-1", "chunk_id": "chunk-1", "job_id": "job-1"}

        response = await service_tickets.update_ticket_chunk(
            "ticket-1",
            "chunk-1",
            service_tickets.UpdateTicketChunkRequest(content="新的切片内容"),
        )

        self.assertIn('"chunk_id":"chunk-1"', response.body.decode("utf-8"))
        mock_update.assert_called_once_with("ticket-1", "chunk-1", "新的切片内容")

    @patch("app.api.v1.admin.service_tickets.service_ticket_service.revectorize_ticket_context", new_callable=AsyncMock)
    async def test_revectorize_ticket_delegates_to_service(self, mock_revectorize):
        mock_revectorize.return_value = {"succeeded": [{"job_id": "job-1"}], "failed": []}

        response = await service_tickets.revectorize_ticket("ticket-1")

        self.assertIn('"succeeded"', response.body.decode("utf-8"))
        mock_revectorize.assert_awaited_once_with("ticket-1")

    @patch("app.api.v1.admin.service_tickets.service_ticket_service.delete_ticket")
    async def test_delete_service_ticket_delegates_to_service(self, mock_delete):
        mock_delete.return_value = {"id": "ticket-1", "deleted": True}

        response = await service_tickets.delete_service_ticket("ticket-1")

        self.assertEqual(response.body.decode("utf-8"), '{"success":true,"data":{"id":"ticket-1","deleted":true}}')
        mock_delete.assert_called_once_with("ticket-1")


if __name__ == "__main__":
    unittest.main()
