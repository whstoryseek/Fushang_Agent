# -*- coding: utf-8 -*-
import asyncio
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

from app.services.knowledge_service import invoke_knowledge_qa


class KnowledgeServiceTicketTests(unittest.TestCase):
    def test_invoke_knowledge_qa_persist_false_returns_raw_miss_metadata(self):
        class FakeAgent:
            async def ainvoke(self, initial_state, config=None):
                return {
                    "answer": "No relevant knowledge found",
                    "confidence": 0.2,
                    "sources": [{"file_name": "policy.md"}],
                    "metrics": SimpleNamespace(total_chunks_retrieved=3, chunks_after_rerank=1),
                    "used_fallback": True,
                    "fallback_reason": "no_relevant_documents",
                    "quality_passed": False,
                    "answer_quality": "low",
                    "image_map": {"img": "url"},
                }

        def fake_create_initial_state(**kwargs):
            return {"query": kwargs["query"], "config": kwargs["config"]}

        with patch("agents.knowledge.get_knowledge_agent", return_value=FakeAgent()), patch(
            "agents.knowledge.create_initial_state",
            side_effect=fake_create_initial_state,
        ), patch(
            "app.services.knowledge_service._load_kb_retrieval",
            return_value=({"id": "kb-1", "name": "kb"}, {}),
        ), patch("app.services.knowledge_service._persist_conversation_messages") as persist:
            result = asyncio.run(
                invoke_knowledge_qa(
                    query="missing answer",
                    model_name="doubao-seed-2-0-pro-260215",
                    session_id="session-1",
                    collection="kb",
                    persist=False,
                )
            )

        self.assertEqual(result["finish_reason"], "stop")
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["fallback_reason"], "no_relevant_documents")
        self.assertFalse(result["quality_passed"])
        self.assertEqual(result["quality_level"], "low")
        self.assertEqual(result["image_map"], {"img": "url"})
        persist.assert_not_called()


if __name__ == "__main__":
    unittest.main()
