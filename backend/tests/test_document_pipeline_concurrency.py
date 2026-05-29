# -*- coding: utf-8 -*-
import asyncio
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("PG_HOST", "localhost")
os.environ.setdefault("PG_USER", "tester")
os.environ.setdefault("PG_PASSWORD", "tester")
os.environ.setdefault("MILVUS_HOST", "localhost")
os.environ.setdefault("VOLCES_API_KEY", "test-key")

if "dashscope" not in sys.modules:
    dashscope_module = types.ModuleType("dashscope")

    class _TextEmbedding:
        @staticmethod
        def call(*args, **kwargs):
            raise NotImplementedError

    dashscope_module.TextEmbedding = _TextEmbedding
    dashscope_module.api_key = ""
    sys.modules["dashscope"] = dashscope_module

if "pymilvus" not in sys.modules:
    pymilvus_module = types.ModuleType("pymilvus")

    class _MilvusClient:
        @staticmethod
        def create_schema(*args, **kwargs):
            raise NotImplementedError

        def __init__(self, *args, **kwargs):
            pass

    pymilvus_module.MilvusClient = _MilvusClient
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

from app.services import document_service


class DocumentPipelineConcurrencyTests(unittest.TestCase):
    def test_pipeline_concurrency_default_is_two(self):
        self.assertEqual(document_service._pipeline_concurrency_limit(), 2)

    def test_start_backend_script_uses_two_workers(self):
        root = Path(__file__).resolve().parents[2]
        script = (root / "start_backend.bat").read_text(encoding="utf-8")

        self.assertIn("--workers 2", script)

    def test_run_pipeline_respects_process_semaphore(self):
        state = {"active": 0, "max_active": 0}

        async def fake_run_job_pipeline(**kwargs):
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            await asyncio.sleep(0.02)
            state["active"] -= 1

        async def runner():
            with patch("app.services.document_service._get_pipeline_semaphore", return_value=asyncio.Semaphore(1)), patch(
                "app.services.job_service.run_job_pipeline",
                new=fake_run_job_pipeline,
            ):
                await asyncio.gather(
                    document_service._run_pipeline(
                        job_id="job-1",
                        file_id="file-1",
                        kb_id="kb-1",
                        kb_name="kb",
                        file_name="a.txt",
                        oss_key="oss/a.txt",
                        image_mode=False,
                        chunk_size=100,
                        chunk_overlap=10,
                    ),
                    document_service._run_pipeline(
                        job_id="job-2",
                        file_id="file-2",
                        kb_id="kb-1",
                        kb_name="kb",
                        file_name="b.txt",
                        oss_key="oss/b.txt",
                        image_mode=False,
                        chunk_size=100,
                        chunk_overlap=10,
                    ),
                )

        asyncio.run(runner())
        self.assertEqual(state["max_active"], 1)


if __name__ == "__main__":
    unittest.main()
