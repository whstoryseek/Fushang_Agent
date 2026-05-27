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

sys.path.append(str(Path(__file__).resolve().parents[1]))

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

from agents.knowledge.nodes.multimodal_retrieve import multimodal_retrieve
from agents.knowledge.state import RAGConfig, RetrievalStrategy, create_initial_state
from app.services.retrieval_bucket import (
    infer_retrieval_bucket,
    preferred_retrieval_bucket,
    resolve_chunking_strategy,
)


class _FakeMultimodalEmbeddingService:
    def embed_text(self, _text, dimension=1024):
        return [0.1] * min(dimension, 4)

    def embed_image(self, _url, dimension=1024):
        return [0.2] * min(dimension, 4)

    def embed_image_bytes(self, _bytes, dimension=1024):
        return [0.2] * min(dimension, 4)


class RetrievalBucketHeuristicTests(unittest.TestCase):
    def test_infer_retrieval_bucket_prefers_excel_and_qa_names_as_faq(self):
        self.assertEqual(infer_retrieval_bucket("tickets.xlsx"), "faq")
        self.assertEqual(infer_retrieval_bucket("商户常见问题.docx"), "faq")
        self.assertEqual(infer_retrieval_bucket("merchant_qa.txt"), "faq")

    def test_infer_retrieval_bucket_prefers_manual_for_guides(self):
        self.assertEqual(infer_retrieval_bucket("开店手册.pdf"), "manual")
        self.assertEqual(infer_retrieval_bucket("培训教程.docx"), "manual")

    def test_preferred_retrieval_bucket_distinguishes_symptom_and_procedural_queries(self):
        self.assertEqual(preferred_retrieval_bucket("店长端无法打开"), "faq")
        self.assertEqual(preferred_retrieval_bucket("开通微信子商户号需要准备哪些资料？"), "manual")
        self.assertIsNone(preferred_retrieval_bucket("收银"))

    def test_resolve_chunking_strategy_uses_smart_mix(self):
        self.assertEqual(
            resolve_chunking_strategy(
                file_name="merchant_qa.txt",
                chunk_profile="smart_mix",
                requested_chunk_strategy="parent_child",
            ),
            "flat",
        )
        self.assertEqual(
            resolve_chunking_strategy(
                file_name="开店手册.pdf",
                chunk_profile="smart_mix",
                requested_chunk_strategy="flat",
            ),
            "parent_child",
        )


class MultimodalRetrievalBucketRoutingTests(unittest.TestCase):
    def _make_state(self, query: str) -> dict:
        state = create_initial_state(
            query=query,
            user_id="tester",
            session_id="session-1",
            config=RAGConfig(
                collection="kb-demo",
                kb_type="multimodal",
                retrieval_strategy=RetrievalStrategy.HYBRID,
                ranker="Weight",
                hybrid_alpha=0.8,
                multi_doc_top_k=10,
                image_vector_dim=4,
            ),
        )
        state["rewritten_query"] = query
        state["retrieval_strategy"] = RetrievalStrategy.HYBRID
        return state

    def test_multimodal_retrieve_prefers_manual_bucket_for_procedural_query(self):
        calls = []

        class FakeMilvus:
            def hybrid_search(self, **kwargs):
                calls.append(kwargs)
                self_ref = kwargs.get("filter_expr")
                if self_ref == 'retrieval_bucket == "manual"':
                    return [
                        {"chunk_id": "m1", "content": "manual-1", "metadata": {"retrieval_bucket": "manual"}},
                        {"chunk_id": "m2", "content": "manual-2", "metadata": {"retrieval_bucket": "manual"}},
                        {"chunk_id": "m3", "content": "manual-3", "metadata": {"retrieval_bucket": "manual"}},
                    ]
                return []

        with (
            patch("agents.knowledge.nodes.multimodal_retrieve.get_milvus_service", return_value=FakeMilvus()),
            patch(
                "app.services.multimodal_embedding_service.get_multimodal_embedding_service",
                return_value=_FakeMultimodalEmbeddingService(),
            ),
        ):
            result = asyncio.run(multimodal_retrieve(self._make_state("开通微信子商户号需要准备哪些资料？")))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["filter_expr"], 'retrieval_bucket == "manual"')
        self.assertEqual(result["processing_log"][0]["preferred_bucket"], "manual")
        self.assertFalse(result["processing_log"][0]["bucket_fallback"])
        self.assertEqual(len(result["merged_chunks"]), 3)

    def test_multimodal_retrieve_falls_back_to_full_search_when_bucket_hits_are_sparse(self):
        calls = []

        class FakeMilvus:
            def hybrid_search(self, **kwargs):
                calls.append(kwargs)
                filter_expr = kwargs.get("filter_expr")
                if filter_expr == 'retrieval_bucket == "faq"':
                    return [
                        {"chunk_id": "f1", "content": "faq-1", "metadata": {"retrieval_bucket": "faq"}},
                    ]
                if filter_expr is None:
                    return [
                        {"chunk_id": "f1", "content": "faq-1", "metadata": {"retrieval_bucket": "faq"}},
                        {"chunk_id": "f2", "content": "faq-2", "metadata": {"retrieval_bucket": "faq"}},
                        {"chunk_id": "m1", "content": "manual-1", "metadata": {"retrieval_bucket": "manual"}},
                    ]
                return []

        with (
            patch("agents.knowledge.nodes.multimodal_retrieve.get_milvus_service", return_value=FakeMilvus()),
            patch(
                "app.services.multimodal_embedding_service.get_multimodal_embedding_service",
                return_value=_FakeMultimodalEmbeddingService(),
            ),
        ):
            result = asyncio.run(multimodal_retrieve(self._make_state("店长端无法打开")))

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["filter_expr"], 'retrieval_bucket == "faq"')
        self.assertIsNone(calls[1]["filter_expr"])
        self.assertEqual(result["processing_log"][0]["preferred_bucket"], "faq")
        self.assertTrue(result["processing_log"][0]["bucket_fallback"])
        self.assertEqual(result["processing_log"][0]["fallback_reason"], "insufficient_bucket_hits")
        self.assertEqual([chunk["chunk_id"] for chunk in result["merged_chunks"]], ["f1", "f2", "m1"])


if __name__ == "__main__":
    unittest.main()
