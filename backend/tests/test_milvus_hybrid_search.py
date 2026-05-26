import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services import document_service
from app.services.milvus_service import MilvusService


class FakeMilvusClient:
    def __init__(self):
        self.last_search = None

    def has_collection(self, collection_name):
        return True

    def hybrid_search(self, **kwargs):
        self.last_search = kwargs
        return [[]]


class MilvusHybridSearchTests(unittest.TestCase):
    def test_weighted_ranker_uses_three_weights_for_multimodal_search(self):
        client = FakeMilvusClient()
        service = MilvusService.__new__(MilvusService)
        service.client = client
        captured_weights = []

        def fake_weighted_ranker(*weights):
            captured_weights.append(weights)
            return {"weights": weights}

        with (
            patch("app.services.milvus_service.WeightedRanker", side_effect=fake_weighted_ranker),
            patch("app.services.milvus_service.AnnSearchRequest", side_effect=lambda **kwargs: kwargs),
        ):
            service.hybrid_search(
                collection_name="kb",
                query="羊毛党定义是什么",
                ranker="Weight",
                hybrid_alpha=0.6,
                query_text_vector=[0.1, 0.2],
                query_image_vector=[0.3, 0.4],
            )

        self.assertEqual(captured_weights, [(0.3, 0.4, 0.3)])
        self.assertEqual(len(client.last_search["reqs"]), 3)

    def test_weighted_ranker_keeps_two_weights_for_text_search(self):
        client = FakeMilvusClient()
        service = MilvusService.__new__(MilvusService)
        service.client = client
        captured_weights = []

        def fake_weighted_ranker(*weights):
            captured_weights.append(weights)
            return {"weights": weights}

        with (
            patch("app.services.milvus_service.WeightedRanker", side_effect=fake_weighted_ranker),
            patch("app.services.milvus_service.AnnSearchRequest", side_effect=lambda **kwargs: kwargs),
        ):
            service.hybrid_search(
                collection_name="kb",
                query="羊毛党定义是什么",
                ranker="Weight",
                hybrid_alpha=0.6,
                query_text_vector=[0.1, 0.2],
            )

        self.assertEqual(captured_weights, [(0.6, 0.4)])
        self.assertEqual(len(client.last_search["reqs"]), 2)


class FakeKbRepository:
    def get_by_name(self, kb_name):
        return {
            "name": kb_name,
            "kb_type": "multimodal",
            "image_mode": True,
            "vector_dim": 1024,
            "retrieval_config": {"image_vector_dim": 1024},
        }


class FakeMultimodalEmbeddingService:
    def embed_text(self, text, dimension=1024):
        return [0.1] * dimension


class FakeMilvusService:
    def __init__(self):
        self.kwargs = None

    def hybrid_search(self, **kwargs):
        self.kwargs = kwargs
        return []


class DocumentSearchTests(unittest.TestCase):
    def test_multimodal_search_uses_multimodal_embedding_dimension(self):
        milvus = FakeMilvusService()

        with (
            patch("app.services.document_service.get_kb_repository", return_value=FakeKbRepository()),
            patch("app.services.multimodal_embedding_service.get_multimodal_embedding_service", return_value=FakeMultimodalEmbeddingService()),
            patch("app.services.milvus_service.get_milvus_service", return_value=milvus),
        ):
            document_service.search_documents(
                query="羊毛党定义是什么",
                kb_name="fushang",
                ranker="Weight",
            )

        self.assertEqual(len(milvus.kwargs["query_text_vector"]), 1024)
        self.assertEqual(len(milvus.kwargs["query_image_vector"]), 1024)


if __name__ == "__main__":
    unittest.main()
