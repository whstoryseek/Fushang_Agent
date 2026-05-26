import unittest
from unittest.mock import MagicMock, patch

from app.services.embedding_service import EmbeddingService


class EmbeddingServiceTests(unittest.TestCase):
    @patch("httpx.post")
    @patch(
        "app.services.embedding_service.TextEmbedding.call",
        side_effect=AssertionError("DashScope path should not be used"),
    )
    @patch("app.services.embedding_service.time.sleep", return_value=None)
    def test_volces_uses_embeddings_endpoint_and_parses_vectors(
        self,
        _mock_sleep,
        _mock_dashscope_call,
        mock_post,
    ):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2], "index": 1},
                {"embedding": [0.3, 0.4], "index": 0},
            ]
        }
        mock_post.return_value = mock_response

        service = EmbeddingService()
        service.provider = "volces"
        service.api_key = "test-key"
        service.base_url = "https://ark.cn-beijing.volces.com/api/v3"
        service.model = "doubao-embedding-text-240715"

        vectors = service.embed_texts(["alpha", "beta"], dimension=1024)

        self.assertEqual(vectors, [[0.3, 0.4], [0.1, 0.2]])
        mock_post.assert_called_once()
        self.assertEqual(
            mock_post.call_args.args[0],
            "https://ark.cn-beijing.volces.com/api/v3/embeddings",
        )
        self.assertEqual(
            mock_post.call_args.kwargs["json"],
            {
                "model": "doubao-embedding-text-240715",
                "input": ["alpha", "beta"],
                "dimensions": 1024,
                "encoding_format": "float",
            },
        )

    @patch("httpx.post")
    @patch(
        "app.services.embedding_service.TextEmbedding.call",
        side_effect=AssertionError("DashScope path should not be used"),
    )
    @patch("app.services.embedding_service.time.sleep", return_value=None)
    def test_volces_remaps_legacy_dashscope_model_name(
        self,
        _mock_sleep,
        _mock_dashscope_call,
        mock_post,
    ):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [{"embedding": [0.5, 0.6], "index": 0}]
        }
        mock_post.return_value = mock_response

        service = EmbeddingService()
        service.provider = "volces"
        service.api_key = "test-key"
        service.base_url = "https://ark.cn-beijing.volces.com/api/v3"
        service.model = "text-embedding-v3"

        vector = service.embed_query("legacy-model-probe")

        self.assertEqual(vector, [0.5, 0.6])
        self.assertEqual(
            mock_post.call_args.kwargs["json"]["model"],
            "doubao-embedding-text-240715",
        )


if __name__ == "__main__":
    unittest.main()
