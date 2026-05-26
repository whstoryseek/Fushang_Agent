import unittest
from unittest.mock import MagicMock, patch

from app.services.multimodal_embedding_service import MultimodalEmbeddingService


class MultimodalEmbeddingServiceTests(unittest.TestCase):
    @patch("httpx.post")
    def test_volces_uses_official_endpoint_and_parses_embedding(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "created": 123,
            "data": {"embedding": [0.1, 0.2, 0.3]},
        }
        mock_post.return_value = mock_response

        service = MultimodalEmbeddingService()
        service.base_url = "https://ark.cn-beijing.volces.com/api/v3"
        service.api_key = "test-key"
        service.model = "test-model"
        service.dimension = 1024

        vector = service._embed_volces([{"type": "text", "text": "hello"}], retry=1)

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        mock_post.assert_called_once()
        self.assertEqual(
            mock_post.call_args.args[0],
            "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal",
        )


if __name__ == "__main__":
    unittest.main()
