# Standard Embedding Volces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make standard knowledge-base embedding work with the existing `VOLCES_API_KEY` so standard KB upsert and RAG query succeed without `DASHSCOPE_API_KEY`.

**Architecture:** Replace the default standard embedding runtime path with a Volces text embedding client while keeping the `EmbeddingService` public contract stable. Preserve multimodal embedding as a separate service and verify the fix with focused regression tests plus a live standard-KB smoke flow.

**Tech Stack:** Python, FastAPI, httpx, Milvus, PostgreSQL, unittest

---

### Task 1: Add Standard Embedding Regression Coverage

**Files:**
- Create: `backend/tests/test_embedding_service.py`
- Reuse: `backend/tests/test_file_storage_repository.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from unittest.mock import MagicMock, patch

from app.services.embedding_service import EmbeddingService


class EmbeddingServiceTests(unittest.TestCase):
    @patch("httpx.post")
    def test_volces_standard_embedding_uses_expected_endpoint_and_parses_vectors(self, mock_post):
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
        self.assertEqual(
            mock_post.call_args.args[0],
            "https://ark.cn-beijing.volces.com/api/v3/embeddings",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
..\\venv\\Scripts\\python.exe -m unittest tests.test_embedding_service
```

Expected: FAIL because `EmbeddingService` still assumes DashScope and does not call the Volces endpoint.

- [ ] **Step 3: Keep the existing file-storage regression green**

Run:

```powershell
..\\venv\\Scripts\\python.exe -m unittest tests.test_file_storage_repository
```

Expected: PASS

### Task 2: Switch Standard Embedding To Volces

**Files:**
- Modify: `backend/app/services/embedding_service.py`
- Modify: `backend/app/core/config.py`
- Modify: `backend/.env.example`

- [ ] **Step 1: Implement Volces-first standard embedding client**

Target shape:

```python
class EmbeddingService:
    def __init__(self):
        self.provider = os.getenv("EMBEDDING_PROVIDER", "volces").lower()
        self.model = settings.embedding_model
        self.batch_size = settings.embedding_batch_size
        self.dimension = settings.embedding_dimension
        self.api_key = settings.volces_api_key
        self.base_url = settings.volces_base_url
```

Use `httpx.post()` against the Volces text embedding endpoint, pass `model`, `input`, and `dimensions`, and normalize the returned list order before returning vectors.

- [ ] **Step 2: Preserve the public contract**

Keep these methods stable:

```python
def embed_texts(self, texts: List[str], dimension: Optional[int] = None) -> List[List[float]]:
    ...

def embed_query(self, text: str, dimension: Optional[int] = None) -> List[float]:
    ...
```

The rest of the RAG pipeline should not need call-site changes.

- [ ] **Step 3: Align config defaults**

Update the config and example env so the default standard embedding setup reflects Volces instead of DashScope:

```python
embedding_model: str = os.getenv("EMBEDDING_MODEL", "doubao-embedding-text-240715")
```

Also adjust required-env validation so standard embedding no longer requires `DASHSCOPE_API_KEY` when `VOLCES_API_KEY` is present.

- [ ] **Step 4: Update setup guidance**

Revise the embedding section in `backend/.env.example` so a fresh setup points at the Volces-based standard embedding path.

### Task 3: Verify Service-Level And End-To-End Behavior

**Files:**
- Reuse: `backend/tests/test_embedding_service.py`
- Reuse: `backend/tests/test_file_storage_repository.py`

- [ ] **Step 1: Run regression tests**

Run:

```powershell
..\\venv\\Scripts\\python.exe -m unittest tests.test_embedding_service tests.test_file_storage_repository tests.test_multimodal_embedding_service
```

Expected: PASS

- [ ] **Step 2: Run a live service smoke call**

Run a small script that imports `EmbeddingService`, calls `embed_text("标准 embedding 实测", dimension=1536)`, and prints the returned vector length.

Expected: a non-empty vector with the configured dimension.

- [ ] **Step 3: Re-run a standard KB end-to-end smoke flow**

Use the live backend to:

```text
1. create a temporary standard KB
2. upload a tiny text file with a known token
3. wait for chunking
4. call /jobs/{job_id}/upsert
5. call /knowledge/ with a question asking for the known token
```

Expected: `upsert_count >= 1` and the answer contains the uploaded token.

- [ ] **Step 4: Record any remaining external blockers**

If the standard path still fails, report the exact failing layer and response payload instead of guessing.
