# -*- coding: utf-8 -*-
import json
import sys
import types
import unittest
from io import BytesIO
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

from fastapi import BackgroundTasks, HTTPException, UploadFile

from app.api.v1 import chat, conversations, documents, files, jobs, knowledge_bases, knowledge_graph, system
from app.api.v1.admin import collection, config
from app.api.v1.auth import LoginRequest, login, logout
from app.api.v1.categories import BatchDeleteFilesRequest, CategoryCreate, CategoryUpdate
from app.api.v1 import categories, chunks
from app.models.requests import ChatRequest, Message


def _json_body(response):
    return json.loads(response.body.decode("utf-8"))


class ApiRouteContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_system_supports_doubao_mini_model(self):
        self.assertIn("doubao-seed-2-0-mini-260428", system.SUPPORTED_MODELS)

    @patch("app.api.v1.system.SUPPORTED_MODELS", {"demo-model": {"description": "Demo", "provider": "mock", "max_tokens": 4096}})
    async def test_system_endpoints_return_expected_shapes(self):
        root_payload = await system.root()
        models_payload = await system.get_models()
        health_payload = await system.health()

        self.assertEqual(root_payload["status"], "ok")
        self.assertIn("demo-model", root_payload["models"])
        self.assertEqual(models_payload.models[0].name, "demo-model")
        self.assertEqual(health_payload.status, "healthy")

    @patch("app.api.v1.auth.create_access_token", return_value="token-123")
    @patch("app.api.v1.auth.authenticate_admin", return_value={"id": "u1", "username": "ops01", "role": "sub_admin"})
    async def test_auth_login_success_returns_token_payload(self, mock_authenticate, mock_create_token):
        response = await login(LoginRequest(username="ops01", password="secret"))

        payload = _json_body(response)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["access_token"], "token-123")
        self.assertEqual(payload["data"]["user"]["role"], "admin")
        self.assertEqual(payload["data"]["user"]["admin_role"], "sub_admin")
        mock_authenticate.assert_called_once_with("ops01", "secret")
        mock_create_token.assert_called_once_with(
            subject="u1",
            username="ops01",
            role="admin",
            admin_role="sub_admin",
        )

    @patch("app.api.v1.auth.authenticate_admin", return_value=None)
    async def test_auth_login_rejects_invalid_credentials(self, _mock_authenticate):
        with self.assertRaises(HTTPException) as ctx:
            await login(LoginRequest(username="admin", password="bad"))

        self.assertEqual(ctx.exception.status_code, 401)

    async def test_auth_logout_returns_success(self):
        payload = _json_body(await logout())
        self.assertTrue(payload["success"])

    @patch("app.api.v1.admin.config.get_milvus_service")
    async def test_admin_config_reads_settings_and_collections(self, mock_get_milvus_service):
        mock_get_milvus_service.return_value.list_collections.return_value = ["kb-a", "kb-b"]

        payload = _json_body(await config.get_config())

        self.assertEqual(payload["data"]["milvus"]["collections"], ["kb-a", "kb-b"])
        self.assertIn("embedding", payload["data"])

    @patch("app.api.v1.admin.collection.get_kb_repository")
    async def test_collection_list_returns_total(self, mock_get_repo):
        mock_get_repo.return_value.list_all.return_value = [{"name": "kb-a"}]

        payload = _json_body(await collection.list_collections())

        self.assertEqual(payload["data"]["total"], 1)

    @patch("app.api.v1.admin.collection.get_milvus_service")
    @patch("app.api.v1.admin.collection.get_kb_repository")
    async def test_collection_create_builds_collection_and_persists_record(self, mock_get_repo, mock_get_milvus_service):
        repo = mock_get_repo.return_value
        repo.get_by_name.return_value = None
        repo.create.return_value = {"id": "kb-1", "name": "kb-a"}

        req = collection.CreateKbRequest(
            name="kb-a",
            retrieval_config=collection.RetrievalConfig(memory_turns=4, kg_enabled=True),
        )

        payload = _json_body(await collection.create_collection(req))

        self.assertTrue(payload["success"])
        mock_get_milvus_service.return_value.get_or_create_collection.assert_called_once()
        self.assertEqual(repo.create.call_args.kwargs["name"], "kb-a")
        self.assertEqual(repo.create.call_args.kwargs["retrieval_config"]["memory_turns"], 4)

    @patch("app.db.get_file_repository")
    @patch("app.api.v1.admin.collection.get_milvus_service")
    @patch("app.api.v1.admin.collection.get_kb_repository")
    async def test_collection_delete_handles_milvus_failure_and_deletes_repo_record(self, mock_get_repo, mock_get_milvus_service, mock_get_file_repo):
        repo = mock_get_repo.return_value
        repo.get_by_name.return_value = {"id": "kb-1", "name": "kb-a"}
        mock_get_file_repo.return_value.list_by_kb.return_value = []
        mock_get_milvus_service.return_value.delete_collection.side_effect = RuntimeError("milvus down")

        payload = _json_body(await collection.delete_collection("kb-a"))

        self.assertTrue(payload["success"])
        repo.delete.assert_called_once_with("kb-1")

    @patch("app.api.v1.categories.category_service")
    async def test_categories_endpoints_delegate_to_service(self, mock_category_service):
        mock_category_service.list_categories.return_value = [{"id": "c1"}]
        mock_category_service.create_category.return_value = {"id": "c1", "name": "Ops"}
        mock_category_service.get_category_with_files.return_value = {"id": "c1", "files": []}
        mock_category_service.update_category.return_value = {"id": "c1", "name": "New Ops"}
        mock_category_service.delete_category_file.return_value = "handbook.pdf"
        mock_category_service.batch_delete_category_files.return_value = {"deleted": ["f1"], "failed": []}

        self.assertEqual(_json_body(await categories.list_categories())["data"][0]["id"], "c1")
        self.assertEqual(_json_body(await categories.create_category(CategoryCreate(name="Ops")))["data"]["id"], "c1")
        self.assertEqual(_json_body(await categories.get_category("c1"))["data"]["id"], "c1")
        self.assertEqual(_json_body(await categories.update_category("c1", CategoryUpdate(name="New Ops")))["data"]["name"], "New Ops")
        self.assertTrue(_json_body(await categories.delete_category("c1"))["success"])
        self.assertIn("handbook.pdf", _json_body(await categories.delete_category_file("c1", "f1"))["message"])
        self.assertEqual(
            _json_body(await categories.batch_delete_category_files("c1", BatchDeleteFilesRequest(file_ids=["f1"])))["data"]["deleted"],
            ["f1"],
        )

    @patch("app.api.v1.files.file_service")
    async def test_files_management_routes_delegate_to_service(self, mock_file_service):
        mock_file_service.list_files.return_value = {"files": [{"id": "f1"}], "total": 1}
        mock_file_service.delete_file.return_value = "guide.pdf"
        mock_file_service.batch_delete_files.return_value = {"deleted": ["f1"], "failed": []}

        self.assertEqual((await files.list_files("kb-a"))["data"]["total"], 1)
        self.assertIn("guide.pdf", (await files.delete_file(files.DeleteFileRequest(file_id="f1")))["message"])
        self.assertEqual((await files.batch_delete_files(files.BatchDeleteFilesRequest(file_ids=["f1"], kb_name="kb-a")))["data"]["deleted"], ["f1"])

    @patch("app.api.v1.files.get_file_storage_repository")
    async def test_serve_file_returns_bytes_and_guessed_mime(self, mock_get_repo):
        repo = mock_get_repo.return_value
        repo.get_by_key.return_value = {"mime_type": None}
        repo.get_bytes.return_value = b"%PDF-1.4"

        response = await files.serve_file("docs/guide.pdf")

        self.assertEqual(response.media_type, "application/pdf")
        self.assertEqual(response.body, b"%PDF-1.4")

    @patch("app.api.v1.chunks.chunk_service")
    async def test_chunk_routes_cover_sync_and_async_paths(self, mock_chunk_service):
        mock_chunk_service.get_chunks_by_job.return_value = [{"idx": 0}]
        mock_chunk_service.clean_single_chunk.return_value = "cleaned"
        mock_chunk_service.clean_job_chunks = AsyncMock(return_value={"success": 2, "total": 2})
        mock_chunk_service.clean_all_chunks = AsyncMock(return_value={"success": 3, "failed": 0})
        mock_chunk_service.upsert_job_chunks = AsyncMock(return_value={"job_id": "job-1"})
        mock_chunk_service.batch_upsert_jobs = AsyncMock(return_value={"succeeded": ["job-1"], "failed": []})
        mock_chunk_service.get_chunk_images.return_value = [{"id": "img-1"}]
        mock_chunk_service.add_chunk_image.return_value = {"id": "img-1"}
        mock_chunk_service.resolve_image_placeholders.return_value = {"placeholder": "url"}

        self.assertEqual(_json_body(await chunks.get_chunks_by_job("job-1"))["data"][0]["idx"], 0)
        self.assertEqual(_json_body(await chunks.clean_single_chunk("job-1", 0, instruction="trim"))["data"]["content"], "cleaned")
        self.assertTrue(_json_body(await chunks.clean_job_chunks("job-1"))["success"])
        self.assertTrue(_json_body(await chunks.clean_all_chunks())["success"])
        self.assertEqual(_json_body(await chunks.upsert_job_chunks("job-1"))["data"]["job_id"], "job-1")
        self.assertEqual(_json_body(await chunks.batch_upsert_jobs(chunks.BatchUpsertRequest(job_ids=["job-1"])))["data"]["succeeded"], ["job-1"])
        self.assertEqual(_json_body(await chunks.get_chunk_images("job-1", 0))["data"]["images"][0]["id"], "img-1")
        self.assertEqual(
            _json_body(
                await chunks.add_chunk_image(
                    "job-1",
                    0,
                    UploadFile(filename="chunk.png", file=BytesIO(b"img")),
                    page=2,
                    insert_position=1,
                )
            )["data"]["id"],
            "img-1",
        )
        self.assertEqual(
            _json_body(await chunks.resolve_images(chunks.ResolveImagesRequest(placeholders=["placeholder"])))["data"]["placeholder"],
            "url",
        )

    @patch("app.services.oss_service.get_oss_service")
    async def test_resolve_oss_keys_skips_failed_items(self, mock_get_oss_service):
        oss_service = mock_get_oss_service.return_value
        oss_service.get_presigned_url.side_effect = lambda key, expires=3600: (_ for _ in ()).throw(RuntimeError("boom")) if key == "bad" else f"url:{key}"

        payload = _json_body(await chunks.resolve_oss_keys(chunks.ResolveOssKeysRequest(oss_keys=["good", "bad"])))

        self.assertEqual(payload["data"], {"good": "url:good"})

    @patch("app.api.v1.jobs.job_service.upsert_job_to_milvus", new_callable=AsyncMock)
    @patch("app.api.v1.jobs.job_service.get_job_detail", return_value={"id": "job-1"})
    @patch("app.api.v1.jobs.job_service.list_jobs", return_value={"jobs": [{"id": "job-1"}], "total": 1})
    async def test_jobs_routes_delegate_to_service(self, mock_list_jobs, mock_get_job_detail, mock_upsert_job):
        mock_upsert_job.return_value = {"job_id": "job-1"}

        self.assertEqual(_json_body(await jobs.list_jobs("kb-a"))["data"]["total"], 1)
        self.assertEqual(_json_body(await jobs.get_job("job-1"))["data"]["id"], "job-1")
        self.assertEqual(_json_body(await jobs.upsert_job("job-1"))["data"]["job_id"], "job-1")

    @patch("app.api.v1.knowledge_bases.get_kb_repository")
    async def test_public_knowledge_base_list_filters_public_fields(self, mock_get_repo):
        mock_get_repo.return_value.list_all.return_value = [
            {"name": "kb-a", "display_name": "A", "description": "desc", "kb_type": "multimodal", "image_mode": 1, "secret": "x"}
        ]

        payload = _json_body(await knowledge_bases.list_public_knowledge_bases())

        self.assertEqual(payload["data"]["collections"][0]["name"], "kb-a")
        self.assertNotIn("secret", payload["data"]["collections"][0])

    @patch("app.api.v1.knowledge_graph.file_service.list_files")
    @patch("app.api.v1.knowledge_graph.get_kb_repository")
    async def test_knowledge_graph_returns_empty_when_no_synced_files(self, mock_get_repo, mock_list_files):
        mock_get_repo.return_value.get_by_name.return_value = {"id": "kb-1"}
        mock_list_files.return_value = {"files": [{"id": "f1", "sync_graph": False}], "total": 1}

        payload = _json_body(await knowledge_graph.get_kb_graph("kb-a"))

        self.assertEqual(payload["data"]["total_files"], 0)

    @patch("app.api.v1.knowledge_graph.kg_graph_sync_service.get_kg_graph_sync_service")
    @patch("app.api.v1.knowledge_graph.file_service.list_files")
    @patch("app.api.v1.knowledge_graph.get_kb_repository")
    async def test_knowledge_graph_aggregates_triples(self, mock_get_repo, mock_list_files, mock_get_sync_service):
        mock_get_repo.return_value.get_by_name.return_value = {"id": "kb-1"}
        mock_list_files.return_value = {
            "files": [{"id": "f1", "file_name": "guide.pdf", "sync_graph": True, "job": {"id": "job-1"}}],
            "total": 1,
        }
        mock_get_sync_service.return_value.query_graph = AsyncMock(return_value={"triples": [{"s": "A", "p": "B", "o": "C"}]})

        payload = _json_body(await knowledge_graph.get_kb_graph("kb-a"))

        self.assertEqual(payload["data"]["total_triples"], 1)
        self.assertEqual(payload["data"]["files"][0]["triples_count"], 1)

    @patch("app.api.v1.documents.document_service.upload_document", new_callable=AsyncMock)
    async def test_document_upload_forwards_form_fields(self, mock_upload_document):
        mock_upload_document.return_value = {"job_id": "job-1"}
        background_tasks = BackgroundTasks()
        upload = UploadFile(filename="guide.txt", file=BytesIO(b"hello"))

        payload = _json_body(
            await documents.upload_document(
                background_tasks=background_tasks,
                file=upload,
                kb_name="kb-a",
                chunk_size=256,
                chunk_overlap=32,
                excel_rows_per_chunk=2,
                image_dpi=120,
                sync_graph=True,
            )
        )

        self.assertEqual(payload["data"]["job_id"], "job-1")
        self.assertEqual(mock_upload_document.await_args.kwargs["chunk_size"], 256)
        self.assertTrue(mock_upload_document.await_args.kwargs["sync_graph"])

    @patch("app.api.v1.documents.document_service.batch_upload_to_category", new_callable=AsyncMock)
    async def test_document_batch_upload_counts_success_and_failures(self, mock_batch_upload):
        mock_batch_upload.return_value = {"succeeded": ["a"], "failed": ["b"], "total": 2}
        uploads = [
            UploadFile(filename="a.txt", file=BytesIO(b"a")),
            UploadFile(filename="b.txt", file=BytesIO(b"b")),
        ]

        payload = _json_body(await documents.batch_upload_to_category(files=uploads, category_id="cat-1"))

        self.assertEqual(payload["data"]["total"], 2)

    async def test_document_start_chunking_excel_rejects_invalid_json(self):
        response = await documents.start_chunking_excel(
            category_id="cat-1",
            background_tasks=BackgroundTasks(),
            kb_name="kb-a",
            excel_configs="{bad json",
        )

        self.assertEqual(response.status_code, 422)

    @patch("app.api.v1.documents.document_service.start_chunking_excel", new_callable=AsyncMock)
    async def test_document_start_chunking_excel_parses_json(self, mock_start_chunking_excel):
        mock_start_chunking_excel.return_value = {"submitted": 1}

        payload = _json_body(
            await documents.start_chunking_excel(
                category_id="cat-1",
                background_tasks=BackgroundTasks(),
                kb_name="kb-a",
                excel_configs='{"file-1": {"Sheet1": ["A"]}}',
            )
        )

        self.assertEqual(payload["data"]["submitted"], 1)
        self.assertEqual(mock_start_chunking_excel.await_args.kwargs["excel_configs"]["file-1"]["Sheet1"], ["A"])

    @patch("app.api.v1.documents.document_service.start_chunking", new_callable=AsyncMock)
    async def test_document_start_chunking_forwards_arguments(self, mock_start_chunking):
        mock_start_chunking.return_value = {"submitted": 3}

        payload = _json_body(
            await documents.start_chunking(
                category_id="cat-1",
                background_tasks=BackgroundTasks(),
                kb_name="kb-a",
                chunk_size=200,
                chunk_overlap=20,
                sync_graph=True,
            )
        )

        self.assertEqual(payload["data"]["submitted"], 3)
        self.assertEqual(mock_start_chunking.await_args.kwargs["chunk_size"], 200)

    @patch("app.api.v1.documents.document_service.search_documents")
    async def test_document_search_validates_kb_and_delegates(self, mock_search_documents):
        invalid = await documents.search_documents(query="hello", kb_name="", collection="")
        self.assertEqual(invalid.status_code, 422)

        mock_search_documents.return_value = [{"id": "doc-1"}]
        payload = _json_body(await documents.search_documents(query="hello", kb_name=None, collection="kb-a", hybrid_search="Weighted", rerank=True))

        self.assertEqual(payload["data"]["total"], 1)
        self.assertEqual(mock_search_documents.call_args.kwargs["kb_name"], "kb-a")

    @patch("app.api.v1.documents.get_oss_service")
    async def test_document_image_proxy_returns_expected_media_type(self, mock_get_oss_service):
        mock_get_oss_service.return_value.get_object_bytes.return_value = b"img"

        response = await documents.image_proxy("path/pic.webp")

        self.assertEqual(response.media_type, "image/webp")
        self.assertEqual(response.body, b"img")

    @patch("app.api.v1.conversations.conversation_service")
    async def test_conversation_routes_delegate_with_user_id(self, mock_conversation_service):
        mock_conversation_service.list_sessions.return_value = {"sessions": [], "total": 0}
        mock_conversation_service.create_session.return_value = {"id": "s1"}
        mock_conversation_service.get_session_messages.return_value = [{"role": "assistant", "content": "hi"}]

        self.assertEqual(_json_body(await conversations.list_sessions(kb_name="kb-a", user_id="guest-1"))["data"]["total"], 0)
        self.assertEqual(
            _json_body(
                await conversations.create_session(
                    conversations.CreateSessionRequest(kb_name="kb-a", title="Title"),
                    user_id="guest-1",
                )
            )["data"]["id"],
            "s1",
        )
        self.assertEqual(_json_body(await conversations.get_messages("s1", user_id="guest-1"))["data"][0]["role"], "assistant")
        self.assertTrue(_json_body(await conversations.delete_session("s1", user_id="guest-1"))["success"])

    @patch("app.api.v1.chat.invoke_chat", new_callable=AsyncMock)
    async def test_chat_route_builds_request_payload(self, mock_invoke_chat):
        mock_invoke_chat.return_value = {
            "request_id": "r1",
            "session_id": "s1",
            "messages": [{"role": "assistant", "content": "hi"}],
            "model": "demo-model",
            "usage": {"total_tokens": 1},
            "thoughts": {"trace": []},
        }
        request = ChatRequest(messages=[Message(role="user", content="hello")], model="demo-model")

        response = await chat.chat(request)

        self.assertEqual(response.request_id, "r1")
        self.assertEqual(mock_invoke_chat.await_args.kwargs["messages"][0]["content"], "hello")
