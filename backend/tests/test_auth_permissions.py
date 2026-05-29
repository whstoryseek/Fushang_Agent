import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("PG_HOST", "localhost")
os.environ.setdefault("PG_USER", "tester")
os.environ.setdefault("PG_PASSWORD", "tester")
os.environ.setdefault("MILVUS_HOST", "localhost")
os.environ.setdefault("VOLCES_API_KEY", "test-key")
os.environ.setdefault("ADMIN_PASSWORD", "admin-secret")
os.environ.setdefault("AUTH_SECRET_KEY", "test-secret-key")

if "psycopg2" not in sys.modules:
    psycopg2_module = types.ModuleType("psycopg2")
    psycopg2_module.pool = types.SimpleNamespace(ThreadedConnectionPool=object)
    psycopg2_module.extras = types.SimpleNamespace(RealDictCursor=object)
    sys.modules["psycopg2"] = psycopg2_module
    sys.modules["psycopg2.pool"] = psycopg2_module.pool
    sys.modules["psycopg2.extras"] = psycopg2_module.extras

if "dashscope" not in sys.modules:
    dashscope_module = types.ModuleType("dashscope")
    dashscope_module.TextEmbedding = types.SimpleNamespace(call=lambda **kwargs: None)
    sys.modules["dashscope"] = dashscope_module

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.main import create_app
from app.services import auth_service
from app.api.v1.auth_deps import require_admin


class AuthPermissionTests(unittest.TestCase):
    def test_admin_dependency_rejects_missing_token(self):
        app = FastAPI()

        @app.get("/protected")
        async def protected(_admin=Depends(require_admin)):
            return {"ok": True}

        response = TestClient(app).get("/protected")

        self.assertEqual(response.status_code, 401)

    def test_admin_dependency_accepts_super_admin_token(self):
        app = FastAPI()

        @app.get("/protected")
        async def protected(admin=Depends(require_admin)):
            return {"username": admin["username"], "role": admin["role"]}

        token = auth_service.create_access_token(
            subject="user-1",
            username="admin",
            role="admin",
            admin_role="super_admin",
        )

        response = TestClient(app).get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"username": "admin", "role": "admin"})

    def test_admin_dependency_accepts_sub_admin_token(self):
        app = FastAPI()

        @app.get("/protected")
        async def protected(admin=Depends(require_admin)):
            return {"username": admin["username"], "role": admin["role"]}

        token = auth_service.create_access_token(
            subject="user-2",
            username="child-admin",
            role="admin",
            admin_role="sub_admin",
        )

        response = TestClient(app).get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"username": "child-admin", "role": "admin"})

    def test_admin_routes_reject_anonymous_requests_before_db_access(self):
        response = TestClient(create_app()).get("/api/v1/admin/collections")

        self.assertEqual(response.status_code, 401)

    def test_file_management_routes_reject_anonymous_requests_before_db_access(self):
        response = TestClient(create_app()).get("/api/v1/files", params={"kb_name": "kb_demo"})

        self.assertEqual(response.status_code, 401)

    def test_super_admin_can_create_sub_admin_with_default_password(self):
        app = create_app()
        token = auth_service.create_access_token(
            subject="user-1",
            username="admin",
            role="admin",
            admin_role="super_admin",
        )

        with patch("app.api.v1.admin.users.get_auth_user_repository") as mock_get_repo:
            repo = mock_get_repo.return_value
            repo.get_by_username.return_value = None
            repo.create_user.return_value = {
                "id": "user-3",
                "username": "ops01",
                "password_hash": auth_service.hash_password("88888888"),
                "role": "sub_admin",
                "is_active": True,
                "created_at": None,
                "updated_at": None,
            }

            response = TestClient(app).post(
                "/api/v1/admin/users/sub-admin",
                headers={"Authorization": f"Bearer {token}"},
                json={"username": "ops01"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["user"]["username"], "ops01")
        self.assertEqual(data["user"]["role"], "sub_admin")
        self.assertEqual(data["default_password"], "88888888")

    def test_sub_admin_cannot_create_sub_admin(self):
        app = create_app()
        token = auth_service.create_access_token(
            subject="user-2",
            username="child-admin",
            role="admin",
            admin_role="sub_admin",
        )

        response = TestClient(app).post(
            "/api/v1/admin/users/sub-admin",
            headers={"Authorization": f"Bearer {token}"},
            json={"username": "ops01"},
        )

        self.assertEqual(response.status_code, 403)

    def test_public_knowledge_base_list_exposes_safe_fields_without_auth(self):
        fake_kbs = [
            {
                "id": "hidden-id",
                "name": "kb_demo",
                "display_name": "Demo",
                "description": "Public demo",
                "kb_type": "standard",
                "image_mode": False,
                "embedding_model": "hidden-model",
                "vector_dim": 2560,
                "retrieval_config": {"hidden": True},
            }
        ]
        with patch("app.api.v1.knowledge_bases.get_kb_repository") as mock_get_repo:
            mock_get_repo.return_value.list_all.return_value = fake_kbs

            response = TestClient(create_app()).get("/api/v1/knowledge-bases")

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["total"], 1)
        self.assertEqual(
            data["collections"][0],
            {
                "name": "kb_demo",
                "display_name": "Demo",
                "description": "Public demo",
                "kb_type": "standard",
                "image_mode": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
