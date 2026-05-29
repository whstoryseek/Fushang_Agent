import os
import sys
import types
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

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

from app.services import auth_service


class AuthServiceTests(unittest.TestCase):
    def test_password_hash_verifies_original_password_only(self):
        hashed = auth_service.hash_password("correct horse battery staple")

        self.assertNotEqual(hashed, "correct horse battery staple")
        self.assertTrue(auth_service.verify_password("correct horse battery staple", hashed))
        self.assertFalse(auth_service.verify_password("wrong password", hashed))

    def test_access_token_round_trip_preserves_subject_and_role(self):
        token = auth_service.create_access_token(
            subject="admin-user-id",
            username="admin",
            role="admin",
            admin_role="super_admin",
            expires_delta=timedelta(minutes=5),
        )

        payload = auth_service.decode_access_token(token)

        self.assertEqual(payload["sub"], "admin-user-id")
        self.assertEqual(payload["username"], "admin")
        self.assertEqual(payload["role"], "admin")
        self.assertEqual(payload["admin_role"], "super_admin")

    def test_seed_admin_creates_missing_admin_with_hashed_password(self):
        repo = Mock()
        repo.get_by_username.return_value = None

        auth_service.seed_admin_user(
            username="admin",
            password="admin-secret",
            repo=repo,
        )

        repo.create_user.assert_called_once()
        kwargs = repo.create_user.call_args.kwargs
        self.assertEqual(kwargs["username"], "admin")
        self.assertEqual(kwargs["role"], "super_admin")
        self.assertNotEqual(kwargs["password_hash"], "admin-secret")
        self.assertTrue(auth_service.verify_password("admin-secret", kwargs["password_hash"]))

    def test_seed_admin_updates_existing_admin_password(self):
        repo = Mock()
        repo.get_by_username.return_value = {"id": "user-1", "username": "admin", "role": "admin"}

        auth_service.seed_admin_user(
            username="admin",
            password="new-secret",
            repo=repo,
        )

        repo.update_password.assert_called_once()
        kwargs = repo.update_password.call_args.kwargs
        self.assertEqual(kwargs["user_id"], "user-1")
        self.assertTrue(auth_service.verify_password("new-secret", kwargs["password_hash"]))
        self.assertEqual(kwargs["role"], "super_admin")

    def test_authenticate_admin_accepts_sub_admin_role(self):
        repo = Mock()
        repo.get_by_username.return_value = {
            "id": "user-2",
            "username": "child-admin",
            "password_hash": auth_service.hash_password("88888888"),
            "role": "sub_admin",
            "is_active": True,
        }

        with patch("app.services.auth_service.get_auth_user_repository", return_value=repo):
            user = auth_service.authenticate_admin("child-admin", "88888888")

        self.assertIsNotNone(user)
        self.assertEqual(user["role"], "sub_admin")

    def test_seed_admin_creates_super_admin_with_hashed_password(self):
        repo = Mock()
        repo.get_by_username.return_value = None

        auth_service.seed_admin_user(
            username="admin",
            password="admin-secret",
            repo=repo,
        )

        repo.create_user.assert_called_once()
        kwargs = repo.create_user.call_args.kwargs
        self.assertEqual(kwargs["username"], "admin")
        self.assertEqual(kwargs["role"], "super_admin")
        self.assertNotEqual(kwargs["password_hash"], "admin-secret")
        self.assertTrue(auth_service.verify_password("admin-secret", kwargs["password_hash"]))


if __name__ == "__main__":
    unittest.main()
