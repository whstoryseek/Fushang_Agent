# -*- coding: utf-8 -*-
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
import importlib.util

if "psycopg2" not in sys.modules:
    psycopg2_module = types.ModuleType("psycopg2")
    psycopg2_module.pool = types.SimpleNamespace(ThreadedConnectionPool=object)
    psycopg2_module.extras = types.SimpleNamespace(RealDictCursor=object)
    sys.modules["psycopg2"] = psycopg2_module
    sys.modules["psycopg2.pool"] = psycopg2_module.pool
    sys.modules["psycopg2.extras"] = psycopg2_module.extras

AUTH_DEPS_PATH = Path(__file__).resolve().parents[1] / "app" / "api" / "v1" / "auth_deps.py"
AUTH_DEPS_SPEC = importlib.util.spec_from_file_location("test_auth_deps_module", AUTH_DEPS_PATH)
AUTH_DEPS_MODULE = importlib.util.module_from_spec(AUTH_DEPS_SPEC)
assert AUTH_DEPS_SPEC and AUTH_DEPS_SPEC.loader
AUTH_DEPS_SPEC.loader.exec_module(AUTH_DEPS_MODULE)
get_request_identity = AUTH_DEPS_MODULE.get_request_identity


class AuthDepsIdentityTests(unittest.TestCase):
    def test_url_params_override_guest_identity_for_h5_entry(self):
        request = SimpleNamespace(
            headers={},
            query_params={"user_id": "store-1", "nickname": "张店长", "sender_id": "sender-9"},
        )

        identity = get_request_identity(request)

        self.assertEqual(identity["user_id"], "store-1")
        self.assertEqual(identity["user_name"], "张店长")
        self.assertEqual(identity["channel"], "h5")
        self.assertEqual(identity["sender_id"], "sender-9")

    def test_guest_header_remains_default_fallback(self):
        request = SimpleNamespace(
            headers={"X-Guest-Id": "guest_abcdefghi"},
            query_params={},
        )

        identity = get_request_identity(request)

        self.assertEqual(identity["user_id"], "guest_abcdefghi")
        self.assertIsNone(identity["user_name"])
        self.assertEqual(identity["channel"], "web")
        self.assertIsNone(identity["sender_id"])

    def test_sender_id_can_arrive_from_reserved_header(self):
        request = SimpleNamespace(
            headers={
                "X-Guest-Id": "guest_abcdefghi",
                "X-Sender-Id": "sender-88",
                "X-Requester-Name": "%E6%9D%8E%E5%BA%97%E9%95%BF",
            },
            query_params={},
        )

        identity = get_request_identity(request)

        self.assertEqual(identity["sender_id"], "sender-88")
        self.assertEqual(identity["requester_name"], "李店长")


if __name__ == "__main__":
    unittest.main()
