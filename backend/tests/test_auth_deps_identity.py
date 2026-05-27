# -*- coding: utf-8 -*-
import unittest
from types import SimpleNamespace

from app.api.v1.auth_deps import get_request_identity


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
