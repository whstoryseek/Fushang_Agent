# -*- coding: utf-8 -*-
"""认证用户仓储。"""
from typing import Any, Dict, Optional

from app.db.base_repository import BaseRepository


class AuthUserRepository(BaseRepository):
    def create_user(self, *, username: str, password_hash: str, role: str = "admin") -> Dict[str, Any]:
        rows = self._execute_returning(
            """
            INSERT INTO auth_user(username, password_hash, role)
            VALUES (%s, %s, %s)
            RETURNING *
            """,
            (username, password_hash, role),
        )
        return self._normalize(rows[0]) if rows else {}

    def list_users(self) -> list[Dict[str, Any]]:
        rows = self._execute_select(
            "SELECT * FROM auth_user ORDER BY created_at DESC, username ASC",
        )
        return [self._normalize(row) for row in rows]

    def get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        rows = self._execute_select(
            "SELECT * FROM auth_user WHERE username = %s LIMIT 1",
            (username,),
        )
        return self._normalize(rows[0]) if rows else None

    def get_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        rows = self._execute_select(
            "SELECT * FROM auth_user WHERE id = %s LIMIT 1",
            (user_id,),
        )
        return self._normalize(rows[0]) if rows else None

    def update_password(
        self,
        *,
        user_id: str,
        password_hash: str,
        role: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if role is None:
            rows = self._execute_returning(
                """
                UPDATE auth_user
                SET password_hash = %s, updated_at = NOW()
                WHERE id = %s
                RETURNING *
                """,
                (password_hash, user_id),
            )
        else:
            rows = self._execute_returning(
                """
                UPDATE auth_user
                SET password_hash = %s, role = %s, updated_at = NOW()
                WHERE id = %s
                RETURNING *
                """,
                (password_hash, role, user_id),
            )
        return self._normalize(rows[0]) if rows else None

    @staticmethod
    def _normalize(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(row["id"]),
            "username": row["username"],
            "password_hash": row["password_hash"],
            "role": row.get("role", "admin"),
            "is_active": bool(row.get("is_active", True)),
            "created_at": str(row["created_at"]) if row.get("created_at") else None,
            "updated_at": str(row["updated_at"]) if row.get("updated_at") else None,
        }


_instance: Optional[AuthUserRepository] = None


def get_auth_user_repository() -> AuthUserRepository:
    global _instance
    if _instance is None:
        _instance = AuthUserRepository()
    return _instance
