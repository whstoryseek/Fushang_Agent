# -*- coding: utf-8 -*-
"""
本地文件存储仓储（替代 OSS）
表：file_storage（BYTEA 存储文件内容）
"""
import logging
from typing import Any, Dict, List, Optional
from app.db.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class FileStorageRepository(BaseRepository):

    def insert(self, file_key: str, content: bytes, mime_type: str = None) -> Dict[str, Any]:
        """插入文件记录，file_key 冲突时覆盖"""
        rows = self._execute_returning(
            """
            INSERT INTO file_storage(file_key, content, mime_type, size)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (file_key) DO UPDATE SET
                content = EXCLUDED.content,
                mime_type = EXCLUDED.mime_type,
                size = EXCLUDED.size,
                created_at = NOW()
            RETURNING *
            """,
            (file_key, content, mime_type, len(content)),
        )
        return self._norm(rows[0]) if rows else {}

    def get_by_key(self, file_key: str) -> Optional[Dict[str, Any]]:
        rows = self._execute_select(
            "SELECT * FROM file_storage WHERE file_key = %s LIMIT 1",
            (file_key,),
        )
        return self._norm(rows[0]) if rows else None

    def get_bytes(self, file_key: str) -> Optional[bytes]:
        rows = self._execute_select(
            "SELECT content FROM file_storage WHERE file_key = %s LIMIT 1",
            (file_key,),
        )
        if not rows:
            return None

        content = rows[0]["content"]
        if isinstance(content, memoryview):
            return content.tobytes()
        return content

    def delete_by_keys(self, file_keys: List[str]) -> int:
        if not file_keys:
            return 0
        placeholders = ",".join(["%s"] * len(file_keys))
        self._execute_sql(
            f"DELETE FROM file_storage WHERE file_key IN ({placeholders})",
            tuple(file_keys),
        )
        return len(file_keys)

    def exists(self, file_key: str) -> bool:
        rows = self._execute_select(
            "SELECT 1 FROM file_storage WHERE file_key = %s LIMIT 1",
            (file_key,),
        )
        return bool(rows)

    @staticmethod
    def _norm(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(row["id"]),
            "file_key": row.get("file_key", ""),
            "mime_type": row.get("mime_type"),
            "size": row.get("size", 0),
            "created_at": str(row["created_at"]) if row.get("created_at") else None,
        }


_instance: Optional[FileStorageRepository] = None


def get_file_storage_repository() -> FileStorageRepository:
    global _instance
    if _instance is None:
        _instance = FileStorageRepository()
    return _instance
