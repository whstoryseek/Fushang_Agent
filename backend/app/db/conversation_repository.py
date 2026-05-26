# -*- coding: utf-8 -*-
"""
对话会话 & 消息仓储
表：conversation_session, conversation_message
"""
import json
import logging
from typing import Any, Dict, List, Optional
from app.db.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class ConversationRepository(BaseRepository):

    # ── Session ───────────────────────────────────────────────────────────────

    def create_session(
        self,
        *,
        kb_name: str,
        user_id: str = "default",
        title: str = "新会话",
    ) -> Dict[str, Any]:
        rows = self._execute_returning(
            """
            INSERT INTO conversation_session(user_id, kb_name, title)
            VALUES (%s, %s, %s) RETURNING *
            """,
            (user_id, kb_name, title),
        )
        return self._norm_session(rows[0]) if rows else {}

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        rows = self._execute_select(
            "SELECT * FROM conversation_session WHERE id = %s LIMIT 1", (session_id,)
        )
        return self._norm_session(rows[0]) if rows else None

    def list_sessions(self, kb_name: Optional[str] = None, user_id: str = "default") -> List[Dict[str, Any]]:
        where_sql = "WHERE s.user_id = %s"
        params: list[Any] = [user_id]
        if kb_name:
            where_sql += " AND s.kb_name = %s"
            params.append(kb_name)

        rows = self._execute_select(
            f"""
            SELECT
                s.*,
                COUNT(m.id) AS message_count,
                lm.role AS last_message_role,
                lm.content AS last_message_content
            FROM conversation_session s
            LEFT JOIN conversation_message m ON m.session_id = s.id
            LEFT JOIN LATERAL (
                SELECT role, content
                FROM conversation_message cm
                WHERE cm.session_id = s.id
                ORDER BY cm.created_at DESC
                LIMIT 1
            ) lm ON TRUE
            {where_sql}
            GROUP BY s.id, lm.role, lm.content
            ORDER BY s.updated_at DESC
            """,
            tuple(params),
        )
        return [self._norm_session(r) for r in rows]

    def touch_session(self, session_id: str) -> None:
        """更新 updated_at，用于排序"""
        self._execute_sql(
            "UPDATE conversation_session SET updated_at = NOW() WHERE id = %s",
            (session_id,),
        )

    def delete_session(self, session_id: str) -> None:
        self._execute_sql(
            "DELETE FROM conversation_session WHERE id = %s", (session_id,)
        )

    # ── Message ───────────────────────────────────────────────────────────────

    def add_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[list] = None,
        confidence: Optional[float] = None,
        image_placeholders: Optional[List[str]] = None,
        query_image_oss_key: Optional[str] = None,
        used_fallback: bool = False,
        fallback_reason: Optional[str] = None,
        quality_passed: Optional[bool] = None,
        quality_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        rows = self._execute_returning(
            """
            INSERT INTO conversation_message
                (session_id, role, content, sources, confidence, image_placeholders, query_image_oss_key,
                 used_fallback, fallback_reason, quality_passed, quality_level)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *
            """,
            (
                session_id,
                role,
                content,
                json.dumps(sources or []),
                confidence,
                image_placeholders or [],
                query_image_oss_key,
                used_fallback,
                fallback_reason,
                quality_passed,
                quality_level,
            ),
        )
        return self._norm_message(rows[0]) if rows else {}

    def list_messages(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self._execute_select(
            """
            SELECT * FROM conversation_message
            WHERE session_id = %s
            ORDER BY created_at ASC
            LIMIT %s
            """,
            (session_id, limit),
        )
        return [self._norm_message(r) for r in rows]

    def get_image_placeholders_by_session(self, session_id: str) -> List[str]:
        """收集会话内所有消息的占位符，用于批量 resolve"""
        rows = self._execute_select(
            "SELECT image_placeholders FROM conversation_message WHERE session_id = %s",
            (session_id,),
        )
        result = []
        for r in rows:
            phs = r.get("image_placeholders") or []
            result.extend(phs)
        return list(set(result))

    # ── Normalize ─────────────────────────────────────────────────────────────

    @staticmethod
    def _norm_session(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(row["id"]),
            "user_id": row.get("user_id", "default"),
            "kb_name": row.get("kb_name", ""),
            "title": row.get("title", "新会话"),
            "message_count": int(row.get("message_count", 0)),
            "last_message_role": row.get("last_message_role"),
            "last_message_preview": ConversationRepository._build_message_preview(row.get("last_message_content")),
            "created_at": str(row["created_at"]) if row.get("created_at") else None,
            "updated_at": str(row["updated_at"]) if row.get("updated_at") else None,
        }

    @staticmethod
    def _build_message_preview(content: Optional[str], limit: int = 72) -> str:
        text = " ".join((content or "").split())
        if len(text) <= limit:
            return text
        return f"{text[:limit].rstrip()}..."

    @staticmethod
    def _norm_message(row: Dict[str, Any]) -> Dict[str, Any]:
        sources = row.get("sources")
        if isinstance(sources, str):
            try:
                sources = json.loads(sources)
            except Exception:
                sources = []
        phs = row.get("image_placeholders") or []
        return {
            "id": str(row["id"]),
            "session_id": str(row["session_id"]),
            "role": row.get("role", ""),
            "content": row.get("content", ""),
            "sources": sources or [],
            "confidence": row.get("confidence"),
            "image_placeholders": list(phs),
            "query_image_oss_key": row.get("query_image_oss_key"),
            "used_fallback": row.get("used_fallback", False),
            "fallback_reason": row.get("fallback_reason"),
            "quality_passed": row.get("quality_passed"),
            "quality_level": row.get("quality_level"),
            "created_at": str(row["created_at"]) if row.get("created_at") else None,
        }


_instance: Optional[ConversationRepository] = None


def get_conversation_repository() -> ConversationRepository:
    global _instance
    if _instance is None:
        _instance = ConversationRepository()
    return _instance
