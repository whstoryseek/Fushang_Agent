# -*- coding: utf-8 -*-
"""
未回答问题仓储（独立表，与会话脱钩，长期保留）
表：unanswered_question
"""
import json
import logging
from typing import Any, Dict, List, Optional
from app.db.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class UnansweredRepository(BaseRepository):

    def create(
        self,
        *,
        session_id: Optional[str] = None,
        query: str,
        answer: Optional[str] = None,
        fallback_reason: Optional[str] = None,
        quality_level: Optional[str] = None,
        confidence: Optional[float] = None,
        kb_name: Optional[str] = None,
        sources: Optional[list] = None,
    ) -> Dict[str, Any]:
        rows = self._execute_returning(
            """
            INSERT INTO unanswered_question
                (session_id, query, answer, fallback_reason, quality_level, confidence, kb_name, sources)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *
            """,
            (
                session_id,
                query,
                answer,
                fallback_reason,
                quality_level,
                confidence,
                kb_name,
                json.dumps(sources or []),
            ),
        )
        return self._norm(rows[0]) if rows else {}

    def list(
        self,
        *,
        kb_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        conditions = ["1=1"]
        params: list = []

        if kb_name:
            conditions.append("kb_name = %s")
            params.append(kb_name)
        if start_date:
            conditions.append("created_at >= %s")
            params.append(f"{start_date} 00:00:00")
        if end_date:
            conditions.append("created_at <= %s")
            params.append(f"{end_date} 23:59:59")

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT * FROM unanswered_question
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        rows = self._execute_select(sql, tuple(params + [limit, offset]))
        return [self._norm(r) for r in rows]

    def count(
        self,
        *,
        kb_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        conditions = ["1=1"]
        params: list = []

        if kb_name:
            conditions.append("kb_name = %s")
            params.append(kb_name)
        if start_date:
            conditions.append("created_at >= %s")
            params.append(f"{start_date} 00:00:00")
        if end_date:
            conditions.append("created_at <= %s")
            params.append(f"{end_date} 23:59:59")

        where_clause = " AND ".join(conditions)
        rows = self._execute_select(
            f"SELECT COUNT(*) AS total FROM unanswered_question WHERE {where_clause}",
            tuple(params),
        )
        return rows[0]["total"] if rows else 0

    def stats(
        self,
        *,
        kb_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        conditions = ["1=1"]
        params: list = []

        if kb_name:
            conditions.append("kb_name = %s")
            params.append(kb_name)
        if start_date:
            conditions.append("created_at >= %s")
            params.append(f"{start_date} 00:00:00")
        if end_date:
            conditions.append("created_at <= %s")
            params.append(f"{end_date} 23:59:59")

        where_clause = " AND ".join(conditions)

        # 总数
        total_rows = self._execute_select(
            f"SELECT COUNT(*) AS total FROM unanswered_question WHERE {where_clause}",
            tuple(params),
        )
        total = total_rows[0]["total"] if total_rows else 0

        # 按日期统计
        daily_sql = f"""
            SELECT DATE(created_at) AS date, COUNT(*) AS count
            FROM unanswered_question
            WHERE {where_clause}
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            LIMIT 30
        """
        daily_rows = self._execute_select(daily_sql, tuple(params))
        daily = [
            {"date": str(r["date"]), "count": r["count"]}
            for r in daily_rows
        ]

        return {"total": total, "daily": daily}

    @staticmethod
    def _norm(row: Dict[str, Any]) -> Dict[str, Any]:
        sources = row.get("sources")
        if isinstance(sources, str):
            try:
                sources = json.loads(sources)
            except Exception:
                sources = []
        return {
            "id": str(row["id"]),
            "session_id": str(row["session_id"]) if row.get("session_id") else None,
            "query": row.get("query", ""),
            "answer": row.get("answer", ""),
            "fallback_reason": row.get("fallback_reason"),
            "quality_level": row.get("quality_level"),
            "confidence": float(row["confidence"]) if row.get("confidence") is not None else None,
            "kb_name": row.get("kb_name"),
            "sources": sources or [],
            "created_at": str(row["created_at"]) if row.get("created_at") else None,
        }


_instance: Optional[UnansweredRepository] = None


def get_unanswered_repository() -> UnansweredRepository:
    global _instance
    if _instance is None:
        _instance = UnansweredRepository()
    return _instance
