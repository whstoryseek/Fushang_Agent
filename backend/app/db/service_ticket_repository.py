# -*- coding: utf-8 -*-
"""
服务记录 / 工单仓储。

每次店长提问都写入 service_ticket；召回切片快照写入
service_ticket_context，便于后台回看和回修原始切片。
"""
import json
from typing import Any, Dict, List, Optional

from app.db.base_repository import BaseRepository


class ServiceTicketRepository(BaseRepository):
    def create_with_contexts(
        self,
        *,
        user_id: str,
        user_name: Optional[str],
        kb_name: Optional[str],
        query: str,
        answer: Optional[str],
        status: str,
        confidence: Optional[float] = None,
        fallback_reason: Optional[str] = None,
        quality_level: Optional[str] = None,
        sources: Optional[list] = None,
        channel: str = "web",
        processing_ms: Optional[float] = None,
        session_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        requester_name: Optional[str] = None,
        clarification_round: int = 0,
        clarification: Optional[Dict[str, Any]] = None,
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        rows = self._execute_returning(
            """
            INSERT INTO service_ticket
                (session_id, user_id, user_name, kb_name, query, answer, status,
                 confidence, fallback_reason, quality_level, sources, channel, processing_ms,
                 sender_id, requester_name, clarification_round, clarification)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                session_id,
                user_id,
                user_name,
                kb_name,
                query,
                answer,
                status,
                confidence,
                fallback_reason,
                quality_level,
                json.dumps(sources or [], ensure_ascii=False),
                channel or "web",
                processing_ms,
                sender_id,
                requester_name,
                int(clarification_round or 0),
                json.dumps(clarification or {}, ensure_ascii=False),
            ),
        )
        ticket = self._norm_ticket(rows[0]) if rows else {}
        if ticket and contexts:
            self._insert_contexts(ticket["id"], contexts)
        return ticket

    def replace_contexts(self, ticket_id: str, contexts: List[Dict[str, Any]]) -> None:
        self._execute_sql("DELETE FROM service_ticket_context WHERE ticket_id = %s", (ticket_id,))
        self._insert_contexts(ticket_id, contexts)

    def find_active_clarification(
        self,
        *,
        session_id: Optional[str],
        user_id: str,
        kb_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        conditions = ["status = 'clarifying'", "user_id = %s"]
        params: List[Any] = [user_id]
        if session_id:
            conditions.append("session_id = %s")
            params.append(session_id)
        if kb_name:
            conditions.append("(kb_name = %s OR kb_name IS NULL)")
            params.append(kb_name)
        rows = self._execute_select(
            f"""
            SELECT *
            FROM service_ticket
            WHERE {' AND '.join(conditions)}
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            tuple(params),
        )
        return self._norm_ticket(rows[0]) if rows else None

    def _insert_contexts(self, ticket_id: str, contexts: List[Dict[str, Any]]) -> None:
        if not contexts:
            return
        params = []
        for idx, ctx in enumerate(contexts):
            params.append(
                (
                    ticket_id,
                    ctx.get("chunk_id"),
                    ctx.get("job_id"),
                    ctx.get("file_name"),
                    ctx.get("chunk_index"),
                    ctx.get("score"),
                    ctx.get("content") or "",
                    json.dumps(ctx.get("metadata") or {}, ensure_ascii=False),
                    ctx.get("sort_order", idx),
                )
            )
        self._execute_many(
            """
            INSERT INTO service_ticket_context
                (ticket_id, chunk_id, job_id, file_name, chunk_index, score, content, metadata, sort_order)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            params,
        )

    def list(
        self,
        *,
        status: Optional[str] = None,
        kb_name: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        where, params = self._build_filters(
            status=status,
            kb_name=kb_name,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        rows = self._execute_select(
            f"""
            SELECT t.*, COUNT(c.id) AS context_count
            FROM service_ticket t
            LEFT JOIN service_ticket_context c ON c.ticket_id = t.id
            WHERE {where}
            GROUP BY t.id
            ORDER BY t.created_at DESC
            LIMIT %s OFFSET %s
            """,
            tuple(params + [limit, offset]),
        )
        return [self._norm_ticket(r) for r in rows]

    def count(
        self,
        *,
        status: Optional[str] = None,
        kb_name: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        where, params = self._build_filters(
            status=status,
            kb_name=kb_name,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        rows = self._execute_select(
            f"SELECT COUNT(*) AS total FROM service_ticket t WHERE {where}",
            tuple(params),
        )
        return int(rows[0]["total"]) if rows else 0

    def get_with_contexts(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        rows = self._execute_select(
            "SELECT * FROM service_ticket WHERE id = %s LIMIT 1",
            (ticket_id,),
        )
        if not rows:
            return None
        ticket = self._norm_ticket(rows[0])
        ctx_rows = self._execute_select(
            """
            SELECT * FROM service_ticket_context
            WHERE ticket_id = %s
            ORDER BY sort_order ASC, created_at ASC
            """,
            (ticket_id,),
        )
        ticket["contexts"] = [self._norm_context(r) for r in ctx_rows]
        return ticket

    def update(
        self,
        ticket_id: str,
        *,
        status: Optional[str] = None,
        answer: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        parts = ["updated_at = NOW()"]
        params: List[Any] = []
        if status is not None:
            parts.append("status = %s")
            params.append(status)
            if status in ("resolved_ai", "resolved_manual", "ignored"):
                parts.append("resolved_at = COALESCE(resolved_at, NOW())")
        if answer is not None:
            parts.append("answer = %s")
            params.append(answer)
        if note is not None:
            parts.append("note = %s")
            params.append(note)
        params.append(ticket_id)
        rows = self._execute_returning(
            f"UPDATE service_ticket SET {', '.join(parts)} WHERE id = %s RETURNING *",
            tuple(params),
        )
        return self._norm_ticket(rows[0]) if rows else None

    def update_clarification(
        self,
        ticket_id: str,
        *,
        status: str,
        answer: str,
        clarification_round: int,
        clarification: Dict[str, Any],
        sender_id: Optional[str] = None,
        requester_name: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        parts = [
            "updated_at = NOW()",
            "status = %s",
            "answer = %s",
            "clarification_round = %s",
            "clarification = %s",
        ]
        params: List[Any] = [
            status,
            answer,
            int(clarification_round or 0),
            json.dumps(clarification or {}, ensure_ascii=False),
        ]
        if sender_id is not None:
            parts.append("sender_id = %s")
            params.append(sender_id)
        if requester_name is not None:
            parts.append("requester_name = %s")
            params.append(requester_name)
        if note is not None:
            parts.append("note = %s")
            params.append(note)
        params.append(ticket_id)
        rows = self._execute_returning(
            f"UPDATE service_ticket SET {', '.join(parts)} WHERE id = %s RETURNING *",
            tuple(params),
        )
        return self._norm_ticket(rows[0]) if rows else None

    def stats(
        self,
        *,
        kb_name: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        where, params = self._build_filters(
            kb_name=kb_name,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        total_rows = self._execute_select(
            f"SELECT COUNT(*) AS total FROM service_ticket t WHERE {where}",
            tuple(params),
        )
        status_rows = self._execute_select(
            f"""
            SELECT status, COUNT(*) AS count
            FROM service_ticket t
            WHERE {where}
            GROUP BY status
            """,
            tuple(params),
        )
        daily_rows = self._execute_select(
            f"""
            SELECT
                DATE(created_at) AS date,
                COUNT(*) AS count,
                SUM(CASE WHEN status = 'pending_manual' THEN 1 ELSE 0 END) AS pending_manual
            FROM service_ticket t
            WHERE {where}
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            LIMIT 30
            """,
            tuple(params),
        )
        return {
            "total": int(total_rows[0]["total"]) if total_rows else 0,
            "by_status": {r["status"]: int(r["count"]) for r in status_rows},
            "daily": [
                {
                    "date": str(r["date"]),
                    "count": int(r["count"]),
                    "pending_manual": int(r.get("pending_manual") or 0),
                }
                for r in daily_rows
            ],
        }

    @staticmethod
    def _build_filters(
        *,
        status: Optional[str] = None,
        kb_name: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> tuple[str, List[Any]]:
        conditions = ["1=1"]
        params: List[Any] = []
        if status:
            conditions.append("t.status = %s")
            params.append(status)
        if kb_name:
            conditions.append("t.kb_name = %s")
            params.append(kb_name)
        if user_id:
            conditions.append("t.user_id = %s")
            params.append(user_id)
        if start_date:
            conditions.append("t.created_at >= %s")
            params.append(f"{start_date} 00:00:00")
        if end_date:
            conditions.append("t.created_at <= %s")
            params.append(f"{end_date} 23:59:59")
        return " AND ".join(conditions), params

    @staticmethod
    def _loads_json(value: Any, fallback: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return fallback
        return value if value is not None else fallback

    @classmethod
    def _norm_ticket(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(row["id"]),
            "session_id": str(row["session_id"]) if row.get("session_id") else None,
            "user_id": row.get("user_id") or "",
            "user_name": row.get("user_name"),
            "kb_name": row.get("kb_name"),
            "query": row.get("query") or "",
            "answer": row.get("answer") or "",
            "status": row.get("status") or "pending_manual",
            "confidence": float(row["confidence"]) if row.get("confidence") is not None else None,
            "fallback_reason": row.get("fallback_reason"),
            "quality_level": row.get("quality_level"),
            "sources": cls._loads_json(row.get("sources"), []),
            "channel": row.get("channel") or "web",
            "processing_ms": float(row["processing_ms"]) if row.get("processing_ms") is not None else None,
            "sender_id": row.get("sender_id"),
            "requester_name": row.get("requester_name"),
            "clarification_round": int(row.get("clarification_round") or 0),
            "clarification": cls._loads_json(row.get("clarification"), {}),
            "note": row.get("note"),
            "context_count": int(row.get("context_count") or 0),
            "created_at": str(row["created_at"]) if row.get("created_at") else None,
            "updated_at": str(row["updated_at"]) if row.get("updated_at") else None,
            "resolved_at": str(row["resolved_at"]) if row.get("resolved_at") else None,
        }

    @classmethod
    def _norm_context(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(row["id"]),
            "ticket_id": str(row["ticket_id"]),
            "chunk_id": str(row["chunk_id"]) if row.get("chunk_id") else None,
            "job_id": str(row["job_id"]) if row.get("job_id") else None,
            "file_name": row.get("file_name"),
            "chunk_index": row.get("chunk_index"),
            "score": float(row["score"]) if row.get("score") is not None else None,
            "content": row.get("content") or "",
            "metadata": cls._loads_json(row.get("metadata"), {}),
            "sort_order": int(row.get("sort_order") or 0),
            "created_at": str(row["created_at"]) if row.get("created_at") else None,
        }


_instance: Optional[ServiceTicketRepository] = None


def get_service_ticket_repository() -> ServiceTicketRepository:
    global _instance
    if _instance is None:
        _instance = ServiceTicketRepository()
    return _instance
