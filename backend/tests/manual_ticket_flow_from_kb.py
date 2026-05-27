# -*- coding: utf-8 -*-
"""Manual end-to-end probe against the local fushang knowledge base.

This is intentionally not a pytest test: it calls the live local Postgres,
Milvus, and configured LLM provider, and may create conversation/ticket rows.
"""
import asyncio
import json
import time
import uuid

from app.api.v1.knowledge import knowledge_qa
from app.models.requests import KnowledgeRequest


CASES = [
    {
        "name": "D/tutorial from KB",
        "query": "微信子商户号开户需要准备哪些资料？",
        "expect_finish": "stop",
    },
    {
        "name": "A/operation with SOP",
        "query": "帮我开通微信子商户号",
        "expect_finish": {"clarification", "manual_ticket_created"},
    },
    {
        "name": "C/ambiguous",
        "query": "这个怎么弄",
        "expect_finish": {"clarification", "manual_ticket_created"},
    },
    {
        "name": "B/RAG miss",
        "query": "量子积分税率自动分摊怎么配置？",
        "expect_finish": {"clarification", "manual_ticket_created", "stop"},
    },
]


async def main() -> int:
    results = []
    for idx, case in enumerate(CASES, start=1):
        session_id = str(uuid.uuid4())
        started = time.perf_counter()
        response = await knowledge_qa(
            KnowledgeRequest(
                query=case["query"],
                session_id=session_id,
                collection="fushang",
            ),
            user_id="manual-flow-user",
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        expected = case["expect_finish"]
        ok = response.finish_reason in expected if isinstance(expected, set) else response.finish_reason == expected
        results.append(
            {
                "name": case["name"],
                "ok": ok,
                "elapsed_ms": elapsed_ms,
                "finish_reason": response.finish_reason,
                "answer_preview": response.answer[:180],
                "source_files": [src.get("file_name") for src in (response.sources or [])[:3]],
                "clarification": (response.thoughts or {}).get("clarification"),
                "ticket_intent": (response.thoughts or {}).get("ticket_intent"),
            }
        )

    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return 0 if all(item["ok"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
