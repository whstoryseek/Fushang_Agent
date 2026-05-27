# -*- coding: utf-8 -*-
import asyncio
import json
import sys
import types
import unittest
from unittest.mock import patch

if "psycopg2" not in sys.modules:
    psycopg2_module = types.ModuleType("psycopg2")
    psycopg2_module.pool = types.SimpleNamespace(ThreadedConnectionPool=object)
    psycopg2_module.extras = types.SimpleNamespace(RealDictCursor=object)
    sys.modules["psycopg2"] = psycopg2_module
    sys.modules["psycopg2.pool"] = psycopg2_module.pool
    sys.modules["psycopg2.extras"] = psycopg2_module.extras

if "dashscope" not in sys.modules:
    dashscope_module = types.ModuleType("dashscope")
    dashscope_module.TextEmbedding = types.SimpleNamespace(call=lambda *args, **kwargs: None)
    dashscope_module.api_key = ""
    sys.modules["dashscope"] = dashscope_module

if "pymilvus" not in sys.modules:
    pymilvus_module = types.ModuleType("pymilvus")
    pymilvus_module.MilvusClient = object
    pymilvus_module.DataType = types.SimpleNamespace(
        VARCHAR="VARCHAR",
        INT64="INT64",
        SPARSE_FLOAT_VECTOR="SPARSE_FLOAT_VECTOR",
        FLOAT_VECTOR="FLOAT_VECTOR",
    )
    pymilvus_module.Function = object
    pymilvus_module.FunctionType = types.SimpleNamespace(BM25="BM25")
    pymilvus_module.AnnSearchRequest = object
    pymilvus_module.RRFRanker = object
    pymilvus_module.WeightedRanker = object
    sys.modules["pymilvus"] = pymilvus_module

from app.api.v1 import knowledge
from app.models.requests import KnowledgeRequest
from app.services.knowledge_service import stream_knowledge_qa_sse


def _event_payload(chunk: str) -> dict:
    data_line = next(line for line in chunk.splitlines() if line.startswith("data: "))
    return json.loads(data_line.removeprefix("data: "))


class FakeSnapshot:
    def __init__(self, values):
        self.values = values


class FakeStreamAgent:
    def __init__(self, final_values):
        self.final_values = final_values
        self.state_reads = 0

    async def ainvoke(self, initial_state, config=None):
        return {}

    async def aupdate_state(self, config, values):
        return None

    async def aget_state(self, config):
        self.state_reads += 1
        if self.state_reads == 1:
            return FakeSnapshot(
                {
                    "query_intent": "qa",
                    "query_complexity": "simple",
                    "query_keywords": [],
                    "metrics": types.SimpleNamespace(total_chunks_retrieved=1, chunks_after_rerank=1),
                }
            )
        return FakeSnapshot(self.final_values)


async def _one_delta(_messages, _model_name):
    yield "normal streamed answer"


class KnowledgeStreamTicketFlowTests(unittest.IsolatedAsyncioTestCase):
    async def _collect_stream(self, *, convert=False, classifier=None, ticket=None, process_side_effect=None):
        final_values = {
            "answer": "No relevant knowledge found" if convert else "normal streamed answer",
            "confidence": 0.2 if convert else 0.82,
            "sources": [],
            "query_intent": "qa",
            "query_complexity": "simple",
            "query_keywords": [],
            "metrics": types.SimpleNamespace(total_chunks_retrieved=1, chunks_after_rerank=1),
            "used_fallback": bool(convert),
            "fallback_reason": "no_relevant_documents" if convert else None,
            "quality_passed": False if convert else True,
            "answer_quality": "low" if convert else "high",
            "image_map": {},
        }
        ctx = {
            "reranked_chunks": [],
            "messages": [],
            "model_name": "doubao-seed-2-0-pro-260215",
            "image_map": {},
            "is_image_mode": False,
            "is_multimodal_kb": False,
        }
        classifier = classifier or {
            "intent_class": "B",
            "needs_ticket_flow": True,
            "fields": {"issue_detail": "new gap"},
            "missing_fields": ["impact"],
            "question": "Please clarify impact",
        }
        ticket = ticket or {
            "answer": "Please clarify impact",
            "confidence": 0.0,
            "finish_reason": "clarification",
            "clarification": {"intent_class": "B"},
        }

        with patch("agents.knowledge.get_knowledge_stream_prep_agent", return_value=FakeStreamAgent(final_values)), patch(
            "agents.knowledge.create_initial_state",
            return_value={"query": "new gap"},
        ), patch(
            "agents.knowledge.nodes.generate.prepare_generation_context",
            return_value=ctx,
        ), patch(
            "agents.knowledge.nodes.generate.build_sources_from_reranked",
            return_value=[],
        ), patch(
            "agents.knowledge.openai_stream.iter_openai_text_deltas",
            side_effect=lambda messages, model_name: _one_delta(messages, model_name),
        ), patch(
            "app.services.knowledge_service.classify_ticket_intent_with_llm",
            return_value=classifier,
        ) as classify, patch(
            "app.services.knowledge_service.process_ticket_clarification_turn",
            return_value=ticket,
            side_effect=process_side_effect,
        ) as process_ticket, patch(
            "app.services.knowledge_service._persist_conversation_messages"
        ) as persist, patch(
            "app.services.knowledge_service.record_ai_resolved_ticket"
        ) as record_ticket:
            chunks = [
                chunk
                async for chunk in stream_knowledge_qa_sse(
                    query="new gap",
                    model_name="doubao-seed-2-0-pro-260215",
                    session_id="session-1",
                    collection="kb",
                    convert_missing_knowledge_to_ticket=convert,
                    user_id="store-1",
                    user_name="Store",
                    channel="h5",
                    sender_id="sender-1",
                    requester_name="Manager",
                )
            ]
        return chunks, classify, process_ticket, persist, record_ticket

    async def test_stream_conversion_disabled_by_default_keeps_normal_done_and_persists(self):
        chunks, classify, process_ticket, persist, record_ticket = await self._collect_stream(convert=False)
        done_payload = _event_payload(chunks[-1])

        self.assertEqual(done_payload["answer"], "normal streamed answer")
        self.assertEqual(done_payload["finish_reason"], "stop")
        classify.assert_not_called()
        process_ticket.assert_not_called()
        persist.assert_called_once()
        record_ticket.assert_called_once()
        self.assertEqual(record_ticket.call_args.kwargs["status"], "resolved_ai")

    async def test_stream_conversion_enabled_converts_done_event_and_skips_persist(self):
        chunks, classify, process_ticket, persist, record_ticket = await self._collect_stream(convert=True)
        done_payload = _event_payload(chunks[-1])

        self.assertEqual(done_payload["answer"], "Please clarify impact")
        self.assertEqual(done_payload["finish_reason"], "clarification")
        self.assertTrue(done_payload["thoughts"]["clarification_required"])
        classify.assert_called_once()
        process_ticket.assert_called_once()
        persist.assert_not_called()
        record_ticket.assert_not_called()

    async def test_stream_conversion_error_falls_back_to_normal_done_and_persists(self):
        chunks, _classify, _process_ticket, persist, record_ticket = await self._collect_stream(
            convert=True,
            process_side_effect=RuntimeError("ticket write failed"),
        )
        done_payload = _event_payload(chunks[-1])

        self.assertEqual(done_payload["answer"], "No relevant knowledge found")
        self.assertEqual(done_payload["finish_reason"], "stop")
        persist.assert_called_once()
        record_ticket.assert_not_called()

    async def test_stream_pre_rag_classifier_task_converts_before_generation(self):
        final_values = {
            "answer": "normal streamed answer",
            "confidence": 0.82,
            "sources": [],
            "query_intent": "qa",
            "query_complexity": "simple",
            "query_keywords": [],
            "metrics": types.SimpleNamespace(total_chunks_retrieved=1, chunks_after_rerank=1),
            "used_fallback": False,
            "fallback_reason": None,
            "quality_passed": True,
            "answer_quality": "high",
            "image_map": {},
        }
        ctx = {
            "reranked_chunks": [],
            "messages": [],
            "model_name": "doubao-seed-2-0-pro-260215",
            "image_map": {},
            "is_image_mode": False,
            "is_multimodal_kb": False,
        }
        decision = {
            "intent_class": "A",
            "needs_ticket_flow": True,
            "fields": {"issue_detail": "open sub merchant"},
            "missing_fields": ["business_license"],
            "question": "",
        }
        ticket = {
            "answer": "Please provide business license",
            "confidence": 0.0,
            "finish_reason": "clarification",
            "clarification": {"intent_class": "A", "workflow_found": True},
        }

        classifier_task = asyncio.get_running_loop().create_future()
        classifier_task.set_result(decision)

        with patch("agents.knowledge.get_knowledge_stream_prep_agent", return_value=FakeStreamAgent(final_values)), patch(
            "agents.knowledge.create_initial_state",
            return_value={"query": "open sub merchant"},
        ), patch(
            "agents.knowledge.nodes.generate.prepare_generation_context",
            return_value=ctx,
        ), patch(
            "agents.knowledge.nodes.generate.build_sources_from_reranked",
            return_value=[{"file_name": "wechat-submerchant.docx", "content": "SOP"}],
        ), patch(
            "agents.knowledge.openai_stream.iter_openai_text_deltas",
            side_effect=lambda messages, model_name: _one_delta(messages, model_name),
        ) as stream_deltas, patch(
            "app.services.knowledge_service.analyze_operation_workflow_with_llm",
            return_value={"workflow_found": True, "required_fields": ["issue_detail", "business_license"]},
        ) as lookup_workflow, patch(
            "app.services.knowledge_service.process_ticket_clarification_turn",
            return_value=ticket,
        ) as process_ticket, patch(
            "app.services.knowledge_service._persist_conversation_messages"
        ) as persist, patch(
            "app.services.knowledge_service.record_ai_resolved_ticket"
        ) as record_ticket:
            chunks = [
                chunk
                async for chunk in stream_knowledge_qa_sse(
                    query="open sub merchant",
                    model_name="doubao-seed-2-0-pro-260215",
                    session_id="session-1",
                    collection="kb",
                    convert_missing_knowledge_to_ticket=True,
                    user_id="store-1",
                    user_name="Store",
                    channel="h5",
                    sender_id="sender-1",
                    requester_name="Manager",
                    pre_rag_ticket_decision_task=classifier_task,
                )
            ]

        self.assertEqual(len(chunks), 1)
        done_payload = _event_payload(chunks[-1])
        self.assertEqual(done_payload["finish_reason"], "clarification")
        self.assertTrue(done_payload["thoughts"]["clarification_required"])
        lookup_workflow.assert_called_once()
        process_ticket.assert_called_once()
        stream_deltas.assert_not_called()
        persist.assert_not_called()
        record_ticket.assert_not_called()

    async def test_stream_api_enables_missing_knowledge_conversion(self):
        captured = {}

        async def fake_stream(**kwargs):
            captured.update(kwargs)
            yield "event: done\ndata: {}\n\n"

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.stream_knowledge_qa_sse",
            side_effect=fake_stream,
        ), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            return_value={"intent_class": "D", "needs_ticket_flow": False, "fields": {}, "missing_fields": []},
        ):
            response = await knowledge.knowledge_qa_stream(
                KnowledgeRequest(query="new gap", session_id="session-1", collection="kb"),
                user_id="store-1",
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        self.assertEqual(chunks, ["event: done\ndata: {}\n\n"])
        self.assertTrue(captured["convert_missing_knowledge_to_ticket"])
        self.assertEqual(captured["user_id"], "store-1")
        self.assertEqual(captured["channel"], "web")

    async def test_stream_api_pre_rag_c_intent_returns_ticket_done(self):
        ticket = {
            "answer": "请补充具体业务场景或报错信息。",
            "confidence": 0.0,
            "finish_reason": "clarification",
            "clarification": {"intent_class": "C"},
        }

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.find_active_clarification_ticket",
            return_value=None,
            create=True,
        ), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            return_value={
                "intent_class": "C",
                "needs_ticket_flow": True,
                "fields": {},
                "missing_fields": ["issue_detail"],
                "question": "请补充具体业务场景或报错信息。",
            },
        ) as classify, patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn",
            return_value=ticket,
        ) as process_ticket, patch(
            "app.api.v1.knowledge.stream_knowledge_qa_sse"
        ) as normal_stream:
            response = await knowledge.knowledge_qa_stream(
                KnowledgeRequest(query="这个怎么弄", session_id="session-1", collection="kb"),
                user_id="store-1",
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        done_payload = _event_payload(chunks[-1])
        self.assertEqual(done_payload["answer"], "请补充具体业务场景或报错信息。")
        self.assertEqual(done_payload["finish_reason"], "clarification")
        self.assertTrue(done_payload["thoughts"]["clarification_required"])
        classify.assert_called_once()
        process_ticket.assert_called_once()
        normal_stream.assert_not_called()

    async def test_stream_api_waits_for_slightly_slow_a_intent_before_rag(self):
        ticket = {
            "answer": "请补充营业执照、法人信息和银行卡。",
            "confidence": 0.0,
            "finish_reason": "clarification",
            "clarification": {"intent_class": "A", "workflow_found": True},
        }

        def slightly_slow_classifier(*args, **kwargs):
            import time

            time.sleep(0.75)
            return {
                "intent_class": "A",
                "needs_ticket_flow": True,
                "fields": {"issue_detail": "开通微信子商户号"},
                "missing_fields": ["business_license"],
                "question": "",
            }

        async def fake_rag(**kwargs):
            return {
                "answer": "开户 SOP",
                "sources": [{"file_name": "wechat-submerchant.docx", "content": "准备营业执照"}],
                "used_fallback": False,
                "quality_passed": True,
                "quality_level": "high",
            }

        async def normal_stream(**kwargs):
            yield "event: done\ndata: {\"finish_reason\":\"stop\"}\n\n"

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.find_active_clarification_ticket",
            return_value=None,
            create=True,
        ), patch(
            "app.api.v1.knowledge.classify_ticket_intent_with_llm",
            side_effect=slightly_slow_classifier,
        ), patch(
            "app.api.v1.knowledge.invoke_knowledge_qa",
            side_effect=fake_rag,
        ), patch(
            "app.api.v1.knowledge.lookup_operation_workflow",
            return_value={
                "workflow_found": True,
                "required_fields": ["issue_detail", "business_license", "legal_person", "bank_card"],
                "question": "请补充营业执照、法人信息和银行卡。",
            },
        ), patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn",
            return_value=ticket,
        ) as process_ticket, patch(
            "app.api.v1.knowledge.stream_knowledge_qa_sse",
            side_effect=normal_stream,
        ) as normal_stream_mock:
            response = await knowledge.knowledge_qa_stream(
                KnowledgeRequest(query="帮我开通微信子商户号", session_id="session-1", collection="kb"),
                user_id="store-1",
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        done_payload = _event_payload(chunks[-1])
        self.assertEqual(done_payload["finish_reason"], "clarification")
        self.assertEqual(done_payload["thoughts"]["clarification"]["intent_class"], "A")
        process_ticket.assert_called_once()
        normal_stream_mock.assert_not_called()

    async def test_stream_api_active_clarification_reply_continues_ticket(self):
        active = {
            "id": "ticket-1",
            "clarification_round": 1,
            "clarification": {
                "intent_class": "B",
                "original_query": "新问题",
                "required_fields": ["issue_detail", "impact"],
                "missing_fields": ["impact"],
            },
        }
        ticket = {
            "answer": "请继续补充影响范围。",
            "confidence": 0.0,
            "finish_reason": "clarification",
            "clarification": {"intent_class": "B"},
        }

        with patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session", return_value="session-1"), patch(
            "app.api.v1.knowledge.find_active_clarification_ticket",
            return_value=active,
            create=True,
        ), patch(
            "app.api.v1.knowledge.analyze_clarification_reply_with_llm",
            return_value={
                "intent_class": "B",
                "needs_ticket_flow": True,
                "fields": {},
                "missing_fields": ["impact"],
                "question": "请继续补充影响范围。",
            },
            create=True,
        ) as analyze, patch(
            "app.api.v1.knowledge.process_ticket_clarification_turn",
            return_value=ticket,
        ) as process_ticket, patch(
            "app.api.v1.knowledge.stream_knowledge_qa_sse"
        ) as normal_stream:
            response = await knowledge.knowledge_qa_stream(
                KnowledgeRequest(query="影响所有门店", session_id="session-1", collection="kb"),
                user_id="store-1",
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        done_payload = _event_payload(chunks[-1])
        self.assertEqual(done_payload["finish_reason"], "clarification")
        analyze.assert_called_once()
        process_ticket.assert_called_once()
        self.assertTrue(process_ticket.call_args.kwargs["continue_only"])
        normal_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
