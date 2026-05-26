# -*- coding: utf-8 -*-
import unittest
from unittest.mock import AsyncMock, patch

from app.api.v1 import knowledge
from app.models.requests import KnowledgeRequest


OP_QUERY = "\u5bcc\u53cb\u90a3\u8fb9\u5e2e\u5fd9\u5b89\u6392\u4e00\u4e0b"
HOW_TO_QUERY = "\u5bcc\u53cb\u8d26\u6237\u600e\u4e48\u7ed1\u5b9a"


class KnowledgeApiLlmClarificationTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.api.v1.knowledge.invoke_knowledge_qa", new_callable=AsyncMock)
    @patch("app.api.v1.knowledge.extract_operation_workflow_requirements")
    @patch("app.api.v1.knowledge.retrieve_operation_workflow", new_callable=AsyncMock)
    @patch("app.api.v1.knowledge.classify_operation_query_with_llm")
    @patch("app.api.v1.knowledge.persist_clarification_message")
    @patch("app.api.v1.knowledge._upload_query_image_if_needed")
    @patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session")
    async def test_llm_operation_decision_retrieves_workflow_before_clarification(
        self,
        mock_ensure_session,
        mock_upload_image,
        mock_persist_clarification,
        mock_llm_decision,
        mock_retrieve_workflow,
        mock_extract_workflow,
        mock_invoke_rag,
    ):
        mock_ensure_session.return_value = "session-1"
        mock_upload_image.return_value = (None, None)
        workflow_sources = [
            {
                "id": "chunk-1",
                "job_id": "job-1",
                "file_name": "\u5bcc\u53cb\u6d41\u7a0b.docx",
                "content": "\u5bcc\u53cb\u5904\u7406\u9700\u8981\u624b\u673a\u53f7\u548c\u622a\u56fe\u3002",
                "score": 0.9,
            }
        ]
        mock_retrieve_workflow.return_value = {
            "sources": workflow_sources,
            "thoughts": {"retrieval": {"chunks_used": 1}},
        }
        mock_extract_workflow.return_value = {
            "workflow_found": True,
            "workflow_summary": "\u77e5\u8bc6\u5e93\u6d41\u7a0b\uff1a\u5bcc\u53cb\u5904\u7406\u9700\u8981\u624b\u673a\u53f7\u548c\u622a\u56fe\u3002",
            "required_fields": ["phone", "image"],
            "field_labels": {"phone": "\u53ef\u8054\u7cfb\u624b\u673a\u53f7", "image": "\u5bcc\u53cb\u9875\u9762\u622a\u56fe"},
            "fallback_used": False,
        }
        mock_persist_clarification.side_effect = [
            None,
            {
                "ticket_id": "ticket-1",
                "answer": "\u6211\u67e5\u5230\u76f8\u5173\u6d41\u7a0b\uff1a\u5bcc\u53cb\u5904\u7406\u9700\u8981\u624b\u673a\u53f7\u548c\u622a\u56fe\u3002\u8bf7\u8865\u5145\u53ef\u8054\u7cfb\u624b\u673a\u53f7\u548c\u5bcc\u53cb\u9875\u9762\u622a\u56fe\u3002",
                "finish_reason": "clarification",
                "clarification": {"reason": "operation_required", "workflow_found": True},
            },
        ]
        mock_llm_decision.return_value = {
            "should_clarify": True,
            "reason": "operation_required",
            "message": "\u9700\u8981\u5148\u8865\u9f50\u5de5\u5355\u4fe1\u606f",
            "source": "llm",
        }

        response = await knowledge.knowledge_qa(
            KnowledgeRequest(query=OP_QUERY, session_id="default", collection="fushang"),
            identity={"user_id": "store-1", "user_name": "\u5f20\u5e97\u957f", "channel": "h5"},
        )

        self.assertEqual(response.finish_reason, "clarification")
        self.assertEqual(response.sources, [])
        self.assertIn("\u6d41\u7a0b", response.answer)
        mock_llm_decision.assert_called_once_with(OP_QUERY, has_image=False)
        mock_retrieve_workflow.assert_awaited_once()
        mock_extract_workflow.assert_called_once_with(OP_QUERY, workflow_sources)
        self.assertEqual(mock_persist_clarification.call_count, 2)
        self.assertTrue(mock_persist_clarification.call_args.kwargs["force_start"])
        self.assertEqual(mock_persist_clarification.call_args.kwargs["workflow"]["required_fields"], ["phone", "image"])
        self.assertEqual(mock_persist_clarification.call_args.kwargs["workflow_sources"], workflow_sources)
        self.assertTrue(response.thoughts["operation_workflow_found"])
        self.assertEqual(response.thoughts["operation_required_fields"], ["phone", "image"])
        mock_invoke_rag.assert_not_awaited()

    @patch("app.api.v1.knowledge.invoke_knowledge_qa", new_callable=AsyncMock)
    @patch("app.api.v1.knowledge.extract_operation_workflow_requirements")
    @patch("app.api.v1.knowledge.retrieve_operation_workflow", new_callable=AsyncMock)
    @patch("app.api.v1.knowledge.classify_operation_query_with_llm")
    @patch("app.api.v1.knowledge.persist_clarification_message")
    @patch("app.api.v1.knowledge._upload_query_image_if_needed")
    @patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session")
    async def test_operation_decision_falls_back_when_no_workflow_found(
        self,
        mock_ensure_session,
        mock_upload_image,
        mock_persist_clarification,
        mock_llm_decision,
        mock_retrieve_workflow,
        mock_extract_workflow,
        mock_invoke_rag,
    ):
        mock_ensure_session.return_value = "session-1"
        mock_upload_image.return_value = (None, None)
        mock_persist_clarification.side_effect = [
            None,
            {
                "ticket_id": "ticket-1",
                "answer": "\u6211\u5148\u5e2e\u4f60\u628a\u8fd9\u4e2a\u95ee\u9898\u6574\u7406\u6210\u540e\u53f0\u5de5\u5355\u3002\u8bf7\u8865\u5145\u8d26\u53f7/\u95e8\u5e97\u3001\u624b\u673a\u53f7\u548c\u622a\u56fe\u3002",
                "finish_reason": "clarification",
                "clarification": {"fallback_used": True},
            },
        ]
        mock_llm_decision.return_value = {
            "should_clarify": True,
            "reason": "operation_required",
            "message": "\u9700\u8981\u5148\u8865\u9f50\u5de5\u5355\u4fe1\u606f",
            "source": "llm",
        }
        mock_retrieve_workflow.return_value = {"sources": [], "thoughts": {}}
        mock_extract_workflow.return_value = {
            "workflow_found": False,
            "workflow_summary": "",
            "required_fields": [],
            "field_labels": {},
            "fallback_used": True,
        }

        response = await knowledge.knowledge_qa(
            KnowledgeRequest(query=OP_QUERY, session_id="default", collection="fushang"),
            identity={"user_id": "store-1", "user_name": "\u5f20\u5e97\u957f", "channel": "h5"},
        )

        self.assertEqual(response.finish_reason, "clarification")
        self.assertFalse(response.thoughts["operation_workflow_found"])
        self.assertEqual(response.thoughts["operation_required_fields"], [])
        self.assertTrue(mock_persist_clarification.call_args.kwargs["workflow"]["fallback_used"])
        mock_invoke_rag.assert_not_awaited()

    @patch("app.services.knowledge_service._persist_conversation_messages")
    @patch("app.api.v1.knowledge.invoke_knowledge_qa", new_callable=AsyncMock)
    @patch("app.api.v1.knowledge.classify_operation_query_with_llm")
    @patch("app.api.v1.knowledge.persist_clarification_message")
    @patch("app.api.v1.knowledge._upload_query_image_if_needed")
    @patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session")
    async def test_llm_knowledge_decision_allows_rag(
        self,
        mock_ensure_session,
        mock_upload_image,
        mock_persist_clarification,
        mock_llm_decision,
        mock_invoke_rag,
        mock_persist_messages,
    ):
        mock_ensure_session.return_value = "session-1"
        mock_upload_image.return_value = (None, None)
        mock_persist_clarification.return_value = None
        mock_llm_decision.return_value = {
            "should_clarify": False,
            "reason": "knowledge_question",
            "message": "",
            "source": "llm",
        }
        mock_invoke_rag.return_value = {
            "request_id": "request-1",
            "session_id": "session-1",
            "answer": "\u8fd9\u91cc\u662f\u7ed1\u5b9a\u6559\u7a0b\u3002",
            "confidence": 0.82,
            "sources": [{"id": "chunk-1"}],
            "model": "doubao-seed-2-0-pro-260215",
            "thoughts": {},
            "image_map": None,
            "finish_reason": "stop",
        }

        response = await knowledge.knowledge_qa(
            KnowledgeRequest(query=HOW_TO_QUERY, session_id="default", collection="fushang"),
            identity={"user_id": "store-1", "user_name": "\u5f20\u5e97\u957f", "channel": "h5"},
        )

        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.answer, "\u8fd9\u91cc\u662f\u7ed1\u5b9a\u6559\u7a0b\u3002")
        mock_llm_decision.assert_called_once_with(HOW_TO_QUERY, has_image=False)
        mock_invoke_rag.assert_awaited_once()
        self.assertFalse(mock_invoke_rag.call_args.kwargs["persist"])
        mock_persist_messages.assert_called_once()

    @patch("app.api.v1.knowledge.start_missing_knowledge_clarification")
    @patch("app.api.v1.knowledge.invoke_knowledge_qa", new_callable=AsyncMock)
    @patch("app.api.v1.knowledge.classify_operation_query_with_llm")
    @patch("app.api.v1.knowledge.persist_clarification_message")
    @patch("app.api.v1.knowledge._upload_query_image_if_needed")
    @patch("app.api.v1.knowledge.conversation_service.ensure_knowledge_session")
    async def test_rag_fallback_starts_missing_knowledge_clarification(
        self,
        mock_ensure_session,
        mock_upload_image,
        mock_persist_clarification,
        mock_llm_decision,
        mock_invoke_rag,
        mock_start_missing,
    ):
        mock_ensure_session.return_value = "session-1"
        mock_upload_image.return_value = (None, None)
        mock_persist_clarification.return_value = None
        mock_llm_decision.return_value = {
            "should_clarify": False,
            "reason": "knowledge_question",
            "message": "",
            "source": "llm",
        }
        mock_invoke_rag.return_value = {
            "request_id": "request-1",
            "session_id": "session-1",
            "answer": "知识库未包含该问题的答案。",
            "confidence": 0.2,
            "sources": [],
            "model": "doubao-seed-2-0-pro-260215",
            "thoughts": {"manual_review_recommended": True},
            "image_map": None,
            "finish_reason": "stop",
            "used_fallback": True,
            "fallback_reason": "no_relevant_knowledge",
            "quality_passed": False,
            "quality_level": "low",
        }
        mock_start_missing.return_value = {
            "ticket_id": "ticket-1",
            "answer": "请补充具体场景。",
            "finish_reason": "clarification",
            "clarification": {"intent_class": "B", "reason": "knowledge_missing"},
        }

        response = await knowledge.knowledge_qa(
            KnowledgeRequest(query="知识库没有的新问题", session_id="default", collection="fushang"),
            identity={"user_id": "store-1", "user_name": "张店长", "channel": "h5"},
        )

        self.assertEqual(response.finish_reason, "clarification")
        self.assertEqual(response.answer, "请补充具体场景。")
        self.assertEqual(response.thoughts["clarification"]["intent_class"], "B")
        mock_invoke_rag.assert_awaited_once()
        self.assertFalse(mock_invoke_rag.call_args.kwargs["persist"])
        mock_start_missing.assert_called_once()


if __name__ == "__main__":
    unittest.main()
