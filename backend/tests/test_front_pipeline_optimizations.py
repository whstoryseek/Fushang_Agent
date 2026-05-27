import asyncio
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("PG_HOST", "localhost")
os.environ.setdefault("PG_USER", "tester")
os.environ.setdefault("PG_PASSWORD", "tester")
os.environ.setdefault("MILVUS_HOST", "localhost")
os.environ.setdefault("VOLCES_API_KEY", "test-key")

if "dashscope" not in sys.modules:
    dashscope_module = types.ModuleType("dashscope")

    class _TextEmbedding:
        @staticmethod
        def call(*args, **kwargs):
            raise NotImplementedError

    dashscope_module.TextEmbedding = _TextEmbedding
    dashscope_module.api_key = ""
    sys.modules["dashscope"] = dashscope_module

if "pymilvus" not in sys.modules:
    pymilvus_module = types.ModuleType("pymilvus")

    class _MilvusClient:
        @staticmethod
        def create_schema(*args, **kwargs):
            raise NotImplementedError

        def __init__(self, *args, **kwargs):
            pass

    pymilvus_module.MilvusClient = _MilvusClient
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

from langchain_core.messages import AIMessage, HumanMessage

from agents.knowledge import graph as graph_module
from agents.knowledge.nodes.kg_query_route import kg_query_route
from agents.knowledge.nodes.query_classify import query_classify
from agents.knowledge.nodes.query_rewrite import query_rewrite
from agents.knowledge.nodes.relevance_filter import relevance_filter
from agents.knowledge.state import RAGConfig, RetrievalStrategy, create_initial_state
from app.core.config import settings
from app.services.knowledge_service import invoke_knowledge_qa


class FrontPipelineNodeTests(unittest.TestCase):
    def _make_state(self, query: str, *, messages=None, config: RAGConfig | None = None):
        return create_initial_state(
            query=query,
            user_id="tester",
            session_id="session-1",
            messages=messages,
            config=config or RAGConfig(model="doubao-seed-2-0-pro-260215"),
        )

    @patch("agents.knowledge.nodes.query_rewrite.get_llm_service")
    def test_query_rewrite_fast_skips_without_history_or_context_terms(self, mock_get_llm_service):
        query = "请总结这份制度的核心要求"
        state = self._make_state(query)

        result = query_rewrite(state)

        self.assertEqual(result["rewritten_query"], query)
        self.assertEqual(result["processing_log"][0]["mode"], "fast_skip")
        mock_get_llm_service.assert_not_called()

    @patch("agents.knowledge.nodes.query_rewrite.get_llm_service")
    def test_query_rewrite_fast_skips_for_kb_scope_query_without_history(self, mock_get_llm_service):
        query = "\u8fd9\u4e2a\u77e5\u8bc6\u5e93\u91cc\u6709\u4ec0\u4e48\u5185\u5bb9\uff1f\u8bf7\u7b80\u8981\u8bf4\u660e\u3002"
        state = self._make_state(query)

        result = query_rewrite(state)

        self.assertEqual(result["rewritten_query"], query)
        self.assertEqual(result["processing_log"][0]["mode"], "fast_skip")
        self.assertEqual(result["processing_log"][0]["reason"], "kb_scope_query")
        mock_get_llm_service.assert_not_called()

    @patch("agents.knowledge.nodes.query_rewrite.get_llm_service")
    def test_query_rewrite_uses_lite_model_when_history_exists(self, mock_get_llm_service):
        llm = MagicMock()
        llm.chat.return_value = "改写后的问题"
        mock_get_llm_service.return_value = llm
        state = self._make_state(
            "这个要怎么配置",
            messages=[
                HumanMessage(content="帮我看一下索引参数"),
                AIMessage(content="可以，主要看 HNSW 和 PQ。"),
                HumanMessage(content="这个要怎么配置"),
            ],
            config=RAGConfig(memory_turns=2, model="doubao-seed-2-0-pro-260215"),
        )

        result = query_rewrite(state)

        self.assertEqual(result["rewritten_query"], "改写后的问题")
        self.assertEqual(result["processing_log"][0]["mode"], "llm_rewrite")
        llm.chat.assert_called_once()
        self.assertEqual(llm.chat.call_args.kwargs["model"], settings.llm_clean_model)
        self.assertEqual(llm.chat.call_args.kwargs["max_tokens"], 64)
        self.assertEqual(llm.chat.call_args.kwargs["timeout"], 5.0)
        self.assertEqual(llm.chat.call_args.kwargs["max_retries"], 0)

    @patch("agents.knowledge.nodes.query_rewrite.get_llm_service")
    def test_query_rewrite_uses_lite_model_for_context_dependent_query(self, mock_get_llm_service):
        llm = MagicMock()
        llm.chat.return_value = "和上一段内容相关的完整问题"
        mock_get_llm_service.return_value = llm
        state = self._make_state("这个是什么意思")

        query_rewrite(state)

        llm.chat.assert_called_once()
        self.assertEqual(llm.chat.call_args.kwargs["model"], settings.llm_clean_model)
        self.assertEqual(llm.chat.call_args.kwargs["max_tokens"], 64)
        self.assertEqual(llm.chat.call_args.kwargs["timeout"], 5.0)
        self.assertEqual(llm.chat.call_args.kwargs["max_retries"], 0)

    @patch("agents.knowledge.nodes.query_rewrite.get_llm_service")
    def test_query_rewrite_falls_back_to_original_query_on_blank_output(self, mock_get_llm_service):
        llm = MagicMock()
        llm.chat.return_value = " "
        mock_get_llm_service.return_value = llm
        state = self._make_state("这个怎么处理")

        result = query_rewrite(state)

        self.assertEqual(result["rewritten_query"], "这个怎么处理")

    @patch("agents.knowledge.nodes.query_classify.get_llm_service")
    def test_query_classify_uses_clean_model_with_tight_max_tokens(self, mock_get_llm_service):
        llm = MagicMock()
        llm.chat.return_value = "single_doc"
        mock_get_llm_service.return_value = llm
        state = self._make_state("执行周期")
        state["rewritten_query"] = "执行周期"
        state["config"].model = "doubao-seed-2-0-pro-260215"

        result = query_classify(state)

        self.assertEqual(result["query_type"], "single_doc")
        self.assertEqual(result["processing_log"][0]["source"], "llm")
        llm.chat.assert_called_once()
        self.assertEqual(llm.chat.call_args.kwargs["model"], settings.llm_clean_model)
        self.assertEqual(llm.chat.call_args.kwargs["max_tokens"], 8)
        self.assertEqual(llm.chat.call_args.kwargs["timeout"], 5.0)
        self.assertEqual(llm.chat.call_args.kwargs["max_retries"], 0)

    @patch("agents.knowledge.nodes.query_classify.get_llm_service")
    def test_query_classify_accepts_chinese_output_labels(self, mock_get_llm_service):
        llm = MagicMock()
        llm.chat.return_value = "单文档"
        mock_get_llm_service.return_value = llm
        state = self._make_state("执行周期")
        state["rewritten_query"] = "执行周期"

        result = query_classify(state)

        self.assertEqual(result["query_type"], "single_doc")
        messages = llm.chat.call_args.kwargs["messages"]
        self.assertIn("只返回：单文档 或 多文档", messages[0]["content"])

    @patch("agents.knowledge.nodes.query_classify.get_llm_service")
    def test_query_classify_falls_back_to_multi_doc_when_llm_fails(self, mock_get_llm_service):
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("boom")
        mock_get_llm_service.return_value = llm
        state = self._make_state("plain query")
        state["rewritten_query"] = "plain query"

        result = query_classify(state)

        self.assertEqual(result["query_type"], "multi_doc")
        self.assertEqual(result["processing_log"][0]["source"], "llm")
        llm.chat.assert_called_once()

    @patch("agents.knowledge.nodes.query_classify.get_llm_service")
    def test_query_classify_force_multi_doc_skips_llm_and_logs_reason(self, mock_get_llm_service):
        state = self._make_state("any query", config=RAGConfig(force_multi_doc=True))
        state["rewritten_query"] = "any query"

        result = query_classify(state)

        self.assertEqual(result["query_type"], "multi_doc")
        self.assertEqual(result["processing_log"][0]["reason"], "force_multi_doc")
        mock_get_llm_service.assert_not_called()

    @patch("agents.knowledge.nodes.query_classify.get_llm_service")
    def test_query_classify_uses_rule_for_multi_doc_keywords(self, mock_get_llm_service):
        query = "\u8bf7\u603b\u7ed3\u6240\u6709\u5236\u5ea6\u7684\u5dee\u5f02"
        state = self._make_state(query)
        state["rewritten_query"] = query

        result = query_classify(state)

        self.assertEqual(result["query_type"], "multi_doc")
        self.assertEqual(result["processing_log"][0]["source"], "rule")
        self.assertEqual(result["processing_log"][0]["rule_hit"], "multi_doc_keyword")
        mock_get_llm_service.assert_not_called()

    @patch("agents.knowledge.nodes.query_classify.get_llm_service")
    def test_query_classify_uses_rule_for_collection_summary_query(self, mock_get_llm_service):
        query = "\u8fd9\u4e2a\u77e5\u8bc6\u5e93\u91cc\u6709\u4ec0\u4e48\u5185\u5bb9\uff1f\u8bf7\u7b80\u8981\u8bf4\u660e\u3002"
        state = self._make_state(query)
        state["rewritten_query"] = query

        result = query_classify(state)

        self.assertEqual(result["query_type"], "multi_doc")
        self.assertEqual(result["processing_log"][0]["source"], "rule")
        self.assertEqual(result["processing_log"][0]["rule_hit"], "multi_doc_collection_summary")
        mock_get_llm_service.assert_not_called()

    @patch("agents.knowledge.nodes.query_classify.get_llm_service")
    def test_query_classify_uses_rule_for_single_doc_reference(self, mock_get_llm_service):
        query = "\u8fd9\u4e2a\u6587\u6863\u8bb2\u4e86\u4ec0\u4e48"
        state = self._make_state(query)
        state["rewritten_query"] = query

        result = query_classify(state)

        self.assertEqual(result["query_type"], "single_doc")
        self.assertEqual(result["processing_log"][0]["source"], "rule")
        self.assertEqual(result["processing_log"][0]["rule_hit"], "single_doc_reference")
        mock_get_llm_service.assert_not_called()

    @patch("agents.knowledge.nodes.query_classify.get_llm_service")
    def test_query_classify_uses_rule_for_filename_pattern(self, mock_get_llm_service):
        query = "policy.pdf \u8bb2\u4e86\u4ec0\u4e48"
        state = self._make_state(query)
        state["rewritten_query"] = query

        result = query_classify(state)

        self.assertEqual(result["query_type"], "single_doc")
        self.assertEqual(result["processing_log"][0]["source"], "rule")
        self.assertEqual(result["processing_log"][0]["rule_hit"], "single_doc_filename")
        mock_get_llm_service.assert_not_called()

    @patch("agents.knowledge.nodes.query_classify.get_llm_service")
    def test_query_classify_uses_rule_for_english_single_doc_reference(self, mock_get_llm_service):
        query = "What does this document say about onboarding?"
        state = self._make_state(query)
        state["rewritten_query"] = query

        result = query_classify(state)

        self.assertEqual(result["query_type"], "single_doc")
        self.assertEqual(result["processing_log"][0]["source"], "rule")
        self.assertEqual(result["processing_log"][0]["rule_hit"], "single_doc_reference")
        mock_get_llm_service.assert_not_called()

    @patch("agents.knowledge.nodes.query_classify.get_llm_service")
    def test_query_classify_uses_rule_for_lookup_style_multi_doc_query(self, mock_get_llm_service):
        query = "Find the relevant policy for onboarding."
        state = self._make_state(query)
        state["rewritten_query"] = query

        result = query_classify(state)

        self.assertEqual(result["query_type"], "multi_doc")
        self.assertEqual(result["processing_log"][0]["source"], "rule")
        self.assertEqual(result["processing_log"][0]["rule_hit"], "multi_doc_lookup_query")
        mock_get_llm_service.assert_not_called()

    @patch("agents.knowledge.nodes.query_classify.get_llm_service")
    def test_query_classify_uses_rule_for_topic_style_multi_doc_query(self, mock_get_llm_service):
        query = "制度适用范围"
        state = self._make_state(query)
        state["rewritten_query"] = query

        result = query_classify(state)

        self.assertEqual(result["query_type"], "multi_doc")
        self.assertEqual(result["processing_log"][0]["source"], "rule")
        self.assertEqual(result["processing_log"][0]["rule_hit"], "multi_doc_topic_query")
        mock_get_llm_service.assert_not_called()

    @patch("agents.knowledge.nodes.kg_query_route.get_llm_service")
    def test_kg_query_route_uses_clean_model_with_tight_max_tokens(self, mock_get_llm_service):
        llm = MagicMock()
        llm.chat.return_value = "yes"
        mock_get_llm_service.return_value = llm
        state = self._make_state("请继续深挖相关依赖")
        state["rewritten_query"] = "请继续深挖相关依赖"
        state["config"].kg_enabled = True

        result = kg_query_route(state)

        self.assertTrue(result["kg_deep_traversal"])
        llm.chat.assert_called_once()
        self.assertEqual(llm.chat.call_args.kwargs["model"], settings.llm_clean_model)
        self.assertEqual(llm.chat.call_args.kwargs["max_tokens"], 8)
        self.assertEqual(llm.chat.call_args.kwargs["timeout"], 5.0)
        self.assertEqual(llm.chat.call_args.kwargs["max_retries"], 0)

    @patch("agents.knowledge.nodes.relevance_filter.get_llm_service")
    def test_relevance_filter_accepts_chinese_relevance_labels(self, mock_get_llm_service):
        llm = MagicMock()
        llm.chat.return_value = "1|相关\n2|不相关"
        mock_get_llm_service.return_value = llm
        state = self._make_state("工资制度")
        state["rewritten_query"] = "工资制度"
        state["merged_chunks"] = [
            {"id": "chunk-1", "content": "工资制度说明"},
            {"id": "chunk-2", "content": "差旅报销说明"},
        ]

        result = relevance_filter(state)

        self.assertEqual([chunk["id"] for chunk in result["filtered_chunks"]], ["chunk-1"])


class GraphRoutingTests(unittest.TestCase):
    def _build_graph_with_spies(self):
        def query_rewrite_node(state):
            return {
                "rewritten_query": state["original_query"],
                "processing_log": [{"stage": "query_rewrite"}],
            }

        def query_classify_node(_state):
            return {"query_type": "single_doc", "processing_log": [{"stage": "query_classify"}]}

        def retrieval_strategy_node(_state):
            return {
                "retrieval_strategy": RetrievalStrategy.HYBRID,
                "processing_log": [{"stage": "retrieval_strategy"}],
            }

        def kg_query_route_node(_state):
            return {"kg_deep_traversal": False, "processing_log": [{"stage": "kg_query_route"}]}

        def graph_retrieve_node(_state):
            return {"kg_graph_chunks": [], "processing_log": [{"stage": "graph_retrieve"}]}

        def single_doc_retrieve_node(_state):
            return {"processing_log": [{"stage": "single_doc_retrieve"}]}

        def multi_doc_retrieve_node(_state):
            return {"processing_log": [{"stage": "multi_doc_retrieve"}]}

        def filter_chunks_node(_state):
            return {"processing_log": [{"stage": "filter_chunks"}]}

        def select_top_k_chunks_node(_state):
            return {"processing_log": [{"stage": "select_top_k_chunks"}]}

        def generate_answer_node(_state):
            return {
                "answer": "ok",
                "confidence": 0.9,
                "sources": [],
                "processing_log": [{"stage": "generate_answer"}],
            }

        def check_quality_node(_state):
            return {"processing_log": [{"stage": "check_quality"}]}

        def finalize_metrics_node(_state):
            return {"processing_log": [{"stage": "finalize_metrics"}]}

        patches = patch.multiple(
            graph_module,
            query_rewrite=query_rewrite_node,
            query_classify=query_classify_node,
            determine_retrieval_strategy=retrieval_strategy_node,
            kg_query_route=kg_query_route_node,
            graph_retrieve=graph_retrieve_node,
            single_doc_retrieve=single_doc_retrieve_node,
            multi_doc_retrieve=multi_doc_retrieve_node,
            filter_chunks=filter_chunks_node,
            select_top_k_chunks=select_top_k_chunks_node,
            generate_answer=generate_answer_node,
            check_quality=check_quality_node,
            finalize_metrics=finalize_metrics_node,
        )
        return patches

    def test_graph_skips_kg_branch_when_disabled(self):
        with self._build_graph_with_spies():
            graph = graph_module.create_knowledge_agent(checkpointer=None)
            result = graph.invoke(
                create_initial_state(
                    query="查制度",
                    user_id="tester",
                    session_id="session-1",
                    config=RAGConfig(
                        model="doubao-seed-2-0-pro-260215",
                        kg_enabled=False,
                        enable_fallback=False,
                    ),
                )
            )

        stages = [entry["stage"] for entry in result["processing_log"]]
        self.assertNotIn("kg_query_route", stages)
        self.assertNotIn("graph_retrieve", stages)

    def test_graph_enters_kg_branch_when_enabled(self):
        with self._build_graph_with_spies():
            graph = graph_module.create_knowledge_agent(checkpointer=None)
            result = graph.invoke(
                create_initial_state(
                    query="查制度",
                    user_id="tester",
                    session_id="session-1",
                    config=RAGConfig(
                        model="doubao-seed-2-0-pro-260215",
                        kg_enabled=True,
                        enable_fallback=False,
                    ),
                )
            )

        stages = [entry["stage"] for entry in result["processing_log"]]
        self.assertIn("kg_query_route", stages)
        self.assertIn("graph_retrieve", stages)


class KnowledgeServiceConfigTests(unittest.TestCase):
    def test_invoke_knowledge_qa_defaults_kg_disabled_when_no_graph_signal(self):
        captured = {}

        class FakeAgent:
            async def ainvoke(self, _initial_state, config=None):
                return {
                    "answer": "ok",
                    "confidence": 0.8,
                    "sources": [],
                    "metrics": MagicMock(total_chunks_retrieved=0, chunks_after_rerank=0),
                }

        def fake_create_initial_state(*, config, **kwargs):
            captured["config"] = config
            return {"query": kwargs["query"]}

        with patch("agents.knowledge.get_knowledge_agent", return_value=FakeAgent()), patch(
            "agents.knowledge.create_initial_state",
            side_effect=fake_create_initial_state,
        ), patch(
            "app.services.knowledge_service._load_kb_retrieval",
            return_value=({"id": "kb-1", "name": "kb-1", "kb_type": "standard"}, {}),
        ), patch("app.services.knowledge_service._persist_conversation_messages"):
            asyncio.run(
                invoke_knowledge_qa(
                    query="帮我查制度",
                    model_name="doubao-seed-2-0-pro-260215",
                    session_id="session-1",
                    collection="kb-1",
                )
            )

        self.assertFalse(captured["config"].kg_enabled)


if __name__ == "__main__":
    unittest.main()
