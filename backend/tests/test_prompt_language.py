# -*- coding: utf-8 -*-
import sys
import types
import unittest

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

from app.core import prompts
from agents.supervisor.services.coordinator import format_agents_info


class PromptLanguageTests(unittest.TestCase):
    def test_supervisor_prompt_and_agent_info_are_chinese(self):
        supervisor_prompt = prompts.SUPERVISOR_SYSTEM_PROMPT
        agent_info = format_agents_info()

        self.assertIn("你是", supervisor_prompt)
        self.assertIn("协调", supervisor_prompt)
        self.assertNotIn("You are a Supervisor Agent", supervisor_prompt)
        self.assertNotIn("Your role", supervisor_prompt)
        self.assertNotIn("Guidelines", supervisor_prompt)

        self.assertIn("可用专用智能体", agent_info)
        self.assertNotIn("Available Specialized Agents", agent_info)
        self.assertNotIn("Description:", agent_info)
        self.assertNotIn("Capabilities:", agent_info)
        self.assertNotIn("Keywords:", agent_info)

    def test_rag_control_prompts_use_chinese_output_labels(self):
        self.assertIn("只输出 是 或 否", prompts.KNOWLEDGE_KG_DEEP_ROUTE_SYSTEM)
        self.assertNotIn("yes 或 no", prompts.KNOWLEDGE_KG_DEEP_ROUTE_SYSTEM)
        self.assertNotIn("yes or no", prompts.KNOWLEDGE_KG_DEEP_ROUTE_SYSTEM)

        self.assertIn("只返回：单文档 或 多文档", prompts.KNOWLEDGE_QUERY_CLASSIFY_SYSTEM)
        self.assertNotIn("single_doc 或 multi_doc", prompts.KNOWLEDGE_QUERY_CLASSIFY_SYSTEM)

        self.assertIn("切片编号|相关", prompts.KNOWLEDGE_RELEVANCE_FILTER_SYSTEM)
        self.assertIn("切片编号|不相关", prompts.KNOWLEDGE_RELEVANCE_FILTER_SYSTEM)
        self.assertNotIn("relevant", prompts.KNOWLEDGE_RELEVANCE_FILTER_SYSTEM)
        self.assertNotIn("irrelevant", prompts.KNOWLEDGE_RELEVANCE_FILTER_SYSTEM)
