# -*- coding: utf-8 -*-
import sys
import types
import unittest

if "dashscope" not in sys.modules:
    dashscope_module = types.ModuleType("dashscope")
    dashscope_module.TextEmbedding = types.SimpleNamespace(call=lambda *args, **kwargs: None)
    dashscope_module.api_key = ""
    sys.modules["dashscope"] = dashscope_module

from agents.knowledge.nodes.quality_check import check_quality
from agents.knowledge.state import RAGConfig


class QualityCheckUnansweredTest(unittest.TestCase):
    def test_knowledge_base_refusal_is_marked_as_fallback(self):
        result = check_quality({
            "answer": '当前知识库未包含与"AI是什么"相关的内容，无法为您解答该问题。',
            "confidence": 0.61,
            "config": RAGConfig(enable_fallback=True),
        })

        self.assertTrue(result["used_fallback"])
        self.assertFalse(result["quality_passed"])
        self.assertIn("knowledge base", result["fallback_reason"].lower())
        self.assertIn("无法为您解答", result["answer"])


if __name__ == "__main__":
    unittest.main()
