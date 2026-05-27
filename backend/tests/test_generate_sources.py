# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.knowledge.nodes.generate import _chunk_to_source_block, build_sources_from_reranked


class GenerateSourcesTests(unittest.TestCase):
    def test_sources_include_editable_chunk_references(self):
        sources = build_sources_from_reranked([
            {
                "chunk_id": "chunk-1",
                "job_id": "job-1",
                "file_name": "faq.docx",
                "chunk_index": 3,
                "content": "上下文内容",
                "score": 0.88,
                "metadata": {"page": 2},
            }
        ])

        self.assertEqual(sources[0]["id"], "chunk-1")
        self.assertEqual(sources[0]["chunk_id"], "chunk-1")
        self.assertEqual(sources[0]["job_id"], "job-1")
        self.assertEqual(sources[0]["chunk_index"], 3)

    def test_source_block_uses_parent_context_when_available(self):
        block = _chunk_to_source_block(
            {
                "chunk_id": "chunk-1",
                "file_name": "guide.txt",
                "content": "子块内容",
                "metadata": {"parent_content": "父块完整上下文"},
            },
            0,
        )

        self.assertIn("父块完整上下文", block)


if __name__ == "__main__":
    unittest.main()
