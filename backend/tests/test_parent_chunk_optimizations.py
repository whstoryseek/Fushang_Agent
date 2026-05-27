# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.knowledge.nodes.generate import _dedupe_chunks_by_parent
from agents.knowledge.nodes.rerank import _collapse_by_parent_candidates
from app.db.chunk_repository import ChunkRepository


class ParentChunkOptimizationTests(unittest.TestCase):
    def test_chunk_repository_extracts_parent_records_without_leaving_parent_content_in_metadata(self):
        cleaned_chunks, parent_records = ChunkRepository._split_parent_payloads(
            job_id="job-1",
            chunks=[
                {
                    "content": "child a",
                    "metadata": {
                        "parent_id": "job-1-parent-0",
                        "parent_index": 0,
                        "parent_content": "parent block text",
                    },
                },
                {
                    "content": "child b",
                    "metadata": {
                        "parent_id": "job-1-parent-0",
                        "parent_index": 0,
                        "parent_content": "parent block text",
                    },
                },
            ],
        )

        self.assertEqual(len(parent_records), 1)
        self.assertEqual(parent_records[0][:4], ("job-1-parent-0", "job-1", 0, "parent block text"))
        self.assertNotIn("parent_content", cleaned_chunks[0]["metadata"])

    def test_collapse_by_parent_candidates_keeps_best_child_per_parent(self):
        collapsed = _collapse_by_parent_candidates([
            {
                "chunk_id": "chunk-a",
                "score": 0.7,
                "content": "child a",
                "metadata": {"parent_id": "parent-1"},
            },
            {
                "chunk_id": "chunk-b",
                "score": 0.9,
                "content": "child b",
                "metadata": {"parent_id": "parent-1"},
            },
            {
                "chunk_id": "chunk-c",
                "score": 0.8,
                "content": "child c",
                "metadata": {"parent_id": "parent-2"},
            },
        ])

        self.assertEqual(len(collapsed), 2)
        self.assertEqual(collapsed[0]["chunk_id"], "chunk-b")
        self.assertEqual(collapsed[0]["metadata"]["parent_hit_count"], 2)
        self.assertEqual(collapsed[0]["metadata"]["representative_child_id"], "chunk-b")

    def test_generation_dedupes_context_by_parent_id(self):
        deduped = _dedupe_chunks_by_parent([
            {
                "chunk_id": "chunk-a",
                "content": "child a",
                "metadata": {"parent_id": "parent-1", "parent_content": "same parent"},
            },
            {
                "chunk_id": "chunk-b",
                "content": "child b",
                "metadata": {"parent_id": "parent-1", "parent_content": "same parent"},
            },
            {
                "chunk_id": "chunk-c",
                "content": "child c",
                "metadata": {"parent_id": "parent-2", "parent_content": "other parent"},
            },
        ])

        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["metadata"]["parent_id"], "parent-1")
        self.assertEqual(deduped[1]["metadata"]["parent_id"], "parent-2")


if __name__ == "__main__":
    unittest.main()
