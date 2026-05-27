# Parent Child Multimodal Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement unified parent-child chunking for standard and image-text document ingestion while preserving backward compatibility and image placeholder behavior.

**Architecture:** Add parent-child splitting primitives in `chunk_splitter.py`, route text and image-text parsers through them, and pass new API/frontend fields through the upload pipeline. Store child chunks as today and put parent context in chunk metadata so retrieval and generation can expand context without a schema migration.

**Tech Stack:** Python FastAPI backend, PostgreSQL chunk metadata JSONB, Milvus retrieval, Vue 3 + Element Plus frontend, Python unittest and Node test runner.

---

## File Structure

- Modify `backend/app/services/chunk_splitter.py`: add parent-child split functions, config validation, and Excel strategy metadata.
- Modify `backend/app/services/doc_image_parser.py`: convert PDF/DOCX image-text streams into parent-child chunks and rebind image records to final child chunk ids.
- Modify `backend/app/services/job_service.py`: accept new chunk config fields and pass them into text/image parsing.
- Modify `backend/app/services/document_service.py`: accept and forward new chunk config fields.
- Modify `backend/app/api/v1/documents.py`: expose new form/query params with old-field compatibility.
- Modify `backend/app/services/milvus_service.py`: merge PG metadata during retrieval backfill.
- Modify `backend/agents/knowledge/nodes/generate.py`: use `metadata.parent_content` when building model context.
- Modify `backend/tests/test_docx_pipeline.py`: add backend coverage for parent-child text and image-mode behavior.
- Modify `frontend/src/components/doc/DocUpload.vue`: replace simple chunk controls with strategy, preset, parent/child size controls, and submit new fields.

## Tasks

### Task 1: Backend Parent-Child Splitter

**Files:**
- Modify: `backend/app/services/chunk_splitter.py`
- Test: `backend/tests/test_docx_pipeline.py`

- [ ] **Step 1: Write failing splitter tests**

Add tests that call `split_parent_child_text` and `split_text_with_metadata`:

```python
from app.services.chunk_splitter import split_parent_child_text, split_text_with_metadata

def test_parent_child_text_chunks_keep_parent_context(self):
    text = "第一段说明开店流程。" * 80 + "\n\n" + "第二段说明人员入企。" * 80
    chunks = split_parent_child_text(
        text,
        parent_chunk_size=300,
        child_chunk_size=120,
        chunk_overlap=20,
        base_metadata={"file_name": "guide.txt", "source": "txt"},
        parent_id_prefix="job-a",
    )
    self.assertGreater(len(chunks), 2)
    self.assertEqual(chunks[0]["metadata"]["chunk_strategy"], "parent_child")
    self.assertEqual(chunks[0]["metadata"]["parent_id"], "job-a-parent-0")
    self.assertIn("parent_content", chunks[0]["metadata"])
    self.assertLessEqual(len(chunks[0]["content"]), 180)

def test_flat_text_strategy_remains_available(self):
    chunks = split_text_with_metadata(
        "A" * 260,
        chunk_size=100,
        chunk_overlap=0,
        base_metadata={"source": "txt"},
        chunk_strategy="flat",
    )
    self.assertEqual(chunks[0]["metadata"]["chunk_strategy"], "flat")
    self.assertNotIn("parent_content", chunks[0]["metadata"])
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest backend/tests/test_docx_pipeline.py -q
```

Expected: fail because `split_parent_child_text` does not exist.

- [ ] **Step 3: Implement splitter**

Add:

```python
def _validate_chunk_config(parent_chunk_size, child_chunk_size, chunk_overlap):
    ...

def split_parent_child_text(...):
    ...
```

Keep `split_text` as the flat primitive and make `split_text_with_metadata` accept `chunk_strategy`, `parent_chunk_size`, `child_chunk_size`, and `parent_id_prefix`.

- [ ] **Step 4: Run tests to verify green**

Run:

```bash
python -m pytest backend/tests/test_docx_pipeline.py -q
```

Expected: new splitter tests pass and existing DOCX tests stay green.

### Task 2: Job And API Config Plumbing

**Files:**
- Modify: `backend/app/api/v1/documents.py`
- Modify: `backend/app/services/document_service.py`
- Modify: `backend/app/services/job_service.py`
- Test: `backend/tests/test_docx_pipeline.py`

- [ ] **Step 1: Write failing config plumbing tests**

Add a `_parse_text_mode` test:

```python
def test_parse_text_mode_accepts_parent_child_config(self):
    chunks, images = job_service._parse_text_mode(
        file_content=("开店流程。" * 120).encode("utf-8"),
        file_name="guide.txt",
        job_id="job-parent-child",
        chunk_size=500,
        chunk_overlap=20,
        parent_chunk_size=300,
        child_chunk_size=120,
        chunk_strategy="parent_child",
    )
    self.assertEqual(images, [])
    self.assertGreater(len(chunks), 1)
    self.assertEqual(chunks[0]["metadata"]["chunk_strategy"], "parent_child")
    self.assertIn("parent_content", chunks[0]["metadata"])
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest backend/tests/test_docx_pipeline.py::DocxPipelineTests::test_parse_text_mode_accepts_parent_child_config -q
```

Expected: fail because `_parse_text_mode` does not accept the new keyword arguments.

- [ ] **Step 3: Implement config plumbing**

Add optional args after existing args so old callers still work:

```python
parent_chunk_size: int | None = None
child_chunk_size: int | None = None
chunk_strategy: str = "parent_child"
```

Forward these through API -> document service -> job service -> parser.

- [ ] **Step 4: Run tests to verify green**

Run:

```bash
python -m pytest backend/tests/test_docx_pipeline.py -q
```

Expected: all tests in the file pass.

### Task 3: Image-Text Parent-Child Chunking

**Files:**
- Modify: `backend/app/services/doc_image_parser.py`
- Modify: `backend/app/services/job_service.py`
- Test: `backend/tests/test_docx_pipeline.py`

- [ ] **Step 1: Write failing DOCX image-mode test**

Extend the existing DOCX image test to call:

```python
chunks, image_records = doc_image_parser.parse_word(
    file_content=_build_docx_bytes(),
    job_id="job-1",
    collection="kb-demo",
    file_name="guide.docx",
    chunk_size=500,
    chunk_overlap=20,
    parent_chunk_size=900,
    child_chunk_size=260,
    chunk_strategy="parent_child",
)
```

Assert:

```python
self.assertEqual(chunks[0]["metadata"]["chunk_strategy"], "parent_child")
self.assertIn("parent_content", chunks[0]["metadata"])
self.assertEqual(len(image_records), 1)
self.assertTrue(any(image_records[0]["placeholder"] in c["content"] or image_records[0]["placeholder"] in c["metadata"].get("parent_content", "") for c in chunks))
self.assertIn(image_records[0]["chunk_id"], {c["chunk_id"] for c in chunks})
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m pytest backend/tests/test_docx_pipeline.py::DocxPipelineTests::test_parse_word_docx_preserves_text_structure_and_inserts_placeholders -q
```

Expected: fail because `parse_word` does not accept parent-child args or does not add parent metadata.

- [ ] **Step 3: Implement image stream splitting**

Add helpers in `doc_image_parser.py`:

```python
def _finalize_stream_chunks(stream_text, image_records, file_name, base_metadata, ...):
    ...

def _rebind_image_records(chunks, image_records):
    ...
```

Refactor `parse_pdf` and `parse_word` so they build a text stream with image placeholders, call the splitter, assign final child `chunk_id`s, and rebind image records.

- [ ] **Step 4: Run tests to verify green**

Run:

```bash
python -m pytest backend/tests/test_docx_pipeline.py -q
```

Expected: DOCX image records point at final child chunk ids.

### Task 4: Retrieval Metadata Backfill And Generation Context

**Files:**
- Modify: `backend/app/services/milvus_service.py`
- Modify: `backend/agents/knowledge/nodes/generate.py`
- Test: `backend/tests/test_generate_sources.py`

- [ ] **Step 1: Write failing generation unit test**

Add a focused test that calls `_chunk_to_source_block` with:

```python
chunk = {
    "chunk_id": "chunk-1",
    "file_name": "guide.txt",
    "content": "子块内容",
    "metadata": {"parent_content": "父块完整上下文"},
}
```

Assert the source block includes `父块完整上下文`.

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m pytest backend/tests/test_generate_sources.py -q
```

Expected: fail because `_chunk_to_source_block` uses only child content.

- [ ] **Step 3: Implement context expansion**

Update `_chunk_to_source_block` to prefer `metadata.parent_content` for model context while keeping source snippets unchanged elsewhere. Update Milvus backfill to merge `metadata` from `chunk_repo.get_by_ids(...)`.

- [ ] **Step 4: Run tests to verify green**

Run:

```bash
python -m pytest backend/tests/test_generate_sources.py backend/tests/test_docx_pipeline.py -q
```

Expected: both files pass.

### Task 5: Frontend Chunk Controls

**Files:**
- Modify: `frontend/src/components/doc/DocUpload.vue`

- [ ] **Step 1: Inspect current component constraints**

Read `DocUpload.vue` and confirm only the chunk parameter form and submit payloads need edits.

- [ ] **Step 2: Implement controls**

Add:

```js
const CHUNK_PRESETS = {
  precise: { parentChunkSize: 1200, childChunkSize: 350, chunkOverlap: 60 },
  balanced: { parentChunkSize: 1800, childChunkSize: 500, chunkOverlap: 80 },
  broad: { parentChunkSize: 2400, childChunkSize: 700, chunkOverlap: 120 },
}
```

Replace `chunkSize` with `chunkStrategy`, `chunkPreset`, `parentChunkSize`, and `childChunkSize` in both single and category config objects.

- [ ] **Step 3: Submit new and compatible fields**

Single upload form data must include:

```js
formData.append('chunk_strategy', config.value.chunkStrategy)
formData.append('parent_chunk_size', String(config.value.parentChunkSize))
formData.append('child_chunk_size', String(config.value.childChunkSize))
formData.append('chunk_size', String(config.value.childChunkSize))
```

Batch chunking params must include the equivalent snake_case params.

- [ ] **Step 4: Run frontend build**

Run:

```bash
npm --prefix frontend run build
```

Expected: build exits 0.

### Task 6: Final Verification

**Files:**
- All modified files

- [ ] **Step 1: Run backend focused tests**

Run:

```bash
python -m pytest backend/tests/test_docx_pipeline.py backend/tests/test_generate_sources.py -q
```

Expected: exit 0.

- [ ] **Step 2: Run frontend build**

Run:

```bash
npm --prefix frontend run build
```

Expected: exit 0.

- [ ] **Step 3: Review diff**

Run:

```bash
git diff -- backend/app/services/chunk_splitter.py backend/app/services/doc_image_parser.py backend/app/services/job_service.py backend/app/services/document_service.py backend/app/api/v1/documents.py backend/app/services/milvus_service.py backend/agents/knowledge/nodes/generate.py backend/tests/test_docx_pipeline.py backend/tests/test_generate_sources.py frontend/src/components/doc/DocUpload.vue docs/superpowers/plans/2026-05-27-parent-child-multimodal-chunking.md
```

Expected: diff only contains parent-child chunking work.
