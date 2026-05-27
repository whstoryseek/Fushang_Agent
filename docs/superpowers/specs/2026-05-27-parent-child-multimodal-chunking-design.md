# Parent Child Multimodal Chunking

## Goal

Refactor the RAG document chunking pipeline into a unified parent-child chunking model that works for standard text and image-text documents.

The new pipeline should improve retrieval precision by embedding smaller child chunks while preserving larger parent context for answer generation. It should also keep the current image placeholder and image record behavior intact for multimodal knowledge bases.

## Existing Context

The current system has two different chunking paths:

- Standard mode uses `backend/app/services/chunk_splitter.py` through `job_service._parse_text_mode`. It produces flat text chunks.
- Image-text mode uses `backend/app/services/doc_image_parser.py` through `job_service._parse_image_mode`. It extracts PDF/DOCX images, uploads them to OSS, inserts `<<IMAGE:...>>` placeholders into chunk content, and stores image records in `knowledge_chunk_image`.
- Excel chunking is separate and row-oriented. It already has a natural structure and should keep its own `excel_rows_per_chunk` behavior.
- Chunk persistence is in `knowledge_chunk` and `knowledge_chunk_origin`. The `metadata` JSONB column can hold parent-child metadata, so no database migration is required.
- Vector upsert reads chunks from PostgreSQL and writes one vector row per stored chunk. That should remain one vector row per child chunk.

The reference implementation from `E:/Rag_Item/integrated_qa_system` uses a parent splitter followed by a child splitter, with `parent_id`, `id`, and `parent_content` attached to each child. This project should adopt that idea, but keep the existing image placeholder and image record lifecycle.

## Recommended Approach

Use a unified chunking abstraction that produces final child chunks for storage and vectorization:

- Parent chunks provide answer context.
- Child chunks provide retrieval granularity.
- Final stored records are child chunks.
- Each child chunk stores parent metadata in `metadata`.
- Image records are rebound to the final child chunk that contains the corresponding placeholder.

This gives one consistent behavior across standard and image-text modes while avoiding a schema change.

## API And Config

Add new upload and batch chunking parameters:

```text
chunk_strategy = "parent_child" | "flat"
parent_chunk_size = 1800
child_chunk_size = 500
chunk_overlap = 80
```

Keep existing fields for compatibility:

```text
chunk_size
chunk_overlap
```

Compatibility rules:

- If `child_chunk_size` is absent, use `chunk_size`.
- If `parent_chunk_size` is absent, use `max(child_chunk_size * 3, 1200)`.
- If `chunk_strategy` is absent, default to `parent_child`.
- If `chunk_strategy="flat"`, keep the existing flat behavior and treat `child_chunk_size` as the old `chunk_size`.
- Validate `parent_chunk_size >= child_chunk_size`.
- Clamp or reject invalid overlaps where `chunk_overlap >= child_chunk_size`.

The backend entry points affected are:

- `backend/app/api/v1/documents.py`
- `backend/app/services/document_service.py`
- `backend/app/services/job_service.py`

## Chunk Metadata

Each stored child chunk should have this metadata shape:

```json
{
  "file_name": "guide.docx",
  "source": "docx",
  "chunk_strategy": "parent_child",
  "parent_id": "job-1-parent-0",
  "parent_index": 0,
  "child_index": 2,
  "parent_content": "larger parent text with optional image placeholders",
  "page": 3,
  "chunk_index": 5
}
```

Required metadata:

- `chunk_strategy`
- `parent_id`
- `parent_index`
- `child_index`
- `parent_content`
- existing file/source/page metadata where available

The top-level `chunk_index` remains the display and persistence order.

## Standard Text Flow

Standard text chunking should live in `backend/app/services/chunk_splitter.py`.

The module should expose:

```python
def split_parent_child_text(
    text: str,
    parent_chunk_size: int = 1800,
    child_chunk_size: int = 500,
    chunk_overlap: int = 80,
    base_metadata: dict | None = None,
    parent_id_prefix: str | None = None,
) -> list[dict]:
    ...
```

Implementation rules:

- Reuse the current Chinese-friendly recursive separators.
- Split parent chunks first.
- Split each parent into child chunks.
- Preserve the current semantic merge behavior for short fragments, list items, and transition words.
- Use sentence-boundary overlap.
- Avoid empty chunks.
- Assign stable sequential `chunk_index` values across all children.

`split_text_with_metadata` should remain as a compatibility wrapper. For the default strategy it can delegate to `split_parent_child_text`; for `flat`, it can keep flat chunks.

## Image-Text Flow

Image-text parsing should preserve the current PDF/DOCX extraction behavior but change the final chunking stage.

The parser should first build a document stream:

- Text appears in reading order.
- Images appear as `<<IMAGE:...>>` placeholders.
- Image bytes are uploaded to OSS as today.
- Temporary image records initially point at a provisional stream or parent context.

Then the unified chunking logic should split the stream into parent and child chunks.

After final child chunks are produced:

- Each image record must be assigned to the child chunk whose content contains its placeholder.
- If a placeholder appears only in `parent_content` but not in the child content, the image should attach to the nearest child in that parent. This prevents orphaned image records.
- `metadata.parent_content` may contain image placeholders. This is useful for answer context, but vectorization already strips placeholders before embedding.

The affected functions are:

- `backend/app/services/doc_image_parser.py::parse_pdf`
- `backend/app/services/doc_image_parser.py::parse_word`
- `backend/app/services/job_service.py::_parse_image_mode`

## Excel Flow

Excel remains row-oriented:

- Text-only Excel keeps `excel_rows_per_chunk`.
- Excel image columns continue one data row per chunk when image columns are configured.
- The new parent-child fields are not required for Excel chunks in this iteration.
- Excel chunks should still add `chunk_strategy="excel_rows"` or `chunk_strategy="excel_image_rows"` to metadata for transparency.

## Retrieval And Generation

No retrieval schema changes are required.

Milvus continues to store one vector per PostgreSQL chunk. Those chunks are now child chunks.

Generation and source construction should prefer child content for source snippets, but answer prompting may use parent context when available:

- Keep `content` as the child chunk text.
- Use `metadata.parent_content` as expanded context when building prompt source blocks.
- Preserve `chunk_id`, `job_id`, `file_name`, and `chunk_index` in source outputs so the chunk editor remains usable.

If prompt expansion is too risky for the first implementation, it can be limited to storing parent metadata first. The important invariant is that parent context is available for later generation improvements.

## Frontend

Update `frontend/src/components/doc/DocUpload.vue` to replace the current simple chunk size fields with a clearer chunking control.

Fields:

```text
chunkStrategy: parent_child | flat
chunkPreset: precise | balanced | broad
parentChunkSize
childChunkSize
chunkOverlap
excelRowsPerChunk
imageDpi
```

Preset defaults:

```text
precise: parent=1200, child=350, overlap=60
balanced: parent=1800, child=500, overlap=80
broad: parent=2400, child=700, overlap=120
```

UX behavior:

- Default to `balanced`.
- Show parent size only when `chunkStrategy="parent_child"`.
- Show `imageDpi` only in image mode.
- Show `excelRowsPerChunk` in category batch mode as today.
- Keep manual numeric controls after a preset is selected.
- Submit both new fields and old-compatible `chunk_size=childChunkSize` to reduce backend compatibility risk.

## Testing

Backend tests should cover:

- Pure text parent-child chunking creates multiple child chunks with `parent_id`, `child_index`, and `parent_content`.
- Flat strategy remains available.
- DOCX image mode preserves placeholders and image records after parent-child splitting.
- `job_service._parse_text_mode` accepts new parent-child parameters and returns child chunks.
- Old callers that only pass `chunk_size` still work.
- Excel metadata includes an explicit chunk strategy and existing image-column behavior is unchanged.

Frontend tests should cover:

- Preset selection updates parent size, child size, and overlap.
- Upload submits `chunk_strategy`, `parent_chunk_size`, `child_chunk_size`, and compatible `chunk_size`.
- Batch chunking submits the same chunk fields plus `excel_rows_per_chunk`.

## Non-Goals

This change will not:

- Add a new database table for parent chunks.
- Change Milvus schema.
- Replace Excel row chunking with parent-child text chunking.
- Add a semantic segmentation model dependency.
- Rework the whole retrieval graph.

## Rollout Notes

The first safe rollout is:

1. Add backend parent-child chunking while preserving flat compatibility.
2. Switch text mode to parent-child by default.
3. Switch image-text mode to parent-child and verify image placeholders.
4. Update frontend controls.
5. Optionally update generation prompt construction to use `metadata.parent_content`.

Existing chunks already in the database remain valid. Only newly uploaded or re-chunked files will use the parent-child metadata.
