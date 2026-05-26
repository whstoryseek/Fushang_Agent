# -*- coding: utf-8 -*-
"""
Document parsers for PDF and DOCX files.

The DOCX path exposes a shared block extractor so text mode and image mode
reuse the same paragraph/table/link ordering rules.
"""

import io
import logging
import re
import uuid
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from app.services.oss_service import get_oss_service

fitz.TOOLS.mupdf_display_errors(False)

logger = logging.getLogger(__name__)

_IMAGE_RE = re.compile(r"<<IMAGE:[0-9a-f]+>>")
_IMAGE_REF_RE = re.compile(r"<<IMGREF:([^>]+)>>")
_SENTENCE_ENDS = re.compile(r"[。？！?!\n]")
_LIST_ITEM_RE = re.compile(r"^[（(]?\d+[）)]|^[①②③④⑤⑥⑦⑧⑨⑩]|^[a-zA-Z]\.")
_TRANSITION_STARTS = (
    "但是",
    "然后",
    "除外",
    "不包括",
    "不含",
    "除非",
    "否则",
    "注意",
    "警告",
)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

W_P = f"{{{W_NS}}}p"
W_TBL = f"{{{W_NS}}}tbl"
W_TR = f"{{{W_NS}}}tr"
W_TC = f"{{{W_NS}}}tc"
W_T = f"{{{W_NS}}}t"
W_HYPERLINK = f"{{{W_NS}}}hyperlink"
A_BLIP = f"{{{A_NS}}}blip"
R_EMBED = f"{{{R_NS}}}embed"
R_ID = f"{{{R_NS}}}id"


def _file_base(file_name: str) -> str:
    return file_name.rsplit(".", 1)[0] if "." in file_name else file_name


def _image_ref_token(relation_id: str) -> str:
    return f"<<IMGREF:{relation_id}>>"


def _strip_image_refs(text: str) -> str:
    return _IMAGE_REF_RE.sub("", text or "")


def _clean_word_text(text: str) -> str:
    return (text or "").replace("\xa0", " ").strip()


def _format_hyperlink(text: str, url: str) -> str:
    label = _clean_word_text(text)
    target = _clean_word_text(url)
    if not target:
        return label
    if not label or label == target:
        return target
    return f"{label} ({target})"


def _escape_markdown_cell(text: str) -> str:
    return _clean_word_text(text).replace("|", r"\|").replace("\n", " <br> ")


def _markdown_row(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _upload_image(image_bytes: bytes, ext: str, collection: str, file_name: str, chunk_id: str) -> str:
    oss_svc = get_oss_service()
    filename = f"{uuid.uuid4().hex[:12]}.{ext}"
    return oss_svc.upload_file(f"rag_image/{collection}/{file_name}/{chunk_id}", filename, image_bytes)


def _smart_overlap(text: str, overlap_size: int) -> str:
    if overlap_size <= 0 or len(text) <= overlap_size:
        return ""
    tail = text[-(overlap_size * 2):]
    matches = list(_SENTENCE_ENDS.finditer(tail))
    if matches:
        candidate = tail[matches[-1].end():].lstrip()
        if candidate:
            return candidate
    return text[-overlap_size:]


def _should_merge(text1: str, text2: str) -> bool:
    c1 = _IMAGE_RE.sub("", text1).strip()
    c2 = _IMAGE_RE.sub("", text2).strip()
    if not c1 or not c2:
        return False
    if c1.endswith((":", "：", ";", "；")):
        return True
    if _LIST_ITEM_RE.match(c2):
        return True
    if c2.startswith(_TRANSITION_STARTS):
        return True
    return False


def _post_process(
    chunks: List[Dict],
    image_records: List[Dict],
    file_base: str,
) -> Tuple[List[Dict], List[Dict]]:
    if not chunks:
        return chunks, image_records

    merged: List[Dict] = [chunks[0]]
    for current in chunks[1:]:
        previous = merged[-1]
        if _should_merge(previous["content"], current["content"]):
            old_chunk_id = current["chunk_id"]
            previous["content"] += current["content"]
            for record in image_records:
                if record["chunk_id"] == old_chunk_id:
                    record["chunk_id"] = previous["chunk_id"]
            logger.debug("[Merge] %s: %s -> %s", file_base, old_chunk_id, previous["chunk_id"])
        else:
            merged.append(current)

    return merged, image_records


def _resolve_hyperlink_target(node, doc: DocxDocument) -> str:
    relation_id = node.get(R_ID)
    if not relation_id:
        return ""
    relation = doc.part.rels.get(relation_id)
    if not relation:
        return ""
    return str(getattr(relation, "target_ref", "") or "")


def _render_word_inline(node, doc: DocxDocument) -> List[str]:
    if node.tag == W_HYPERLINK:
        text = "".join(
            part
            for child in node.iterchildren()
            for part in _render_word_inline(child, doc)
        )
        text = _strip_image_refs(text)
        formatted = _format_hyperlink(text, _resolve_hyperlink_target(node, doc))
        return [formatted] if formatted else []
    if node.tag == W_T:
        return [node.text or ""]
    if node.tag == A_BLIP:
        relation_id = node.get(R_EMBED)
        return [_image_ref_token(relation_id)] if relation_id else []

    parts: List[str] = []
    for child in node.iterchildren():
        parts.extend(_render_word_inline(child, doc))
    return parts


def _render_word_paragraph(node, doc: DocxDocument) -> str:
    parts: List[str] = []
    for child in node.iterchildren():
        parts.extend(_render_word_inline(child, doc))
    return _clean_word_text("".join(parts))


def _render_word_table_cell(node, doc: DocxDocument) -> str:
    blocks: List[str] = []
    for child in node.iterchildren():
        if child.tag == W_P:
            paragraph_text = _render_word_paragraph(child, doc)
            if paragraph_text:
                blocks.append(paragraph_text)
        elif child.tag == W_TBL:
            table_markdown = _render_word_table(child, doc)
            if table_markdown:
                blocks.append(table_markdown.replace("\n", " <br> "))
    return _escape_markdown_cell(" <br> ".join(blocks))


def _render_word_table(node, doc: DocxDocument) -> str:
    rows: List[List[str]] = []
    for row in node.findall(f"./{W_TR}"):
        cells = [_render_word_table_cell(cell, doc) for cell in row.findall(f"./{W_TC}")]
        if any(cell for cell in cells):
            rows.append(cells)

    if not rows:
        return ""

    column_count = max(len(row) for row in rows)
    normalized = [row + [""] * (column_count - len(row)) for row in rows]
    lines = [
        _markdown_row(normalized[0]),
        _markdown_row(["---"] * column_count),
    ]
    lines.extend(_markdown_row(row) for row in normalized[1:])
    return "\n".join(lines)


def extract_word_blocks(file_content: bytes) -> List[str]:
    doc = DocxDocument(io.BytesIO(file_content))
    blocks: List[str] = []

    for child in doc.element.body.iterchildren():
        if child.tag == W_P:
            text = _render_word_paragraph(child, doc)
        elif child.tag == W_TBL:
            text = _render_word_table(child, doc)
        else:
            continue

        if text.strip():
            blocks.append(text)

    return blocks


def extract_word_text(file_content: bytes) -> str:
    blocks = []
    for block in extract_word_blocks(file_content):
        cleaned_block = _strip_image_refs(block).strip()
        if cleaned_block:
            blocks.append(cleaned_block)
    return "\n\n".join(blocks)


def parse_pdf(
    file_content: bytes,
    job_id: str,
    collection: str,
    file_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    image_dpi: int = 150,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    doc = fitz.open(stream=file_content, filetype="pdf")

    elements: List[Dict] = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))

        for block in blocks:
            if block["type"] != 0:
                continue
            text = "".join(
                span["text"]
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ).strip()
            if text:
                y_center = (block["bbox"][1] + block["bbox"][3]) / 2
                elements.append(
                    {
                        "type": "text",
                        "page": page_num,
                        "y_center": y_center,
                        "text": text,
                    }
                )

        for image_info in page.get_images(full=True):
            xref = image_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                if len(image_bytes) < 1000:
                    continue
                image_rects = page.get_image_rects(xref)
                if not image_rects:
                    continue
                y_center = (image_rects[0].y0 + image_rects[0].y1) / 2
                elements.append(
                    {
                        "type": "image",
                        "page": page_num,
                        "y_center": y_center,
                        "img_bytes": image_bytes,
                        "ext": ext,
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[Parser] image extraction failed page=%s xref=%s: %s", page_num, xref, exc)

    doc.close()

    elements.sort(key=lambda element: (element["page"], element["y_center"]))

    file_base = _file_base(file_name)
    chunks: List[Dict] = []
    image_records: List[Dict] = []

    buffer = ""
    text_len = 0
    chunk_idx = 0
    img_sort = 0
    overlap_buf = ""
    first_page = None

    def _new_chunk_id() -> str:
        return str(uuid.uuid4())

    current_chunk_id = _new_chunk_id()

    def _seal() -> None:
        nonlocal buffer, text_len, chunk_idx, img_sort, overlap_buf, first_page, current_chunk_id
        if buffer.strip():
            chunks.append(
                {
                    "chunk_id": current_chunk_id,
                    "chunk_index": chunk_idx,
                    "content": buffer,
                    "metadata": {
                        "page": first_page,
                        "chunk_id": current_chunk_id,
                        "prev_chunk_id": None,
                        "next_chunk_id": None,
                    },
                }
            )
            overlap_buf = _smart_overlap(_IMAGE_RE.sub("", buffer), chunk_overlap)
        chunk_idx += 1
        img_sort = 0
        buffer = ""
        text_len = 0
        first_page = None
        current_chunk_id = _new_chunk_id()

    for element in elements:
        if element["type"] == "text":
            text = element["text"]
            if first_page is None:
                first_page = element["page"]

            if not buffer and overlap_buf:
                buffer = overlap_buf
                text_len = len(overlap_buf)
                overlap_buf = ""

            remaining = text
            while remaining:
                space = chunk_size - text_len
                part = remaining[:space]
                buffer += part
                text_len += len(part)
                remaining = remaining[space:]
                if text_len >= chunk_size:
                    _seal()
                    if remaining and overlap_buf:
                        buffer = overlap_buf
                        text_len = len(overlap_buf)
                        overlap_buf = ""
            continue

        if not buffer and overlap_buf:
            buffer = overlap_buf
            text_len = len(overlap_buf)
            overlap_buf = ""

        try:
            oss_key = _upload_image(
                element["img_bytes"],
                element["ext"],
                collection,
                file_name,
                current_chunk_id,
            )
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("[Parser] image upload failed: %s", exc)
            continue

        placeholder = f"<<IMAGE:{uuid.uuid4().hex[:8]}>>"
        buffer += placeholder
        image_records.append(
            {
                "id": str(uuid.uuid4()),
                "chunk_id": current_chunk_id,
                "job_id": job_id,
                "placeholder": placeholder,
                "oss_key": oss_key,
                "page": element["page"],
                "sort_order": img_sort,
            }
        )
        img_sort += 1

    if buffer.strip():
        chunks.append(
            {
                "chunk_id": current_chunk_id,
                "chunk_index": chunk_idx,
                "content": buffer,
                "metadata": {
                    "page": first_page,
                    "chunk_id": current_chunk_id,
                    "prev_chunk_id": None,
                    "next_chunk_id": None,
                },
            }
        )

    chunks, image_records = _post_process(chunks, image_records, file_base)
    return chunks, image_records


def parse_word(
    file_content: bytes,
    job_id: str,
    collection: str,
    file_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    doc = DocxDocument(io.BytesIO(file_content))
    tokenized_content = "\n\n".join(extract_word_blocks(file_content))

    file_base = _file_base(file_name)
    chunks: List[Dict] = []
    image_records: List[Dict] = []

    buffer = ""
    text_len = 0
    chunk_idx = 0
    img_sort = 0
    overlap_buf = ""
    bound_relations: set[str] = set()

    def _new_chunk_id() -> str:
        return str(uuid.uuid4())

    current_chunk_id = _new_chunk_id()

    def _seal() -> None:
        nonlocal buffer, text_len, chunk_idx, img_sort, overlap_buf, current_chunk_id
        if buffer.strip():
            chunks.append(
                {
                    "chunk_id": current_chunk_id,
                    "chunk_index": chunk_idx,
                    "content": buffer,
                    "metadata": {
                        "page": None,
                        "chunk_id": current_chunk_id,
                        "prev_chunk_id": None,
                        "next_chunk_id": None,
                    },
                }
            )
            overlap_buf = _smart_overlap(_IMAGE_RE.sub("", buffer), chunk_overlap)
        chunk_idx += 1
        img_sort = 0
        buffer = ""
        text_len = 0
        current_chunk_id = _new_chunk_id()

    def _append_text(text: str) -> None:
        nonlocal buffer, text_len, overlap_buf
        if not text:
            return
        if not buffer and overlap_buf:
            buffer = overlap_buf
            text_len = len(overlap_buf)
            overlap_buf = ""

        remaining = text
        while remaining:
            space = chunk_size - text_len
            part = remaining[:space]
            buffer += part
            text_len += len(part)
            remaining = remaining[space:]
            if text_len >= chunk_size:
                _seal()
                if remaining and overlap_buf:
                    buffer = overlap_buf
                    text_len = len(overlap_buf)
                    overlap_buf = ""

    def _insert_image(relation_id: str) -> None:
        nonlocal img_sort, buffer, text_len, overlap_buf
        if not relation_id or relation_id in bound_relations:
            return

        if not buffer and overlap_buf:
            buffer = overlap_buf
            text_len = len(overlap_buf)
            overlap_buf = ""

        try:
            image_part = doc.part.related_parts[relation_id]
            image_bytes = image_part.blob
            if len(image_bytes) < 1000:
                return
            ext = image_part.content_type.split("/")[-1].replace("jpeg", "jpg")
            oss_key = _upload_image(image_bytes, ext, collection, file_name, current_chunk_id)
        except Exception as exc:  # pragma: no cover - network/format failure
            logger.warning("[WordParser] image extraction failed rId=%s: %s", relation_id, exc)
            return

        placeholder = f"<<IMAGE:{uuid.uuid4().hex[:8]}>>"
        buffer += placeholder
        image_records.append(
            {
                "id": str(uuid.uuid4()),
                "chunk_id": current_chunk_id,
                "job_id": job_id,
                "placeholder": placeholder,
                "oss_key": oss_key,
                "page": None,
                "sort_order": img_sort,
            }
        )
        bound_relations.add(relation_id)
        img_sort += 1

    cursor = 0
    for match in _IMAGE_REF_RE.finditer(tokenized_content):
        _append_text(tokenized_content[cursor:match.start()])
        _insert_image(match.group(1))
        cursor = match.end()
    _append_text(tokenized_content[cursor:])

    if buffer.strip():
        chunks.append(
            {
                "chunk_id": current_chunk_id,
                "chunk_index": chunk_idx,
                "content": buffer,
                "metadata": {
                    "page": None,
                    "chunk_id": current_chunk_id,
                    "prev_chunk_id": None,
                    "next_chunk_id": None,
                },
            }
        )

    chunks, image_records = _post_process(chunks, image_records, file_base)
    return chunks, image_records
