# -*- coding: utf-8 -*-
"""
纯文本切分器（标准模式）
中文友好，支持句子边界 overlap，复用图文模式的 should_merge 逻辑
"""
import io
import re
from typing import List, Optional, Tuple
import uuid

# 句子结束符（用于 overlap 从句子边界开始）
_SENTENCE_END = re.compile(r'[。！？.!?]')

# 段落分隔符优先级（从粗到细）
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ", ""]

# 列表项开头（用于 should_merge 检测）
_LIST_PREFIX = re.compile(r'^[\s]*[①②③④⑤⑥⑦⑧⑨⑩\-\*•◆▶➤]|^\s*\d+[\.、\)）]')
# 转折词开头
_TRANSITION_START = re.compile(r'^(但是|然而|不过|此外|另外|同时|因此|所以|综上|总之|首先|其次|最后|另一方面)')


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    将文本切分为 chunks。
    1. 按分隔符递归切分到 chunk_size 以内
    2. 合并过短的片段
    3. should_merge：检测语义断裂，合并相邻 chunk
    4. 添加 overlap
    """
    if not text or not text.strip():
        return []

    raw_chunks = _recursive_split(text.strip(), chunk_size)
    merged = _merge_short(raw_chunks, chunk_size)
    merged = _should_merge(merged, chunk_size)
    result = _add_overlap(merged, chunk_overlap)
    return [c for c in result if c.strip()]


def split_text_with_metadata(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    base_metadata: dict = None,
    chunk_strategy: str = "parent_child",
    parent_chunk_size: Optional[int] = None,
    child_chunk_size: Optional[int] = None,
    parent_id_prefix: Optional[str] = None,
) -> List[dict]:
    """返回带 metadata 的 chunk 列表，格式与图文模式一致"""
    strategy = chunk_strategy or "parent_child"
    child_size = child_chunk_size or chunk_size

    if strategy == "parent_child":
        return split_parent_child_text(
            text=text,
            parent_chunk_size=parent_chunk_size or max(child_size * 3, 1200),
            child_chunk_size=child_size,
            chunk_overlap=chunk_overlap,
            base_metadata=base_metadata,
            parent_id_prefix=parent_id_prefix,
        )
    if strategy != "flat":
        raise ValueError(f"不支持的切块策略: {strategy}")

    _validate_chunk_config(child_size, child_size, chunk_overlap)
    chunks = split_text(text, child_size, chunk_overlap)
    meta = base_metadata or {}
    return [
        {
            "content": c,
            "chunk_index": i,
            "metadata": {**meta, "chunk_index": i, "chunk_strategy": "flat"},
        }
        for i, c in enumerate(chunks)
    ]


# ── 内部实现 ──────────────────────────────────────────────────────────────────

def _validate_chunk_config(
    parent_chunk_size: int,
    child_chunk_size: int,
    chunk_overlap: int,
) -> None:
    if parent_chunk_size <= 0:
        raise ValueError("parent_chunk_size 必须大于 0")
    if child_chunk_size <= 0:
        raise ValueError("child_chunk_size 必须大于 0")
    if parent_chunk_size < child_chunk_size:
        raise ValueError("parent_chunk_size 必须大于或等于 child_chunk_size")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")
    if chunk_overlap >= child_chunk_size:
        raise ValueError("chunk_overlap 必须小于 child_chunk_size")


def split_parent_child_text(
    text: str,
    parent_chunk_size: int = 1800,
    child_chunk_size: int = 500,
    chunk_overlap: int = 80,
    base_metadata: dict = None,
    parent_id_prefix: Optional[str] = None,
) -> List[dict]:
    """
    将文本切成父子块。

    最终返回子块；每个子块在 metadata 中携带父块上下文，便于检索命中后扩展回答上下文。
    """
    if not text or not text.strip():
        return []

    _validate_chunk_config(parent_chunk_size, child_chunk_size, chunk_overlap)

    meta = base_metadata or {}
    prefix = parent_id_prefix or "doc"
    parent_chunks = split_text(text, parent_chunk_size, chunk_overlap)
    child_chunks: List[dict] = []
    global_child_index = 0

    for parent_index, parent_content in enumerate(parent_chunks):
        parent_id = f"{prefix}-parent-{parent_index}"
        children = split_text(parent_content, child_chunk_size, chunk_overlap)
        for child_index, child_content in enumerate(children):
            child_chunks.append({
                "content": child_content,
                "chunk_index": global_child_index,
                "metadata": {
                    **meta,
                    "chunk_index": global_child_index,
                    "chunk_strategy": "parent_child",
                    "parent_id": parent_id,
                    "parent_index": parent_index,
                    "child_index": child_index,
                    "parent_content": parent_content,
                },
            })
            global_child_index += 1

    return child_chunks

def _recursive_split(text: str, chunk_size: int) -> List[str]:
    """递归按分隔符切分，直到每段 <= chunk_size"""
    if len(text) <= chunk_size:
        return [text]

    for sep in _SEPARATORS:
        if sep == "":
            # 最后手段：按字符硬切
            return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
        if sep in text:
            parts = text.split(sep)
            result = []
            current = ""
            for part in parts:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        result.append(current)
                    if len(part) > chunk_size:
                        result.extend(_recursive_split(part, chunk_size))
                        current = ""
                    else:
                        current = part
            if current:
                result.append(current)
            return result

    return [text]


def _merge_short(chunks: List[str], chunk_size: int, min_size: int = 50) -> List[str]:
    """将过短的 chunk 合并到前一个"""
    if not chunks:
        return []
    result = [chunks[0]]
    for chunk in chunks[1:]:
        if len(chunk) < min_size and len(result[-1]) + len(chunk) <= chunk_size:
            result[-1] = result[-1] + chunk
        else:
            result.append(chunk)
    return result


def _should_merge(chunks: List[str], chunk_size: int) -> List[str]:
    """
    检测语义断裂并合并：
    - 上一个 chunk 以冒号结尾
    - 当前 chunk 以列表项开头
    - 当前 chunk 以转折词开头
    合并后若超过 chunk_size 则不合并
    """
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for chunk in chunks[1:]:
        prev = result[-1]
        should = (
            prev.rstrip().endswith(("：", ":"))
            or _LIST_PREFIX.match(chunk)
            or _TRANSITION_START.match(chunk)
        )
        if should and len(prev) + len(chunk) <= chunk_size * 1.5:
            result[-1] = prev + chunk
        else:
            result.append(chunk)
    return result


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    """
    在每个 chunk 开头加上前一个 chunk 末尾的 overlap 字符。
    从句子边界开始，不在词中间截断。
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        tail = prev[-overlap:] if len(prev) > overlap else prev
        # 从句子边界开始
        m = _SENTENCE_END.search(tail)
        if m:
            tail = tail[m.end():]
        if tail.strip():
            result.append(tail + chunks[i])
        else:
            result.append(chunks[i])
    return result


def _find_header_row(rows: list, expected_cols: list = None) -> int:
    """
    在 rows 中定位 header 行。
    如果 expected_cols 给出，找第一个包含所有 expected_cols 的行；
    否则返回第一个非全空行。
    """
    if not rows:
        return 0

    if expected_cols:
        expected_set = set(expected_cols)
        for i, row in enumerate(rows):
            row_set = set(str(c).strip() for c in row if c is not None)
            if expected_set.issubset(row_set):
                return i

    # fallback: 第一个非全空行
    for i, row in enumerate(rows):
        if any(c is not None and str(c).strip() for c in row):
            return i

    return 0


# ── Excel 切分 ────────────────────────────────────────────────────────────────

def split_excel(
    file_content: bytes,
    file_name: str,
    job_id: str,
    rows_per_chunk: int = 50,
    base_metadata: dict = None,
    column_config: dict = None,
) -> Tuple[List[dict], List[dict]]:
    """
    将 Excel 文件按 sheet 切分为 chunks。
    支持单元格内嵌图片：column_config 中 type="image" 的列会被提取为占位符。

    column_config 格式（每文件独立，由前端传入）：
        {
          "Sheet1": [
            {"original": "省份", "alias": "省份", "type": "text"},
            {"original": "产品照片", "alias": "photo", "type": "image"},
            ...
          ],
          "Sheet2": [...]
        }
    None 表示全选、不改名、无图片。

    切片 content 格式（key=value，每行一条记录）：
        文件：{file_name}  Sheet：{sheet_name}
        省份=浙江, 城市=杭州, 产品照片=<<IMAGE:abc123>>
        省份=山东, 城市=聊城, 产品照片=[无图片]
        ...

    返回 (chunks, image_data)：
        chunks: [{"chunk_id": str, "content": str, "metadata": dict}, ...]
        image_data: [{"chunk_id": str, "placeholder": str, "image_bytes": bytes,
                      "ext": str, "sheet_name": str, "row": int, "sort_order": int}, ...]
    """
    from openpyxl import load_workbook

    meta = base_metadata or {}
    chunks: List[dict] = []
    image_data: List[dict] = []
    chunk_index = 0

    wb = load_workbook(io.BytesIO(file_content), data_only=True)

    for sheet_idx, sheet_name in enumerate(wb.sheetnames):
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue

        # 确定本 sheet 使用的列配置
        sheet_cfg = None
        if column_config and sheet_name in column_config:
            sheet_cfg = column_config[sheet_name]  # [{original, alias, type?}, ...]

        # 找到 header 行
        expected_cols = [c["original"] for c in sheet_cfg] if sheet_cfg else None
        header_row_idx = _find_header_row(rows, expected_cols)
        headers = rows[header_row_idx]

        # 构建列名 -> 列索引映射
        col_idx_map: dict = {}
        for i, h in enumerate(headers):
            if h is not None:
                col_idx_map[str(h).strip()] = i

        # 确定实际使用的列
        image_cols: set = set()
        if sheet_cfg is not None:
            valid_cfg = [c for c in sheet_cfg if c["original"] in col_idx_map]
            if not valid_cfg:
                continue
            orig_cols = [c["original"] for c in valid_cfg]
            alias_map = {c["original"]: c["alias"] or c["original"] for c in valid_cfg}
            image_cols = {c["original"] for c in valid_cfg if c.get("type") == "image"}
        else:
            orig_cols = [str(h).strip() for h in headers if h is not None]
            alias_map = {c: c for c in orig_cols}

        # 提取图片映射: data_row_idx -> {col_name: (image_bytes, ext, placeholder)}
        image_map: dict = {}
        for img in ws._images:
            img_row = img.anchor._from.row  # 0-based from worksheet start
            img_col = img.anchor._from.col  # 0-based

            if img_row <= header_row_idx:
                continue  # 图片在 header 或之前，忽略

            data_row_idx = img_row - header_row_idx - 1

            # 获取图片字节（openpyxl 3.1+ 中 img.ref 为 BytesIO）
            if hasattr(img.ref, "read"):
                img.ref.seek(0)
                img_bytes = img.ref.read()
            else:
                continue

            ext = (img.format or "png").lower()
            placeholder = f"<<IMAGE:{uuid.uuid4().hex[:8]}>>"

            # 找到该列对应的列名
            col_name = None
            for name, idx in col_idx_map.items():
                if idx == img_col:
                    col_name = name
                    break

            if col_name and col_name in image_cols:
                if data_row_idx not in image_map:
                    image_map[data_row_idx] = {}
                image_map[data_row_idx][col_name] = (img_bytes, ext, placeholder)

        # 数据行（header 之后），过滤全空行
        data_rows = rows[header_row_idx + 1:]
        data_rows = [
            r for r in data_rows
            if any(c is not None and str(c).strip() for c in r)
        ]

        has_image_column = bool(image_cols)

        if has_image_column:
            # 逐行切分：1 行 = 1 个 chunk（确保每行图片独立归属）
            for data_row_idx, row_data in enumerate(data_rows):
                chunk_id = str(uuid.uuid4())
                parts = []
                row_images = {}

                for col_name in orig_cols:
                    col_idx = col_idx_map[col_name]
                    val = row_data[col_idx] if col_idx < len(row_data) else None

                    if col_name in image_cols:
                        if data_row_idx in image_map and col_name in image_map[data_row_idx]:
                            img_bytes, ext, placeholder = image_map[data_row_idx][col_name]
                            parts.append(f"{alias_map[col_name]}={placeholder}")
                            row_images[col_name] = (img_bytes, ext, placeholder)
                        else:
                            parts.append(f"{alias_map[col_name]}=[无图片]")
                    else:
                        if val is not None:
                            val_str = str(val).strip()
                            if val_str:
                                parts.append(f"{alias_map[col_name]}={val_str}")

                if not parts:
                    continue

                content = f"文件：{file_name}  Sheet：{sheet_name}\n" + ", ".join(parts)

                chunks.append({
                    "chunk_id": chunk_id,
                    "content": content,
                    "metadata": {
                        **meta,
                        "chunk_index": chunk_index,
                        "file_name": file_name,
                        "sheet_name": sheet_name,
                        "source": "excel",
                        "chunk_strategy": "excel_image_rows",
                        "row_start": data_row_idx,
                        "row_end": data_row_idx,
                    },
                })

                # 收集 image_data（由调用方在 chunk 入库后上传 OSS）
                for col_name, (img_bytes, ext, placeholder) in row_images.items():
                    image_data.append({
                        "chunk_id": chunk_id,
                        "placeholder": placeholder,
                        "image_bytes": img_bytes,
                        "ext": ext,
                        "sheet_name": sheet_name,
                        "row": data_row_idx,
                        "sort_order": 0,
                    })

                chunk_index += 1
        else:
            # 按 rows_per_chunk 聚合切分（纯文本模式）
            total_rows = len(data_rows)
            start = 0
            while start < total_rows:
                end = min(start + rows_per_chunk, total_rows)
                batch_rows = data_rows[start:end]

                lines = []
                for row_data in batch_rows:
                    parts = []
                    for col_name in orig_cols:
                        col_idx = col_idx_map[col_name]
                        val = row_data[col_idx] if col_idx < len(row_data) else None
                        if val is not None:
                            val_str = str(val).strip()
                            if val_str:
                                parts.append(f"{alias_map[col_name]}={val_str}")
                    if parts:
                        lines.append(", ".join(parts))

                if not lines:
                    start = end
                    continue

                content = f"文件：{file_name}  Sheet：{sheet_name}\n" + "\n".join(lines)
                chunk_id = str(uuid.uuid4())

                chunks.append({
                    "chunk_id": chunk_id,
                    "content": content,
                    "metadata": {
                        **meta,
                        "chunk_index": chunk_index,
                        "file_name": file_name,
                        "sheet_name": sheet_name,
                        "source": "excel",
                        "chunk_strategy": "excel_rows",
                        "row_start": start,
                        "row_end": end - 1,
                    },
                })
                chunk_index += 1
                start = end

    return chunks, image_data
