# -*- coding: utf-8 -*-
"""
本地文件存储服务（替代阿里云 OSS）
文件内容存入 PostgreSQL BYTEA，通过本地代理 URL 供前端访问
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LocalStorageService:
    """本地文件存储服务（基于 PostgreSQL BYTEA）"""

    def upload_bytes(self, file_key: str, file_content: bytes) -> str:
        """用完整 file_key 上传，返回 file_key"""
        try:
            from app.db import get_file_storage_repository
            repo = get_file_storage_repository()
            repo.insert(file_key, file_content)
            logger.info(f"本地存储上传成功: {file_key}, size={len(file_content)}")
            return file_key
        except Exception as e:
            logger.error(f"本地存储上传失败: {file_key}, error={e}")
            raise Exception(f"本地存储上传失败: {e}")

    def upload_file(self, category_name: str, file_name: str, file_content: bytes) -> str:
        """上传文件，file_key 格式为 category_name/file_name"""
        file_key = f"{category_name}/{file_name}"
        return self.upload_bytes(file_key, file_content)

    def get_object_bytes(self, file_key: str) -> bytes:
        """下载文件字节内容（后端专用）"""
        try:
            from app.db import get_file_storage_repository
            repo = get_file_storage_repository()
            data = repo.get_bytes(file_key)
            if data is None:
                raise Exception(f"文件不存在: {file_key}")
            logger.info(f"本地存储下载成功: {file_key}, size={len(data)}")
            return data
        except Exception as e:
            logger.error(f"本地存储下载失败: {file_key}, error={e}")
            raise Exception(f"本地存储下载失败: {e}")

    def delete_objects(self, file_keys: list) -> int:
        """批量删除文件，返回删除数量"""
        if not file_keys:
            return 0
        try:
            from app.db import get_file_storage_repository
            repo = get_file_storage_repository()
            count = repo.delete_by_keys(file_keys)
            logger.info(f"本地存储批量删除: 请求 {len(file_keys)} 个, 成功 {count} 个")
            return count
        except Exception as e:
            logger.error(f"本地存储批量删除失败: {e}")
            return 0

    def get_presigned_url(self, file_key: str, expires: int = 3600) -> str:
        """生成本地代理 URL（供前端访问）"""
        return f"/api/v1/files/{file_key}"

    def get_presigned_url_by_category(
        self, category_name: str, file_name: str, expires: int = 3600
    ) -> str:
        """通过类目名和文件名生成本地代理 URL"""
        file_key = f"{category_name}/{file_name}"
        return self.get_presigned_url(file_key, expires)


_instance: Optional[LocalStorageService] = None


def get_oss_service() -> LocalStorageService:
    """兼容旧接口名，返回本地存储服务"""
    global _instance
    if _instance is None:
        _instance = LocalStorageService()
    return _instance
