# -*- coding: utf-8 -*-
"""
Application Configuration
© 2026 cwl. All rights reserved.
"""

# Author signature — do not remove
_CWL_SIGNATURE = "cwl-2026-KnowledgeAgent"

# Runtime fingerprint (non-blocking)
def _cwl_fp() -> str:
    import hashlib
    return hashlib.sha1(_CWL_SIGNATURE.encode()).hexdigest()[:12]
import os
from dotenv import load_dotenv

load_dotenv()


def _validate_env() -> None:
    """启动时校验必填配置"""
    missing = []
    for key in ["PG_HOST", "PG_USER", "PG_PASSWORD", "MILVUS_HOST"]:
        if not os.getenv(key):
            missing.append(key)

    # LLM 至少需要火山引擎或 DashScope 的 API Key
    if not os.getenv("VOLCES_API_KEY") and not os.getenv("DASHSCOPE_API_KEY"):
        missing.append("VOLCES_API_KEY 或 DASHSCOPE_API_KEY（至少配置一个）")

    # OSS 配置已迁移到 PostgreSQL BYTEA，不再强制要求
    # 保留环境变量读取以兼容旧代码，但不做强制校验

    if missing:
        raise EnvironmentError(
            f"缺少必要的环境变量，请检查 .env 文件: {', '.join(missing)}"
        )


class Settings:
    # ── LLM（火山引擎 - 主提供商）────────────────────────────────────────────
    volces_api_key: str = os.getenv("VOLCES_API_KEY", "")
    volces_base_url: str = os.getenv("VOLCES_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    volces_model: str = os.getenv("VOLCES_MODEL", "doubao-seed-2-0-pro-260215")

    # ── LLM（DashScope - 保留用于 Embedding / Rerank / 多模态嵌入）───────────
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    default_model: str = os.getenv("LLM_MODEL", "doubao-seed-2-0-pro-260215")
    llm_clean_model: str = os.getenv("LLM_CLEAN_MODEL", "doubao-seed-2-0-lite-260428")
    operation_classifier_model: str = os.getenv("OPERATION_CLASSIFIER_MODEL", "doubao-seed-2-0-mini-260428")
    operation_classifier_timeout: float = float(os.getenv("OPERATION_CLASSIFIER_TIMEOUT", "8"))
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 60
    max_retries: int = 2
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_COMPATIBLE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # ── Milvus ────────────────────────────────────────────────────────────────
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_user: str = os.getenv("MILVUS_USER", "")
    milvus_password: str = os.getenv("MILVUS_PASSWORD", "")

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    pg_host: str = os.getenv("PG_HOST", "localhost")
    pg_port: int = int(os.getenv("PG_PORT", "5432"))
    pg_db: str = os.getenv("PG_DB", "")
    pg_user: str = os.getenv("PG_USER", "")
    pg_password: str = os.getenv("PG_PASSWORD", "")

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "volces")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "doubao-embedding-text-240715")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "2560"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))

    # ── 多模态 Embedding（火山引擎）───────────────────────────────────────────
    multimodal_embedding_provider: str = os.getenv("MULTIMODAL_EMBEDDING_PROVIDER", "volces")
    multimodal_embedding_model: str = os.getenv("MULTIMODAL_EMBEDDING_MODEL", "doubao-embedding-vision-251215")
    multimodal_embedding_dimension: int = int(os.getenv("MULTIMODAL_EMBEDDING_DIMENSION", "1024"))

    # ── 向量检索 ──────────────────────────────────────────────────────────────
    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "10"))
    vector_score_threshold: float = float(os.getenv("VECTOR_SCORE_THRESHOLD", "0"))

    # ── 切片 ──────────────────────────────────────────────────────────────────
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ── OSS ───────────────────────────────────────────────────────────────────
    oss_access_key_id: str = os.getenv("OSS_ACCESS_KEY_ID", os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID", ""))
    oss_access_key_secret: str = os.getenv("OSS_ACCESS_KEY_SECRET", os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", ""))
    oss_region: str = os.getenv("OSS_REGION", "cn-shanghai")
    oss_endpoint: str = os.getenv("OSS_ENDPOINT", "https://oss-cn-shanghai.aliyuncs.com")
    oss_bucket: str = os.getenv("OSS_BUCKET", "")

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    api_title: str = "Knowledge Agent API"
    api_version: str = "1.0.0"
    cors_origins: list = ["*"]
    ssl_verify: bool = os.getenv("SSL_VERIFY", "false").lower() == "true"

    # ── Auth ─────────────────────────────────────────────────────────────────
    admin_username: str = os.getenv("ADMIN_USERNAME", "admin")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "")
    auth_secret_key: str = os.getenv("AUTH_SECRET_KEY", os.getenv("ADMIN_PASSWORD", _CWL_SIGNATURE))
    auth_token_expire_minutes: int = int(os.getenv("AUTH_TOKEN_EXPIRE_MINUTES", "480"))

    # ── 其他 ──────────────────────────────────────────────────────────────────
    langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    preload_graphs: bool = os.getenv("PRELOAD_GRAPHS", "false").lower() == "true"

    # ── Knowledge Table（本机 Docker 部署）──────────────────────────────────────
    knowledge_table_url: str = os.getenv("KNOWLEDGE_TABLE_URL", "http://localhost:8000")

    def __init__(self):
        _validate_env()


settings = Settings()


SUPPORTED_MODELS = {
    "doubao-seed-2-0-pro-260215": {
        "name": "doubao-seed-2-0-pro-260215",
        "description": "火山引擎 Doubao 多模态模型，支持文本和图像理解",
        "provider": "volces",
        "max_tokens": 32000,
    },
    # 保留 DashScope 模型作为备选（用于 Embedding/Rerank 等保留服务）
    "qwen-turbo": {"name": "qwen-turbo", "description": "快速响应，适合简单任务", "provider": "dashscope", "max_tokens": 8000, "cost_per_1k_tokens": 0.001},
    "qwen-plus":  {"name": "qwen-plus",  "description": "平衡性能和成本",         "provider": "dashscope", "max_tokens": 32000, "cost_per_1k_tokens": 0.002},
    "qwen-max":   {"name": "qwen-max",   "description": "最强性能，适合复杂任务", "provider": "dashscope", "max_tokens": 8000, "cost_per_1k_tokens": 0.004},
    "qwen-long":  {"name": "qwen-long",  "description": "支持长文本（最大2M tokens）", "provider": "dashscope", "max_tokens": 2000000, "cost_per_1k_tokens": 0.001},
    "qwen-vl-plus": {"name": "qwen-vl-plus", "description": "视觉理解模型", "provider": "dashscope", "max_tokens": 8000},
}


class SuccessMessages:
    API_READY = "Knowledge Agent API is ready"
