# -*- coding: utf-8 -*-
"""
LLM 统一调用服务
使用火山引擎 OpenAI 兼容接口，替代原生的 dashscope.Generation.call()
支持文本生成和多模态（图片输入）生成。
"""

from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.core.config import settings


class LLMService:
    """火山引擎 LLM 调用封装（OpenAI 兼容接口）"""

    def __init__(self):
        self.client = OpenAI(
            base_url=settings.volces_base_url,
            api_key=settings.volces_api_key,
            timeout=settings.timeout,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        disable_thinking: bool = False,
    ) -> str:
        """
        通用文本生成。

        Args:
            messages: DashScope 风格的 messages（含 role/content，多模态时为 list）
            model: 模型名称，默认使用 settings.volces_model
            temperature: 温度
            max_tokens: 最大输出 token 数

        Returns:
            生成的文本内容
        """
        model = model or settings.volces_model
        oa_messages = _convert_messages_to_openai(messages)
        client = self.client

        client_options = {}
        if timeout is not None:
            client_options["timeout"] = timeout
        if max_retries is not None:
            client_options["max_retries"] = max_retries
        if client_options:
            client = self.client.with_options(**client_options)

        request_kwargs = {
            "model": model,
            "messages": oa_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if disable_thinking:
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        response = client.chat.completions.create(
            **request_kwargs,
        )
        return response.choices[0].message.content or ""

    def chat_with_images(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        disable_thinking: bool = False,
    ) -> str:
        """
        多模态生成（支持图片输入）。
        与 chat() 实现相同，因为 OpenAI 兼容接口统一用 content list 处理图片。
        """
        return self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            disable_thinking=disable_thinking,
        )

    def responses_text(
        self,
        input_items: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> str:
        """Use the Volces/OpenAI Responses API for lightweight text tasks."""
        model = model or settings.volces_model
        client = self.client

        client_options = {}
        if timeout is not None:
            client_options["timeout"] = timeout
        if max_retries is not None:
            client_options["max_retries"] = max_retries
        if client_options:
            client = self.client.with_options(**client_options)

        response = client.responses.create(
            model=model,
            input=input_items,
            temperature=temperature,
            max_output_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)


def _convert_messages_to_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 DashScope 风格 messages 转为 OpenAI Chat 格式。"""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if role == "user" and isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if "image" in part:
                    parts.append({"type": "image_url", "image_url": {"url": part["image"]}})
                if "text" in part:
                    parts.append({"type": "text", "text": part["text"]})
            out.append({"role": "user", "content": parts})
        else:
            out.append({"role": role, "content": content if isinstance(content, str) else str(content)})
    return out


# 全局单例
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """获取 LLMService 单例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
