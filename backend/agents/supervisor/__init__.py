# -*- coding: utf-8 -*-
from .graph import create_supervisor_agent

_agent = None


def get_supervisor_agent():
    """获取 Supervisor Agent 实例（延迟初始化，注入持久化 checkpointer）"""
    global _agent
    if _agent is None:
        from app.core.checkpointer import get_checkpointer
        checkpointer = get_checkpointer()
        _agent = create_supervisor_agent(checkpointer=checkpointer)
    return _agent


__all__ = ["get_supervisor_agent", "create_supervisor_agent"]
