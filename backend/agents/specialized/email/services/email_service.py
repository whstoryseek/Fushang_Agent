# -*- coding: utf-8 -*-
"""
Email Service
Email operations and tools
"""

from langchain_core.tools import tool


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    发送一封邮件。
    
    Args:
        to: 收件人邮箱地址
        subject: 邮件主题
        body: 邮件正文内容
        
    Returns:
        发送结果确认信息
    """
    print(f"[Email Agent] Sending email to: {to}")
    print(f"[Email Agent] Subject: {subject}")
    
    return f"Email sent successfully to {to}"


@tool
def check_inbox(folder: str = "inbox", limit: int = 10) -> str:
    """
    查看邮箱文件夹中的邮件。
    
    Args:
        folder: 要查看的邮箱文件夹，例如收件箱、已发送、草稿箱
        limit: 最多返回的邮件数量
        
    Returns:
        邮件列表
    """
    print(f"[Email Agent] Checking {folder}, limit: {limit}")
    
    return f"Found 3 emails in {folder}:\n1. Email 1...\n2. Email 2...\n3. Email 3..."


@tool
def search_emails(query: str, limit: int = 10) -> str:
    """
    按关键词搜索邮件。
    
    Args:
        query: 搜索关键词
        limit: 最多返回的结果数量
        
    Returns:
        邮件搜索结果
    """
    print(f"[Email Agent] Searching emails for: {query}")
    
    return f"Found 2 emails matching '{query}':\n1. Email 1...\n2. Email 2..."


def get_email_tools():
    """
    Get list of email tools
    
    Returns:
        List of email tool functions
    """
    return [send_email, check_inbox, search_emails]


# Agent metadata for supervisor
EMAIL_AGENT_INFO = {
    "name": "email_agent",
    "display_name": "邮件智能体",
    "description": "专门处理邮件发送、查收、搜索等邮件相关任务",
    "capabilities": [
        "发送邮件",
        "查收邮件",
        "搜索邮件",
        "管理邮箱"
    ],
    "keywords": ["邮件", "发送", "查收", "收件箱", "草稿箱"]
}
