"""
全局配置 - 从环境变量读取
请设置环境变量 DEEPSEEK_API_KEY，或复制 .env.example 为 .env 后填入
绝对不要把 API Key 硬编码到代码里或提交到 Git
"""
from __future__ import annotations
import os

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # 必须从环境变量读取，禁止硬编码
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "15"))

# Chroma 向量库
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sales_knowledge")

# 服务
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

# 检索
DEFAULT_TOP_K = 3
SIMILARITY_THRESHOLD = 0.05


def validate():
    """启动时检查必要配置"""
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "❌ DEEPSEEK_API_KEY 未设置！\n"
            "请在 .env 文件中填入你的 DeepSeek API Key，\n"
            "或在系统环境变量中设置 DEEPSEEK_API_KEY。\n"
            "申请地址：https://platform.deepseek.com/"
        )
