"""
v5.0 配置 - 从环境变量读取
请在 .env 文件中填入 DEEPSEEK_API_KEY
绝对不要把 API Key 硬编码到代码里
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

# 自动加载 .env 文件（如果存在）
# 优先级：系统环境变量 > .env 文件
load_dotenv()

# ─── DeepSeek（兼容 OpenAI 接口）───
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "30"))

# ─── 向量库 ───
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ecommerce_service")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Chroma 默认

# ─── 服务 ───
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

# ─── 检索 ───
DEFAULT_TOP_K = 3


def validate():
    """启动时检查必要配置"""
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "❌ DEEPSEEK_API_KEY 未设置！\n"
            "请在 .env 文件中填入你的 DeepSeek API Key，\n"
            "或在系统环境变量中设置 DEEPSEEK_API_KEY。\n"
            "申请地址：https://platform.deepseek.com/"
        )


if __name__ == "__main__":
    print("当前配置：")
    print(f"  DEEPSEEK_BASE_URL: {DEEPSEEK_BASE_URL}")
    print(f"  DEEPSEEK_MODEL:    {DEEPSEEK_MODEL}")
    print(f"  CHROMA_PATH:       {CHROMA_PATH}")
    print(f"  COLLECTION_NAME:   {COLLECTION_NAME}")
    print(f"  APP_PORT:          {APP_PORT}")
    print(f"  API Key 长度:      {len(DEEPSEEK_API_KEY)} 字符")
