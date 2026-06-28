"""
v5.0 向量库 - 基于 LangChain + Chroma
负责：embedding、灌入知识、语义检索
Embedding: chroma 自带 ONNX (all-MiniLM-L6-v2, 384维)，零额外下载
"""
from __future__ import annotations
import os
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings  # 备用

# 优先用 chroma 自带的 ONNX embedding
try:
    from chromadb.utils import embedding_functions
    _DEFAULT_EF = embedding_functions.DefaultEmbeddingFunction()
    _HAS_DEFAULT = True
except Exception as e:
    print(f"⚠️  chroma 默认 embedding 不可用: {e}")
    _DEFAULT_EF = None
    _HAS_DEFAULT = False


from config import CHROMA_PATH, COLLECTION_NAME


class ChromaONNXEmbeddings:
    """
    把 chroma 的 ONNX embedding 包装成 LangChain Embeddings 接口
    解决 sentence-transformers 装不上的问题
    """
    def __init__(self, ef=None):
        self.ef = ef or _DEFAULT_EF

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.ef(texts).tolist() if hasattr(self.ef(texts), 'tolist') else self.ef(texts)

    def embed_query(self, text: str) -> List[float]:
        result = self.ef([text])
        return result.tolist()[0] if hasattr(result, 'tolist') else result[0]


class VectorStore:
    """LangChain + Chroma 向量库封装"""

    def __init__(self):
        if not _HAS_DEFAULT:
            raise RuntimeError(
                "chroma ONNX embedding 初始化失败，\n"
                "请检查 chromadb 是否正确安装"
            )

        print(f"📚 初始化 Chroma 向量库（ONNX all-MiniLM-L6-v2）...")
        self.embedding_fn = ChromaONNXEmbeddings(_DEFAULT_EF)
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            persist_directory=CHROMA_PATH,
            collection_metadata={"hnsw:space": "cosine"},
        )
        count = self.vectorstore._collection.count()
        print(f"✅ 向量库就绪，当前 {count} 条知识")

    def is_empty(self) -> bool:
        return self.vectorstore._collection.count() == 0

    def add_documents(self, docs: List[dict]) -> None:
        """
        灌入文档到向量库
        docs: [{"content": "...", "metadata": {...}}, ...]
        """
        if not docs:
            return
        lc_docs = [
            Document(page_content=d["content"], metadata=d.get("metadata", {}))
            for d in docs
        ]
        self.vectorstore.add_documents(lc_docs)
        print(f"   已灌入 {len(lc_docs)} 条对话")

    def search(self, query: str, top_k: int = 3) -> Tuple[List[str], List[float]]:
        """
        语义检索
        返回: (docs, scores)  scores 是相似度 0~1
        """
        if not query.strip():
            return [], []
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=top_k
        )
        docs = [doc.page_content for doc, _ in results]
        scores = [float(score) for _, score in results]
        return docs, scores

    def count(self) -> int:
        return self.vectorstore._collection.count()

    def get_all(self, limit: int = 20) -> List[dict]:
        """查看知识库前 N 条"""
        data = self.vectorstore._collection.get(limit=limit)
        return [
            {"id": data["ids"][i], "content": data["documents"][i]}
            for i in range(len(data["ids"]))
        ]

    def clear(self) -> None:
        """清空 collection"""
        self.vectorstore.delete_collection()
        print("🗑️  知识库已清空")
