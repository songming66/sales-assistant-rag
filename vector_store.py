"""
Chroma 向量库封装
负责：初始化、灌入知识、语义检索
"""
import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_PATH, COLLECTION_NAME


class VectorStore:
    """Chroma 向量库单例"""

    def __init__(self):
        print("📚 初始化 Chroma 向量库...")
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        # 默认 ONNX embedding: all-MiniLM-L6-v2 (384维)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ 向量库就绪，当前 {self.collection.count()} 条知识")

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def add_documents(self, chunks: list):
        """灌入文档片段到向量库"""
        if not chunks:
            return
        self.collection.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"chunk_id": i, "source": "sales_knowledge"} for i in range(len(chunks))]
        )
        print(f"   已灌入 {len(chunks)} 条知识")

    def search(self, query: str, top_k: int = 3) -> tuple[list, list]:
        """
        语义检索
        返回: (docs, scores)  scores 是相似度 0~1
        """
        if not query.strip():
            return [], []
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        # cosine distance (越小越相似) → similarity (越大越相似)
        scores = [1 - d for d in distances]
        return docs, scores

    def count(self) -> int:
        return self.collection.count()

    def get_all(self, limit: int = 20) -> list:
        """查看知识库前 N 条"""
        data = self.collection.get(limit=limit)
        return [
            {"id": data["ids"][i], "content": data["documents"][i]}
            for i in range(len(data["ids"]))
        ]

    def clear(self):
        """清空 collection"""
        self.client.delete_collection(COLLECTION_NAME)
        print("🗑️  知识库已清空")
