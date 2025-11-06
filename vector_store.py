"""
Vector Store Module
向量存儲與檢索系統，支援高效相似度搜索
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS
from embedding import TextEmbedding
import os


class VectorStore:
    """
    向量存儲管理器
    
    功能:
    - 存儲文檔向量
    - 執行相似度檢索
    - 支援多種向量數據庫後端 (Chroma, FAISS)
    """
    
    def __init__(
        self,
        embedding: TextEmbedding,
        store_type: str = "chroma",
        persist_directory: Optional[str] = None
    ):
        """
        初始化向量存儲
        
        Args:
            embedding: 文本嵌入器實例
            store_type: 向量數據庫類型 ("chroma" 或 "faiss")
            persist_directory: 持久化存儲路徑
        """
        self.embedding = embedding
        self.store_type = store_type
        self.persist_directory = persist_directory or f"{store_type}_db"
        self.vector_store = None
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        創建向量存儲並索引文檔
        
        Args:
            documents: 待索引的文檔列表
        """
        if self.store_type == "chroma":
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding.embeddings,
                persist_directory=self.persist_directory
            )
        elif self.store_type == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding.embeddings
            )
            # FAISS 需要手動保存
            if self.persist_directory:
                self.vector_store.save_local(self.persist_directory)
        else:
            raise ValueError(f"不支援的向量存儲類型: {self.store_type}")
    
    def load_vector_store(self) -> None:
        """
        從持久化存儲載入向量數據庫
        """
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"向量存儲路徑不存在: {self.persist_directory}")
        
        if self.store_type == "chroma":
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding.embeddings
            )
        elif self.store_type == "faiss":
            self.vector_store = FAISS.load_local(
                self.persist_directory,
                self.embedding.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            raise ValueError(f"不支援的向量存儲類型: {self.store_type}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        向現有向量存儲添加新文檔
        
        Args:
            documents: 待添加的文檔列表
        """
        if self.vector_store is None:
            raise RuntimeError("向量存儲尚未初始化，請先調用 create_vector_store 或 load_vector_store")
        
        self.vector_store.add_documents(documents)
        
        # 保存更新
        if self.store_type == "faiss" and self.persist_directory:
            self.vector_store.save_local(self.persist_directory)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        執行相似度檢索
        
        Args:
            query: 查詢文本
            k: 返回最相似的 k 個文檔
            score_threshold: 相似度分數閾值（僅部分存儲支持）
            
        Returns:
            最相似的文檔列表
        """
        if self.vector_store is None:
            raise RuntimeError("向量存儲尚未初始化")
        
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple[Document, float]]:
        """
        執行相似度檢索並返回分數
        
        Args:
            query: 查詢文本
            k: 返回最相似的 k 個文檔
            
        Returns:
            (文檔, 相似度分數) 元組列表
        """
        if self.vector_store is None:
            raise RuntimeError("向量存儲尚未初始化")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
    
    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """
        將向量存儲轉換為檢索器接口
        
        Args:
            search_kwargs: 檢索參數，例如 {"k": 4}
            
        Returns:
            LangChain 檢索器對象
        """
        if self.vector_store is None:
            raise RuntimeError("向量存儲尚未初始化")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
