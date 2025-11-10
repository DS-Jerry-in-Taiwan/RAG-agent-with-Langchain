"""
Embedding Module
將文本轉換為高維度向量表示，保留語意資訊
"""

from typing import List
from langchain_openai import OpenAIEmbeddings
import os


class TextEmbedding:
    """
    文本嵌入向量化器
    
    功能:
    - 將文本轉換為高維度向量
    - 保留語意資訊與上下文關聯性
    - 支援批次處理提升效率
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        初始化嵌入模型
        
        Args:
            model_name: OpenAI 嵌入模型名稱
                      - text-embedding-3-small: 性能與成本平衡 (1536 維度)
                      - text-embedding-3-large: 高性能 (3072 維度)
                      - text-embedding-ada-002: 傳統模型 (1536 維度)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 環境變量未設定，請在 .env 文件中配置")
        
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )
        self.model_name = model_name
    
    def embed_text(self, text: str) -> List[float]:
        """
        將單個文本轉換為向量
        
        Args:
            text: 輸入文本
            
        Returns:
            文本的向量表示
        """
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批次處理多個文本的向量化
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        return self.embeddings.embed_documents(texts)
    
    def get_embedding_dimension(self) -> int:
        """
        獲取嵌入向量維度
        
        Returns:
            向量維度
        """
        if "3-small" in self.model_name or "ada-002" in self.model_name:
            return 1536
        elif "3-large" in self.model_name:
            return 3072
        else:
            return 1536  # 預設值
