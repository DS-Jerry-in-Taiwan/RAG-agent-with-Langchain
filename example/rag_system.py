"""
RAG System Module
檢索增強生成系統，整合向量檢索與大型語言模型
"""

from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from vector_store import VectorStore
import os


class RAGSystem:
    """
    RAG (Retrieval-Augmented Generation) 系統
    
    功能:
    - 整合向量檢索與 LLM
    - 將檢索到的文檔作為上下文輸入
    - 生成基於知識庫的精確回答
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        retrieval_k: int = 4
    ):
        """
        初始化 RAG 系統
        
        Args:
            vector_store: 向量存儲實例
            model_name: OpenAI 模型名稱
            temperature: 生成溫度 (0.0 = 確定性, 1.0 = 隨機性)
            retrieval_k: 檢索文檔數量
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 環境變量未設定")
        
        self.vector_store = vector_store
        self.retrieval_k = retrieval_k
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )
        
        # 設置檢索器
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": retrieval_k}
        )
        
        # 創建 RAG 鏈
        self._setup_qa_chain()
    
    def _setup_qa_chain(self) -> None:
        """
        設置問答鏈
        """
        # 自定義提示詞模板
        template = """使用以下檢索到的上下文資訊來回答問題。
如果你不知道答案，就說不知道，不要試圖編造答案。
請用繁體中文回答，並保持回答簡潔、準確。

上下文資訊：
{context}

問題：{query}

詳細回答："""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "query"]
        )
        
        # 創建 RetrievalQA 鏈
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def query(self, question: str) -> dict:
        """
        執行問答查詢
        
        Args:
            question: 用戶問題
            
        Returns:
            包含答案和來源文檔的字典
            {
                "answer": "回答文本",
                "source_documents": [Document, ...]
            }
        """
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    
    def query_with_context(self, question: str) -> dict:
        """
        執行問答並返回詳細上下文資訊
        
        Args:
            question: 用戶問題
            
        Returns:
            包含答案、來源文檔和元數據的字典
        """
        result = self.query(question)
        
        # 提取來源文檔的詳細資訊
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page", "未知")
            })
        
        return {
            "answer": result["answer"],
            "sources": sources,
            "num_sources": len(sources)
        }
    
    def chat(self, question: str) -> str:
        """
        簡化的聊天接口，僅返回答案文本
        
        Args:
            question: 用戶問題
            
        Returns:
            答案文本
        """
        result = self.query(question)
        return result["answer"]


class SimpleRAG:
    """
    簡化的 RAG 系統，適合快速原型開發
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        初始化簡化 RAG 系統
        
        Args:
            vector_store: 向量存儲實例
        """
        self.vector_store = vector_store
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 環境變量未設定")
        
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key
        )
    
    def answer(self, question: str, k: int = 3) -> str:
        """
        基於檢索的簡單問答
        
        Args:
            question: 用戶問題
            k: 檢索文檔數量
            
        Returns:
            答案文本
        """
        # 檢索相關文檔
        docs = self.vector_store.similarity_search(question, k=k)
        
        # 組合上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 構建提示
        prompt = f"""基於以下上下文資訊回答問題，使用繁體中文：

上下文：
{context}

問題：{question}

回答："""
        
        # 調用 LLM
        response = self.llm.invoke(prompt)
        
        return response.content
