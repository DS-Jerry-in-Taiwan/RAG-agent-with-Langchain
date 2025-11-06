"""
PDF Document Loader Module
載入並處理 PDF 文件，將其切分為可管理的文本段落
"""

from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFDocumentLoader:
    """
    PDF 文件載入器
    
    功能:
    - 解析 PDF 文件內容
    - 將文本切分為結構化段落
    - 保留元數據（頁碼、來源等）
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化文件載入器
        
        Args:
            chunk_size: 每個文本段落的字符數上限
            chunk_overlap: 段落之間的重疊字符數，保持上下文連貫性
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        載入並處理單個 PDF 文件
        
        Args:
            file_path: PDF 文件路徑
            
        Returns:
            切分後的文檔段落列表
        """
        # 使用 PyPDFLoader 載入 PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # 切分文檔為段落
        chunks = self.text_splitter.split_documents(documents)
        
        return chunks
    
    def load_multiple_pdfs(self, file_paths: List[str]) -> List[Document]:
        """
        載入並處理多個 PDF 文件
        
        Args:
            file_paths: PDF 文件路徑列表
            
        Returns:
            所有文檔的切分段落列表
        """
        all_chunks = []
        for file_path in file_paths:
            chunks = self.load_pdf(file_path)
            all_chunks.extend(chunks)
        
        return all_chunks
