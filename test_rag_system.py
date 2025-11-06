"""
單元測試 - PDF 智能搜尋引擎
測試文檔載入、嵌入、向量存儲和 RAG 系統功能
"""

import unittest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from document_loader import PDFDocumentLoader
from embedding import TextEmbedding
from vector_store import VectorStore
from langchain_core.documents import Document


class TestPDFDocumentLoader(unittest.TestCase):
    """測試 PDF 文檔載入器"""
    
    def test_initialization(self):
        """測試初始化"""
        loader = PDFDocumentLoader(chunk_size=500, chunk_overlap=100)
        self.assertEqual(loader.chunk_size, 500)
        self.assertEqual(loader.chunk_overlap, 100)
        self.assertIsNotNone(loader.text_splitter)
    
    def test_default_parameters(self):
        """測試默認參數"""
        loader = PDFDocumentLoader()
        self.assertEqual(loader.chunk_size, 1000)
        self.assertEqual(loader.chunk_overlap, 200)
    
    @patch('document_loader.PyPDFLoader')
    def test_load_pdf(self, mock_pdf_loader):
        """測試 PDF 載入"""
        # Mock PDF loader
        mock_doc = Document(page_content="測試內容", metadata={"page": 1})
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_pdf_loader.return_value = mock_loader_instance
        
        loader = PDFDocumentLoader(chunk_size=100, chunk_overlap=20)
        results = loader.load_pdf("test.pdf")
        
        # 驗證調用
        mock_pdf_loader.assert_called_once_with("test.pdf")
        mock_loader_instance.load.assert_called_once()
        self.assertIsInstance(results, list)
    
    @patch('document_loader.PyPDFLoader')
    def test_load_multiple_pdfs(self, mock_pdf_loader):
        """測試多文件載入"""
        mock_doc = Document(page_content="測試內容", metadata={"page": 1})
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_pdf_loader.return_value = mock_loader_instance
        
        loader = PDFDocumentLoader()
        results = loader.load_multiple_pdfs(["test1.pdf", "test2.pdf"])
        
        # 驗證調用次數
        self.assertEqual(mock_pdf_loader.call_count, 2)
        self.assertIsInstance(results, list)


class TestTextEmbedding(unittest.TestCase):
    """測試文本嵌入"""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('embedding.OpenAIEmbeddings')
    def test_initialization(self, mock_embeddings):
        """測試初始化"""
        embedding = TextEmbedding(model_name="text-embedding-3-small")
        self.assertEqual(embedding.model_name, "text-embedding-3-small")
        mock_embeddings.assert_called_once()
    
    def test_initialization_without_api_key(self):
        """測試無 API Key 時的錯誤處理"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                TextEmbedding()
            self.assertIn("OPENAI_API_KEY", str(context.exception))
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('embedding.OpenAIEmbeddings')
    def test_get_embedding_dimension(self, mock_embeddings):
        """測試獲取嵌入維度"""
        embedding = TextEmbedding(model_name="text-embedding-3-small")
        dim = embedding.get_embedding_dimension()
        self.assertEqual(dim, 1536)
        
        embedding_large = TextEmbedding(model_name="text-embedding-3-large")
        dim_large = embedding_large.get_embedding_dimension()
        self.assertEqual(dim_large, 3072)


class TestVectorStore(unittest.TestCase):
    """測試向量存儲"""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('embedding.OpenAIEmbeddings')
    def setUp(self, mock_embeddings):
        """設置測試環境"""
        self.mock_embedding = Mock(spec=TextEmbedding)
        self.mock_embedding.embeddings = Mock()
    
    def test_initialization(self):
        """測試初始化"""
        vector_store = VectorStore(
            embedding=self.mock_embedding,
            store_type="chroma",
            persist_directory="test_db"
        )
        self.assertEqual(vector_store.store_type, "chroma")
        self.assertEqual(vector_store.persist_directory, "test_db")
    
    def test_invalid_store_type(self):
        """測試無效的存儲類型"""
        vector_store = VectorStore(
            embedding=self.mock_embedding,
            store_type="invalid",
            persist_directory="test_db"
        )
        
        mock_doc = Document(page_content="測試", metadata={})
        with self.assertRaises(ValueError):
            vector_store.create_vector_store([mock_doc])
    
    @patch('vector_store.Chroma')
    def test_create_chroma_store(self, mock_chroma):
        """測試創建 Chroma 存儲"""
        vector_store = VectorStore(
            embedding=self.mock_embedding,
            store_type="chroma"
        )
        
        mock_docs = [Document(page_content="測試", metadata={})]
        vector_store.create_vector_store(mock_docs)
        
        mock_chroma.from_documents.assert_called_once()
    
    def test_search_without_initialization(self):
        """測試未初始化時的搜索錯誤"""
        vector_store = VectorStore(
            embedding=self.mock_embedding,
            store_type="chroma"
        )
        
        with self.assertRaises(RuntimeError):
            vector_store.similarity_search("測試查詢")


class TestIntegration(unittest.TestCase):
    """集成測試"""
    
    @patch('document_loader.PyPDFLoader')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('embedding.OpenAIEmbeddings')
    def test_document_to_vector_pipeline(self, mock_embeddings, mock_pdf_loader):
        """測試文檔到向量的完整流程"""
        # Mock PDF loader
        mock_doc = Document(
            page_content="這是一份關於機器學習的技術文件。" * 10,
            metadata={"page": 1, "source": "test.pdf"}
        )
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_pdf_loader.return_value = mock_loader_instance
        
        # 1. 載入文檔
        loader = PDFDocumentLoader(chunk_size=100, chunk_overlap=20)
        documents = loader.load_pdf("test.pdf")
        
        # 驗證文檔載入
        self.assertIsInstance(documents, list)
        self.assertTrue(len(documents) > 0)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('embedding.OpenAIEmbeddings')
    def test_embedding_dimension_consistency(self, mock_embeddings):
        """測試嵌入維度一致性"""
        embedding = TextEmbedding(model_name="text-embedding-3-small")
        dim = embedding.get_embedding_dimension()
        self.assertEqual(dim, 1536)


def run_tests():
    """運行所有測試"""
    # 創建測試套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加測試
    suite.addTests(loader.loadTestsFromTestCase(TestPDFDocumentLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestTextEmbedding))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorStore))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 運行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    result = run_tests()
    
    # 輸出測試結果摘要
    print("\n" + "="*60)
    print("測試摘要")
    print("="*60)
    print(f"運行測試: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"錯誤: {len(result.errors)}")
    print("="*60)
    
    # 返回退出碼
    exit(0 if result.wasSuccessful() else 1)
