"""
PDF 智能搜尋引擎主程序
提供命令行接口進行文檔處理和問答
"""

import os
import argparse
from dotenv import load_dotenv
from document_loader import PDFDocumentLoader
from embedding import TextEmbedding
from vector_store import VectorStore
from rag_system import RAGSystem


class PDFSearchEngine:
    """
    PDF 智能搜尋引擎主類
    """
    
    def __init__(self, store_type: str = "chroma", persist_dir: str = None):
        """
        初始化搜尋引擎
        
        Args:
            store_type: 向量存儲類型 ("chroma" 或 "faiss")
            persist_dir: 持久化目錄
        """
        load_dotenv()
        
        self.store_type = store_type
        self.persist_dir = persist_dir or f"{store_type}_db"
        
        # 初始化組件
        self.embedding = TextEmbedding(model_name="text-embedding-3-small")
        self.vector_store = VectorStore(
            embedding=self.embedding,
            store_type=store_type,
            persist_directory=self.persist_dir
        )
        self.rag_system = None
    
    def index_pdfs(self, pdf_paths: list, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        索引 PDF 文件
        
        Args:
            pdf_paths: PDF 文件路徑列表
            chunk_size: 段落大小
            chunk_overlap: 段落重疊
        """
        print(f"開始索引 {len(pdf_paths)} 個 PDF 文件...")
        
        # 載入文檔
        loader = PDFDocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = loader.load_multiple_pdfs(pdf_paths)
        
        print(f"載入了 {len(documents)} 個文檔段落")
        
        # 創建向量存儲
        print("正在創建向量索引...")
        self.vector_store.create_vector_store(documents)
        
        print(f"✓ 索引完成！向量存儲保存在: {self.persist_dir}")
    
    def load_index(self):
        """
        載入現有索引
        """
        print(f"從 {self.persist_dir} 載入向量存儲...")
        self.vector_store.load_vector_store()
        print("✓ 向量存儲載入成功")
    
    def initialize_rag(self, model: str = "gpt-3.5-turbo", retrieval_k: int = 4):
        """
        初始化 RAG 系統
        
        Args:
            model: LLM 模型名稱
            retrieval_k: 檢索文檔數量
        """
        print(f"初始化 RAG 系統 (模型: {model}, 檢索數量: {retrieval_k})...")
        self.rag_system = RAGSystem(
            vector_store=self.vector_store,
            model_name=model,
            retrieval_k=retrieval_k
        )
        print("✓ RAG 系統初始化完成")
    
    def search(self, query: str, k: int = 4):
        """
        執行向量檢索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
        
        Returns:
            檢索結果列表
        """
        print(f"\n查詢: {query}")
        print("-" * 60)
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n結果 {i} (相似度: {score:.4f})")
            print(f"來源: {doc.metadata.get('source', '未知')}")
            print(f"頁碼: {doc.metadata.get('page', '未知')}")
            print(f"內容: {doc.page_content[:200]}...")
            print("-" * 60)
        
        return results
    
    def ask(self, question: str):
        """
        執行問答
        
        Args:
            question: 問題
        
        Returns:
            回答文本
        """
        if self.rag_system is None:
            self.initialize_rag()
        
        print(f"\n問題: {question}")
        print("-" * 60)
        
        result = self.rag_system.query_with_context(question)
        
        print(f"回答:\n{result['answer']}")
        print(f"\n參考來源 ({result['num_sources']} 個):")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['source']} (頁 {source['page']})")
        print("-" * 60)
        
        return result['answer']
    
    def interactive_mode(self):
        """
        互動式問答模式
        """
        if self.rag_system is None:
            self.initialize_rag()
        
        print("\n=== 進入互動問答模式 ===")
        print("輸入問題進行查詢，輸入 'exit' 或 'quit' 退出")
        print("-" * 60)
        
        while True:
            try:
                question = input("\n您的問題: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("退出互動模式")
                    break
                
                if not question:
                    continue
                
                self.ask(question)
                
            except KeyboardInterrupt:
                print("\n\n退出互動模式")
                break
            except Exception as e:
                print(f"錯誤: {e}")


def main():
    """
    主函數 - 命令行接口
    """
    parser = argparse.ArgumentParser(
        description="PDF 智能搜尋引擎 - 基於 RAG 的知識庫問答系統"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # index 命令
    index_parser = subparsers.add_parser("index", help="索引 PDF 文件")
    index_parser.add_argument("pdf_files", nargs="+", help="PDF 文件路徑")
    index_parser.add_argument("--store-type", default="chroma", choices=["chroma", "faiss"], help="向量存儲類型")
    index_parser.add_argument("--persist-dir", help="持久化目錄")
    index_parser.add_argument("--chunk-size", type=int, default=1000, help="段落大小")
    index_parser.add_argument("--chunk-overlap", type=int, default=200, help="段落重疊")
    
    # search 命令
    search_parser = subparsers.add_parser("search", help="執行向量檢索")
    search_parser.add_argument("query", help="查詢文本")
    search_parser.add_argument("--store-type", default="chroma", choices=["chroma", "faiss"], help="向量存儲類型")
    search_parser.add_argument("--persist-dir", help="持久化目錄")
    search_parser.add_argument("-k", type=int, default=4, help="返回結果數量")
    
    # ask 命令
    ask_parser = subparsers.add_parser("ask", help="執行問答")
    ask_parser.add_argument("question", help="問題")
    ask_parser.add_argument("--store-type", default="chroma", choices=["chroma", "faiss"], help="向量存儲類型")
    ask_parser.add_argument("--persist-dir", help="持久化目錄")
    ask_parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM 模型")
    ask_parser.add_argument("--retrieval-k", type=int, default=4, help="檢索文檔數量")
    
    # interactive 命令
    interactive_parser = subparsers.add_parser("interactive", help="互動式問答模式")
    interactive_parser.add_argument("--store-type", default="chroma", choices=["chroma", "faiss"], help="向量存儲類型")
    interactive_parser.add_argument("--persist-dir", help="持久化目錄")
    interactive_parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM 模型")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # 創建搜尋引擎實例
    engine = PDFSearchEngine(
        store_type=args.store_type,
        persist_dir=args.persist_dir
    )
    
    # 執行命令
    if args.command == "index":
        engine.index_pdfs(
            pdf_paths=args.pdf_files,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    
    elif args.command == "search":
        engine.load_index()
        engine.search(args.query, k=args.k)
    
    elif args.command == "ask":
        engine.load_index()
        engine.initialize_rag(model=args.model, retrieval_k=args.retrieval_k)
        engine.ask(args.question)
    
    elif args.command == "interactive":
        engine.load_index()
        engine.initialize_rag(model=args.model)
        engine.interactive_mode()


if __name__ == "__main__":
    main()
