"""
PDF 智能搜尋引擎 - 使用示例
展示如何使用 Document Loader、Embedding、Vector Store 和 RAG 系統
"""

import os
from dotenv import load_dotenv
from document_loader import PDFDocumentLoader
from embedding import TextEmbedding
from vector_store import VectorStore
from rag_system import RAGSystem, SimpleRAG


def example_basic_usage():
    """
    基本使用示例：載入 PDF、建立向量存儲、執行問答
    """
    print("=== PDF 智能搜尋引擎 - 基本使用示例 ===\n")
    
    # 載入環境變量
    load_dotenv()
    
    # 1. 載入 PDF 文件
    print("步驟 1: 載入並切分 PDF 文件...")
    loader = PDFDocumentLoader(chunk_size=1000, chunk_overlap=200)
    
    # 假設有一個 PDF 文件
    pdf_path = "data/sample.pdf"
    
    # 檢查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"警告: 示例 PDF 文件不存在: {pdf_path}")
        print("請將您的 PDF 文件放置在 data/ 目錄下")
        return
    
    documents = loader.load_pdf(pdf_path)
    print(f"✓ 成功載入並切分為 {len(documents)} 個段落\n")
    
    # 2. 初始化嵌入模型
    print("步驟 2: 初始化文本嵌入模型...")
    embedding = TextEmbedding(model_name="text-embedding-3-small")
    print(f"✓ 使用模型: text-embedding-3-small\n")
    
    # 3. 創建向量存儲
    print("步驟 3: 創建向量存儲並索引文檔...")
    vector_store = VectorStore(
        embedding=embedding,
        store_type="chroma",
        persist_directory="chroma_db"
    )
    vector_store.create_vector_store(documents)
    print(f"✓ 向量存儲創建完成\n")
    
    # 4. 執行相似度檢索
    print("步驟 4: 執行相似度檢索...")
    query = "這份文件的主要內容是什麼？"
    results = vector_store.similarity_search(query, k=3)
    
    print(f"查詢: {query}")
    print(f"找到 {len(results)} 個相關段落：")
    for i, doc in enumerate(results, 1):
        print(f"\n段落 {i}:")
        print(f"內容: {doc.page_content[:150]}...")
        print(f"來源: {doc.metadata.get('source', '未知')}, 頁碼: {doc.metadata.get('page', '未知')}")
    print()
    
    # 5. 使用 RAG 系統問答
    print("步驟 5: 使用 RAG 系統進行問答...")
    rag = RAGSystem(vector_store=vector_store, retrieval_k=4)
    
    question = "請總結這份文件的核心要點"
    print(f"問題: {question}")
    
    answer = rag.chat(question)
    print(f"回答: {answer}\n")
    
    print("=== 示例完成 ===")


def example_advanced_usage():
    """
    進階使用示例：多文件載入、檢索評分、詳細上下文
    """
    print("=== PDF 智能搜尋引擎 - 進階使用示例 ===\n")
    
    load_dotenv()
    
    # 載入多個 PDF 文件
    print("載入多個 PDF 文件...")
    loader = PDFDocumentLoader(chunk_size=800, chunk_overlap=150)
    
    pdf_files = [
        "data/doc1.pdf",
        "data/doc2.pdf"
    ]
    
    # 過濾存在的文件
    existing_files = [f for f in pdf_files if os.path.exists(f)]
    
    if not existing_files:
        print("警告: 沒有找到示例 PDF 文件")
        return
    
    documents = loader.load_multiple_pdfs(existing_files)
    print(f"✓ 載入 {len(existing_files)} 個文件，共 {len(documents)} 個段落\n")
    
    # 使用 FAISS 向量存儲
    print("使用 FAISS 向量存儲...")
    embedding = TextEmbedding()
    vector_store = VectorStore(
        embedding=embedding,
        store_type="faiss",
        persist_directory="faiss_index"
    )
    vector_store.create_vector_store(documents)
    print("✓ FAISS 索引創建完成\n")
    
    # 檢索並顯示分數
    print("執行相似度檢索（含分數）...")
    query = "重要的技術細節"
    results_with_scores = vector_store.similarity_search_with_score(query, k=3)
    
    print(f"查詢: {query}")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\n結果 {i} (相似度分數: {score:.4f}):")
        print(f"內容: {doc.page_content[:100]}...")
    print()
    
    # 使用 RAG 獲取詳細上下文
    print("使用 RAG 系統獲取詳細回答...")
    rag = RAGSystem(vector_store=vector_store)
    
    result = rag.query_with_context("這些文件討論了哪些關鍵技術？")
    
    print(f"問題: 這些文件討論了哪些關鍵技術？")
    print(f"回答: {result['answer']}")
    print(f"\n參考來源 ({result['num_sources']} 個):")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['source']} (頁 {source['page']})")
    
    print("\n=== 進階示例完成 ===")


def example_simple_rag():
    """
    簡化 RAG 使用示例
    """
    print("=== 簡化 RAG 系統示例 ===\n")
    
    load_dotenv()
    
    # 假設向量存儲已經創建
    print("載入現有向量存儲...")
    embedding = TextEmbedding()
    vector_store = VectorStore(
        embedding=embedding,
        store_type="chroma",
        persist_directory="chroma_db"
    )
    
    try:
        vector_store.load_vector_store()
        print("✓ 向量存儲載入成功\n")
    except FileNotFoundError:
        print("錯誤: 向量存儲不存在，請先運行 example_basic_usage()")
        return
    
    # 使用簡化 RAG
    print("使用簡化 RAG 系統...")
    simple_rag = SimpleRAG(vector_store=vector_store)
    
    questions = [
        "這份文件的主題是什麼？",
        "有哪些重要的結論？",
        "文件中提到的主要挑戰是什麼？"
    ]
    
    for question in questions:
        print(f"\n問: {question}")
        answer = simple_rag.answer(question, k=3)
        print(f"答: {answer}")
    
    print("\n=== 簡化示例完成 ===")


if __name__ == "__main__":
    """
    運行示例
    
    使用前準備：
    1. 安裝依賴: pip install -r requirements.txt
    2. 配置 API Key: 複製 .env.example 為 .env 並填入 OPENAI_API_KEY
    3. 準備 PDF 文件: 將 PDF 文件放在 data/ 目錄下
    """
    
    print("PDF 智能搜尋引擎示例程序\n")
    print("可用的示例：")
    print("1. 基本使用 (example_basic_usage)")
    print("2. 進階使用 (example_advanced_usage)")
    print("3. 簡化 RAG (example_simple_rag)")
    print()
    
    # 運行基本示例
    # 取消註釋以下行來運行不同的示例
    
    # example_basic_usage()
    # example_advanced_usage()
    # example_simple_rag()
    
    print("\n提示: 取消註釋 main 函數中的示例調用來運行")
    print("確保已設定 OPENAI_API_KEY 並準備好 PDF 文件")
