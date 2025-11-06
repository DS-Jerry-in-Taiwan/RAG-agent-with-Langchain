# PDF 智能搜尋引擎 (RAG Agent with LangChain)

基於 **LangChain** 的 PDF 智能搜尋引擎，整合文檔載入、向量嵌入、向量存儲三層技術棧，實現語意檢索與 RAG（檢索增強生成）系統。

## 🎯 專案核心

建構 **PDF 智能搜尋引擎**，提供企業級知識庫問答解決方案：
- **文件向量化管線**：PDF 解析 → 文本切分 → 向量嵌入
- **向量檢索系統**：高效相似度搜索，優於傳統關鍵字比對
- **RAG 整合應用**：檢索 + LLM 生成，實現精準知識問答

## 🏗️ 技術架構

### 三層技術棧

```
┌─────────────────────────────────────────────────┐
│              RAG 系統 (rag_system.py)            │
│  ┌──────────────────────────────────────────┐  │
│  │  LLM (GPT-3.5/4) + 檢索器                 │  │
│  │  問答鏈、上下文整合、答案生成              │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                       ↑
┌─────────────────────────────────────────────────┐
│          向量存儲 (vector_store.py)              │
│  ┌──────────────────────────────────────────┐  │
│  │  Chroma / FAISS                          │  │
│  │  向量索引、相似度檢索、持久化存儲          │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                       ↑
┌─────────────────────────────────────────────────┐
│         文本嵌入 (embedding.py)                  │
│  ┌──────────────────────────────────────────┐  │
│  │  OpenAI Embeddings                       │  │
│  │  text-embedding-3-small (1536維度)       │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                       ↑
┌─────────────────────────────────────────────────┐
│       文檔載入器 (document_loader.py)            │
│  ┌──────────────────────────────────────────┐  │
│  │  PyPDFLoader + Text Splitter             │  │
│  │  PDF 解析、文本切分、結構化處理            │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 核心組件

| 組件 | 功能 | 技術 |
|------|------|------|
| **Document Loader** | PDF 解析與文本切分 | PyPDFLoader, RecursiveCharacterTextSplitter |
| **Embedding** | 文本向量化 | OpenAI text-embedding-3-small |
| **Vector Store** | 向量存儲與檢索 | Chroma, FAISS |
| **RAG System** | 檢索增強生成 | LangChain RetrievalQA, ChatOpenAI |

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

複製環境變量模板並填入 OpenAI API Key：

```bash
cp .env.example .env
# 編輯 .env 文件，填入您的 OPENAI_API_KEY
```

### 3. 準備 PDF 文件

```bash
mkdir -p data
# 將您的 PDF 文件放入 data/ 目錄
```

### 4. 索引 PDF 文件

使用命令行工具索引 PDF：

```bash
python main.py index data/your_document.pdf
```

支援批次索引多個文件：

```bash
python main.py index data/*.pdf --chunk-size 1000 --chunk-overlap 200
```

### 5. 執行檢索或問答

**向量檢索**：
```bash
python main.py search "技術文件的主要內容" -k 5
```

**智能問答**：
```bash
python main.py ask "這份文件討論了哪些關鍵技術？"
```

**互動模式**：
```bash
python main.py interactive
```

## 📖 使用示例

### 程式化使用

```python
from dotenv import load_dotenv
from document_loader import PDFDocumentLoader
from embedding import TextEmbedding
from vector_store import VectorStore
from rag_system import RAGSystem

# 載入環境變量
load_dotenv()

# 1. 載入 PDF
loader = PDFDocumentLoader(chunk_size=1000, chunk_overlap=200)
documents = loader.load_pdf("data/sample.pdf")

# 2. 初始化嵌入模型
embedding = TextEmbedding(model_name="text-embedding-3-small")

# 3. 創建向量存儲
vector_store = VectorStore(
    embedding=embedding,
    store_type="chroma",
    persist_directory="chroma_db"
)
vector_store.create_vector_store(documents)

# 4. 執行相似度檢索
results = vector_store.similarity_search("關鍵技術", k=4)
for doc in results:
    print(f"內容: {doc.page_content}")
    print(f"來源: {doc.metadata}")

# 5. 使用 RAG 問答
rag = RAGSystem(vector_store=vector_store)
answer = rag.chat("這份文件的主要結論是什麼？")
print(f"答案: {answer}")
```

### 命令行使用

```bash
# 索引文檔（使用 Chroma）
python main.py index data/tech_doc.pdf --store-type chroma

# 索引文檔（使用 FAISS）
python main.py index data/*.pdf --store-type faiss --persist-dir faiss_index

# 向量檢索
python main.py search "機器學習算法" -k 3

# 問答（使用 GPT-4）
python main.py ask "文件中提到的主要挑戰是什麼？" --model gpt-4

# 互動模式
python main.py interactive --model gpt-3.5-turbo
```

## 📁 專案結構

```
RAG-agent-with-Langchain/
├── document_loader.py    # PDF 文檔載入與切分
├── embedding.py          # 文本向量嵌入
├── vector_store.py       # 向量存儲管理
├── rag_system.py         # RAG 問答系統
├── main.py               # 命令行主程序
├── examples.py           # 使用示例
├── requirements.txt      # Python 依賴
├── .env.example          # 環境變量模板
├── .gitignore            # Git 忽略配置
└── README.md             # 專案文檔
```

## 🔧 核心功能

### 1. Document Loader (文檔載入器)

```python
from document_loader import PDFDocumentLoader

loader = PDFDocumentLoader(
    chunk_size=1000,      # 每個段落 1000 字符
    chunk_overlap=200     # 段落重疊 200 字符
)

# 載入單個文件
documents = loader.load_pdf("data/doc.pdf")

# 載入多個文件
documents = loader.load_multiple_pdfs([
    "data/doc1.pdf",
    "data/doc2.pdf"
])
```

**特性**：
- 自動 PDF 解析
- 智能文本切分（保持段落完整性）
- 保留元數據（頁碼、來源）
- 支援批次處理

### 2. Embedding (文本嵌入)

```python
from embedding import TextEmbedding

embedding = TextEmbedding(
    model_name="text-embedding-3-small"  # 或 text-embedding-3-large
)

# 單文本嵌入
vector = embedding.embed_text("這是一段測試文本")

# 批次嵌入
vectors = embedding.embed_documents([
    "文本1", "文本2", "文本3"
])

# 獲取維度
dim = embedding.get_embedding_dimension()  # 1536
```

**支援模型**：
- `text-embedding-3-small`: 1536 維度，性價比高
- `text-embedding-3-large`: 3072 維度，高精度
- `text-embedding-ada-002`: 1536 維度，傳統模型

### 3. Vector Store (向量存儲)

```python
from vector_store import VectorStore

# 使用 Chroma
vector_store = VectorStore(
    embedding=embedding,
    store_type="chroma",
    persist_directory="chroma_db"
)

# 創建索引
vector_store.create_vector_store(documents)

# 相似度檢索
results = vector_store.similarity_search("查詢文本", k=4)

# 帶分數的檢索
results_with_scores = vector_store.similarity_search_with_score("查詢", k=4)

# 載入已有索引
vector_store.load_vector_store()
```

**支援後端**：
- **Chroma**: 適合中小規模，支援持久化
- **FAISS**: 適合大規模，高性能檢索

### 4. RAG System (檢索增強生成)

```python
from rag_system import RAGSystem

rag = RAGSystem(
    vector_store=vector_store,
    model_name="gpt-3.5-turbo",
    temperature=0.0,
    retrieval_k=4
)

# 簡單問答
answer = rag.chat("這份文件的主題是什麼？")

# 詳細問答（含來源）
result = rag.query_with_context("關鍵技術有哪些？")
print(result['answer'])
print(result['sources'])
```

**特性**：
- 自動檢索相關段落
- 整合上下文生成答案
- 返回來源引用
- 支援多輪對話

## 🎯 應用場景

### 1. 企業知識庫問答
- 技術文檔智能檢索
- 內部規範快速查詢
- 員工培訓材料問答

### 2. 法規條文查詢
- 法律文件語意搜索
- 合規條款精準定位
- 政策文件智能解讀

### 3. 學術研究助手
- 論文文獻檢索
- 研究資料整理
- 知識圖譜構建

### 4. 技術支援系統
- 產品手冊智能查詢
- 故障排除知識庫
- 客戶服務自動化

## ⚙️ 進階配置

### 自定義文本切分

```python
loader = PDFDocumentLoader(
    chunk_size=800,           # 較小的段落
    chunk_overlap=150         # 較少重疊
)
```

### 選擇向量數據庫

```python
# Chroma (推薦用於開發和中小規模)
vector_store = VectorStore(
    embedding=embedding,
    store_type="chroma",
    persist_directory="chroma_db"
)

# FAISS (推薦用於生產和大規模)
vector_store = VectorStore(
    embedding=embedding,
    store_type="faiss",
    persist_directory="faiss_index"
)
```

### 自定義 RAG 提示詞

修改 `rag_system.py` 中的 prompt template：

```python
template = """使用以下上下文回答問題。
如果不知道答案，請說不知道。

上下文：
{context}

問題：{question}

回答："""
```

### 使用不同 LLM 模型

```python
rag = RAGSystem(
    vector_store=vector_store,
    model_name="gpt-4",           # 使用 GPT-4
    temperature=0.2,              # 增加創造性
    retrieval_k=6                 # 檢索更多段落
)
```

## 🔍 性能優化

### 1. 批次處理
```python
# 一次索引多個文件
documents = loader.load_multiple_pdfs(pdf_files)
vector_store.create_vector_store(documents)
```

### 2. 調整檢索參數
```python
# 平衡相關性與多樣性
results = vector_store.similarity_search(query, k=3)  # 減少 k 值
```

### 3. 選擇合適的嵌入模型
```python
# 性價比: text-embedding-3-small
# 高精度: text-embedding-3-large
embedding = TextEmbedding(model_name="text-embedding-3-small")
```

## 🐛 故障排除

### 常見問題

**1. ModuleNotFoundError: No module named 'xxx'**
```bash
pip install -r requirements.txt
```

**2. OpenAI API Key 錯誤**
```bash
# 檢查 .env 文件
cat .env
# 確保 OPENAI_API_KEY 已設定
```

**3. PDF 解析失敗**
```bash
# 檢查 PDF 文件格式
# 確保 PDF 包含可提取的文本（非掃描件）
```

**4. 向量存儲載入失敗**
```bash
# 確保先執行 index 命令
python main.py index data/your.pdf
```

## 📚 依賴項

核心依賴：
- `langchain`: LangChain 框架
- `langchain-community`: 社區集成
- `langchain-openai`: OpenAI 集成
- `pypdf`: PDF 解析
- `chromadb`: Chroma 向量數據庫
- `faiss-cpu`: FAISS 向量數據庫
- `openai`: OpenAI API
- `python-dotenv`: 環境變量管理

詳見 `requirements.txt`

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

MIT License

## 🔗 相關資源

- [LangChain 文檔](https://python.langchain.com/)
- [OpenAI API 文檔](https://platform.openai.com/docs)
- [Chroma 文檔](https://docs.trychroma.com/)
- [FAISS 文檔](https://faiss.ai/)

---

**建構者**: DS-Jerry-in-Taiwan  
**版本**: 1.0.0  
**更新日期**: 2024 
