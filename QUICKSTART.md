# 快速開始指南

這是一個完整的 PDF 智能搜尋引擎，使用 RAG（檢索增強生成）技術。

## 🚀 快速設置（5 分鐘）

### 步驟 1: 安裝依賴

```bash
pip install -r requirements.txt
```

### 步驟 2: 配置 API Key

1. 複製環境變量模板：
```bash
cp .env.example .env
```

2. 編輯 `.env` 文件，添加您的 OpenAI API Key：
```
OPENAI_API_KEY=sk-your-api-key-here
```

獲取 API Key：https://platform.openai.com/api-keys

### 步驟 3: 準備 PDF 文件

將您的 PDF 文件放入 `data/` 目錄：

```bash
# 創建目錄（如果不存在）
mkdir -p data

# 複製您的 PDF 文件
cp your-document.pdf data/
```

### 步驟 4: 索引 PDF 文件

```bash
python main.py index data/your-document.pdf
```

這將：
- 解析 PDF 文件
- 切分為段落（預設 1000 字符/段）
- 生成向量嵌入
- 創建向量索引（儲存在 `chroma_db/`）

### 步驟 5: 開始提問！

#### 方法 A: 命令行問答

```bash
python main.py ask "這份文件的主要內容是什麼？"
```

#### 方法 B: 互動模式

```bash
python main.py interactive
```

然後輸入您的問題，輸入 `exit` 退出。

#### 方法 C: 向量檢索（不使用 LLM）

```bash
python main.py search "關鍵技術" -k 5
```

## 📝 使用示例

### 索引多個 PDF 文件

```bash
python main.py index data/*.pdf
```

### 使用 FAISS 作為向量存儲

```bash
python main.py index data/doc.pdf --store-type faiss --persist-dir faiss_index
```

### 使用 GPT-4 模型

```bash
python main.py ask "總結這份文件" --model gpt-4
```

### 調整檢索數量

```bash
python main.py ask "主要結論是什麼？" --retrieval-k 6
```

## 💻 程式化使用

創建一個 Python 腳本：

```python
from dotenv import load_dotenv
from document_loader import PDFDocumentLoader
from embedding import TextEmbedding
from vector_store import VectorStore
from rag_system import RAGSystem

# 載入環境變量
load_dotenv()

# 1. 載入 PDF
loader = PDFDocumentLoader()
documents = loader.load_pdf("data/sample.pdf")
print(f"載入了 {len(documents)} 個段落")

# 2. 創建向量存儲
embedding = TextEmbedding()
vector_store = VectorStore(embedding=embedding, store_type="chroma")
vector_store.create_vector_store(documents)

# 3. 使用 RAG 問答
rag = RAGSystem(vector_store=vector_store)
answer = rag.chat("這份文件討論了什麼？")
print(f"答案: {answer}")
```

## 🎯 應用場景

### 1. 企業知識庫
- 技術文檔檢索
- 內部規範查詢
- 員工培訓材料

### 2. 學術研究
- 論文文獻檢索
- 研究資料整理
- 知識提取

### 3. 法律合規
- 法規條文查詢
- 合約分析
- 政策解讀

### 4. 客戶支援
- 產品手冊查詢
- 故障排除
- FAQ 自動化

## ⚙️ 進階配置

### 調整文本切分參數

```bash
python main.py index data/doc.pdf --chunk-size 800 --chunk-overlap 150
```

### 使用不同的嵌入模型

編輯 Python 代碼：

```python
embedding = TextEmbedding(model_name="text-embedding-3-large")  # 更高精度
```

### 自定義 RAG 提示詞

編輯 `rag_system.py` 中的 `template` 變量來自定義提示詞。

## 🔍 常見問題

**Q: 需要多少費用？**  
A: 主要成本來自 OpenAI API：
- Embedding: $0.00002 / 1K tokens (text-embedding-3-small)
- LLM: $0.0005 / 1K tokens (GPT-3.5-turbo)
- 索引 100 頁文檔約 $0.01，每次問答約 $0.001

**Q: 支援哪些 PDF 格式？**  
A: 支援所有包含可提取文本的 PDF。不支援純掃描圖片的 PDF（需要 OCR）。

**Q: 可以索引多少文檔？**  
A: 沒有硬性限制。Chroma 適合中小規模（<10GB），FAISS 適合大規模（>10GB）。

**Q: 如何提高檢索準確度？**  
A: 
1. 調整 `chunk_size` 和 `chunk_overlap`
2. 增加 `retrieval_k` 值
3. 使用 `text-embedding-3-large` 模型
4. 優化提示詞模板

**Q: 支援其他語言嗎？**  
A: 是的！OpenAI 模型支援多種語言。只需用對應語言提問即可。

## 📚 更多資源

- 完整文檔：查看 `README.md`
- 使用示例：查看 `examples.py`
- 測試代碼：查看 `test_rag_system.py`

## 🐛 問題回報

如遇到問題，請檢查：
1. OPENAI_API_KEY 是否正確設置
2. PDF 文件是否包含可提取的文本
3. 依賴是否全部安裝

---

**開始探索智能文檔檢索的強大功能吧！** 🚀
