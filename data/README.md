# 資料目錄

此目錄用於存放 PDF 文件，供 RAG 系統索引和檢索使用。

## 使用方式

1. 將您的 PDF 文件放入此目錄
2. 執行索引命令：
   ```bash
   python main.py index data/your_file.pdf
   ```

## 建議的文件組織

```
data/
├── technical/      # 技術文件
├── legal/          # 法律文件
├── research/       # 研究論文
└── general/        # 一般文件
```

## 注意事項

- 確保 PDF 文件包含可提取的文本（非純掃描件）
- 大型 PDF 文件可能需要較長的處理時間
- 建議文件大小不超過 100MB
