import bs4
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# load environment variables from .env file
load_dotenv()

# 網頁網址清單
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/"
]

# splitter tools
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True
)

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings()

#only keep posts, title, headers, content from full HTML
bs4_strainer = bs4.SoupStrainer(
    class_=("post-title", "post-header", "post-content")
)

# Initialize the web loader and load urls
documents = []
for i in range(len(urls)):
    loader = WebBaseLoader(web_path=urls[i], bs_kwargs={"parse_only": bs4_strainer})
    docs = loader.load()
    assert len(docs) == 1
    documents.extend(docs)
    print(f"The {i+1} url loaded document content:")
    print(docs[0].page_content)
    print(f"{len(docs[0].page_content)} characters loaded.")
    
# use splitter to split documents
split_docs = splitter.split_documents(documents)
print(f"分段後共 {len(split_docs)} 段")

# convert split documents to vectors
vectors = embeddings.embed_documents([doc.page_content for doc in split_docs])
print(f"共生成 {len(vectors)} 筆向量")
print(f"第一筆向量維度：{len(vectors[0])}")