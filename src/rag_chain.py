import bs4
from langchain_community.document_loaders import WebBaseLoader

# 網頁網址清單
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/"
]

#only keep posts, title, headers, content from full HTML
bs4_strainer = bs4.SoupStrainer(
    class_=("post-title", "post-header", "post-content")
)

# Initialize the web loader
loader = WebBaseLoader(web_path=urls, bs_kwargs={"parse_only": bs4_strainer})
docs = loader.load()
assert len(docs) == 1
print(type(docs[0]))