"""
1. 串接第三方 API（如 OpenAI）
負責與外部 LLM 服務建立連線，取得模型資源。

2. 初始化 LLM 模型物件

根據 API Key 與參數，建立可用的語言模型物件，供後續生成任務使用。
3. 驗證模型物件是否可用

檢查 API Key 是否有效，模型是否能正常回應，確保服務穩定。
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# load environment variables from .env file
load_dotenv()

# get API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")
os.environ["OPENAI_API_KEY"] = api_key

# initialize the chat model
model = ChatOpenAI(model="gpt-4")

try:
    response = model.invoke("Hello, please introduce yourself briefly.")
    print("Model response:", response)
except Exception as e:
    print("Model validation failed, error message:", e)