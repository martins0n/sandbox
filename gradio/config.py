import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SYSTEM_PROMPT = "Hello, I'm a chatbot. Ask me anything!"
