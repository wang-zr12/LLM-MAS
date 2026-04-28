import aiohttp
import requests
import logging
import tiktoken
from abc import ABC, abstractmethod
import openai
import asyncio
from typing import Dict, List, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMInterface(ABC):
    @abstractmethod
    async def generate_response(self, messages: List[Dict], max_tokens: int = 1000) -> str:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

# DeepSeek LLM Implementation
class DeepSeekLLM(LLMInterface):
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.encoder = tiktoken.get_encoding("cl100k_base")

    async def generate_response(self, messages: List[Dict], max_tokens: int = 1024) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": max_tokens,
            "top_p": 0.8
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API error {response.status}: {error_text}")
                        raise Exception(f"API error {response.status}: {error_text}")
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

# OpenAI GPT实现
class OpenAILLM(LLMInterface):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        openai.api_key = api_key
        self.model = model
        self.encoder = tiktoken.encoding_for_model(model)

    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message["content"]
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error: {e}"

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

# 通用的HTTP LLM实现（支持Grok、Gemini等）
class GenericHTTPLLM(LLMInterface):
    def __init__(self, api_endpoint: str, api_key: str, model: str, encoder_name: str = "cl100k_base"):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model = model
        self.encoder = tiktoken.get_encoding(encoder_name)

    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens
            }
            response = await asyncio.to_thread(
                requests.post, self.api_endpoint, headers=headers, json=payload, timeout=10
            )
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("text", "No response")
        except Exception as e:
            logger.error(f"HTTP API error: {e}")
            return f"Error: {e}"

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))