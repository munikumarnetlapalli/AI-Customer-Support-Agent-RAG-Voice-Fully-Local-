# core/llm_loader.py
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class GeminiLLM:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing! Please set GEMINI_API_KEY in your environment or in a .env file."
            )
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise RuntimeError("google-genai package is not installed. Install it with: pip install google-genai")
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text.strip()


class LocalLlamaLLM:
    def __init__(self, model_path: str = "data/models/mistral-7b-instruct.gguf",
                 n_ctx: int = 4096, n_threads: int = 4, temp: float = 0.2):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError("llama-cpp-python is not installed.")
        self.model = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
        self.temp = temp

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        resp = self.model.create(prompt=prompt, max_tokens=max_tokens, temperature=self.temp)
        return resp['choices'][0]['text'].strip()


def load_llm(model_path: Optional[str] = None):
    """
    Loads Gemini API LLM by default if GEMINI_API_KEY is set,
    otherwise attempts to fallback to local Llama GGUF model.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return GeminiLLM(api_key=api_key)
    
    # Check if local model file exists
    model_path = model_path or "data/models/mistral-7b-instruct.gguf"
    if os.path.exists(model_path):
        try:
            return LocalLlamaLLM(model_path=model_path)
        except Exception:
            pass
            
    # Default to GeminiLLM (will ask for GEMINI_API_KEY if missing)
    return GeminiLLM()