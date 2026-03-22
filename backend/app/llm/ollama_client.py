import requests
from app.config import settings


class OllamaClient:
    def __init__(self, model: str = None, base_url: str = None, timeout: int = None):
        self.model = model or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout or settings.llm_timeout

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")