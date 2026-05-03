import os
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "Sched_LLM Dynamic Scheduling System")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    # 将 LLM 的默认超时时间从 120 秒延长到 300 秒（5分钟）
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "300"))


settings = Settings()