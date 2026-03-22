import json
import re
from typing import Dict, Any


class ResponseParseError(ValueError):
    pass


def parse_llm_response(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ResponseParseError("LLM 返回为空")

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ResponseParseError(f"提取到 JSON 但解析失败: {exc}") from exc

    raise ResponseParseError(f"无法从 LLM 返回中解析 JSON: {text}")