from typing import List, Optional

class StrategicExperience:
    """
    存储 LLM 通过反思总结出的调度策略经验。
    """
    _instance = None
    _experiences: List[str] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StrategicExperience, cls).__new__(cls)
        return cls._instance

    def add_experience(self, text: str):
        """添加一条新经验"""
        if text not in self._experiences:
            self._experiences.append(text)

    def get_all(self) -> List[str]:
        """获取所有已存储的经验"""
        return self._experiences

    def clear(self):
        """清空经验"""
        self._experiences = []

    def get_formatted_text(self) -> str:
        """返回格式化后的经验文本供 Prompt 使用"""
        if not self._experiences:
            return "暂无存储的调度经验。"
        
        return "\n".join([f"- {exp}" for exp in self._experiences])

# 全局单例
memory = StrategicExperience()
