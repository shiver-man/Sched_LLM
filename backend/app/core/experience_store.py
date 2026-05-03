import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from app.models.schema import SchedulingExperience, ScheduleMetrics

class ExperienceStore:
    def __init__(self):
        self.data_dir = Path(__file__).resolve().parents[2] / "data"
        self.experience_file = self.data_dir / "scheduling_experiences.json"
        self._ensure_data_dir()
        self.experiences: List[SchedulingExperience] = self._load_experiences()

    def _ensure_data_dir(self):
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
        if not self.experience_file.exists():
            with open(self.experience_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load_experiences(self) -> List[SchedulingExperience]:
        try:
            with open(self.experience_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [SchedulingExperience(**item) for item in data]
        except Exception:
            return []

    def save_experience(self, exp: SchedulingExperience):
        self.experiences.append(exp)
        with open(self.experience_file, "w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in self.experiences], f, indent=2, ensure_ascii=False)

    def search_similar(self, current_summary: Dict[str, Any], limit: int = 3) -> List[SchedulingExperience]:
        """
        基于简单的特征距离（工件数、机器数等）寻找相似经验。
        未来可以升级为向量搜索。
        """
        def calculate_distance(exp_summary: Dict[str, Any]):
            dist = 0
            for key in ["jobs_count", "machines_count", "vehicles_count"]:
                dist += abs(current_summary.get(key, 0) - exp_summary.get(key, 0))
            return dist

        sorted_exps = sorted(self.experiences, key=lambda x: calculate_distance(x.case_summary))
        return sorted_exps[:limit]

experience_store = ExperienceStore()
