from typing import Dict, Any, Callable, Optional
import copy
from app.core.dispatcher import Dispatcher

class Simulator:
    @staticmethod
    def _release_resources(state: Dict[str, Any]) -> None:
        for m in state["machines"]:
            if m["status"] == "busy" and m["available_time"] <= state["time"]:
                m["status"] = "idle"
                m["current_job"] = None

        for v in state["vehicles"]:
            if v["status"] == "busy" and v["available_time"] <= state["time"]:
                v["status"] = "idle"
                v["current_task"] = None

    @staticmethod
    def _advance_time(state: Dict[str, Any]) -> bool:
        future_times = []
        future_times.extend(
            j["release_time"]
            for j in state["jobs"]
            if not j["finished"] and j["release_time"] > state["time"]
        )
        future_times.extend(
            j.get("ready_time", j["release_time"])
            for j in state["jobs"]
            if not j["finished"] and j.get("ready_time", j["release_time"]) > state["time"]
        )
        future_times.extend(
            m["available_time"] for m in state["machines"] if m["available_time"] > state["time"]
        )
        future_times.extend(
            v["available_time"] for v in state["vehicles"] if v["available_time"] > state["time"]
        )

        if not future_times:
            return False

        state["time"] = min(future_times)
        Simulator._release_resources(state)
        return True

    @staticmethod
    def run_simulation(
            initial_state: Dict[str, Any],
            policy_func: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
            max_steps: int = 1000,
    ) -> Dict[str, Any]:
        state = copy.deepcopy(initial_state)
        step = 0

        while not all(j.get("finished") for j in state["jobs"]) and step < max_steps:
            Simulator._release_resources(state)
            decision = policy_func(state)

            if decision:
                try:
                    state = Dispatcher.apply_decision(state, decision)
                except Exception as e:
                    # 决策应用失败，尝试推进时间以解决资源冲突
                    if not Simulator._advance_time(state):
                        break
            else:
                moved = Simulator._advance_time(state)
                if not moved:
                    # 尝试强行推进到下一个可能的时间点（如 release_time）
                    # 防止因为 time=0 且 release_time=0 导致的 stuck
                    break

            step += 1

        final_finish = 0.0
        if state["history"]:
            final_finish = max(h["finish_time"] for h in state["history"])
        state["time"] = max(state.get("time", 0.0), final_finish)
        Simulator._release_resources(state)
        return state