from typing import List, Dict, Any, Callable, Optional
from app.core.dispatcher import Dispatcher
import copy

class Simulator:
    """
    仿真引擎：负责运行完整的调度轨迹。
    """

    @staticmethod
    def run_simulation(
        initial_state: Dict[str, Any], 
        policy_func: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
        max_steps: int = 1000
    ) -> Dict[str, Any]:
        """
        从初始状态开始运行仿真，直到所有工件完成。
        
        policy_func: 一个函数，接收 state 并返回一个 decision (Dict)。
        """
        state = copy.deepcopy(initial_state)
        step = 0
        
        while not all(j["finished"] for j in state["jobs"]) and step < max_steps:
            # 1. 获取决策
            decision = policy_func(state)
            
            if decision:
                # 2. 应用决策
                state = Dispatcher.apply_decision(state, decision)
                # 决策后，时间应该推进
                # 在简单的事件驱动仿真中，我们通常将时间推进到下一个“可用时间”
                # 这里我们简单地将时间推进到所有忙碌资源中最早完成的那个时间
                busy_times = [m["available_time"] for m in state["machines"] if m["status"] == "busy"]
                busy_times += [v["available_time"] for v in state["vehicles"] if v["status"] == "busy"]
                
                if busy_times:
                    next_time = min(busy_times)
                    if next_time > state["time"]:
                        state["time"] = next_time
                        # 释放已经完成任务的资源
                        for m in state["machines"]:
                            if m["status"] == "busy" and m["available_time"] <= state["time"]:
                                m["status"] = "idle"
                                m["current_job"] = None
                        for v in state["vehicles"]:
                            if v["status"] == "busy" and v["available_time"] <= state["time"]:
                                v["status"] = "idle"
                                v["current_task"] = None
            else:
                # 如果没有可执行的动作，但还有工件没完成，说明需要等待资源释放或工件到达
                # 寻找下一个时间点
                possible_times = [j["release_time"] for j in state["jobs"] if not j["finished"] and j["release_time"] > state["time"]]
                possible_times += [m["available_time"] for m in state["machines"] if m["status"] == "busy"]
                possible_times += [v["available_time"] for v in state["vehicles"] if v["status"] == "busy"]
                
                if possible_times:
                    state["time"] = min(possible_times)
                    # 释放资源
                    for m in state["machines"]:
                        if m["status"] == "busy" and m["available_time"] <= state["time"]:
                            m["status"] = "idle"
                            m["current_job"] = None
                    for v in state["vehicles"]:
                        if v["status"] == "busy" and v["available_time"] <= state["time"]:
                            v["status"] = "idle"
                            v["current_task"] = None
                else:
                    # 无法继续推进，退出
                    break
            
            step += 1
            
        return state
