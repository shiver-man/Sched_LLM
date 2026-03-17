from typing import Dict, Any, List
from app.models.state import get_dispatchable_jobs

def _summarize_jobs(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for job in state["jobs"]:
        if job["finished"]:
            rows.append(
                {
                    "job_id": job["job_id"],
                    "status": "finished",
                }
            )
            continue
        op = job["operations"][job["current_op_index"]]
        rows.append(
            {
                "job_id": job["job_id"],
                "current_op": op["op_id"],
                "current_location": job["current_location"],
                "release_time": job["release_time"],
                "ready_time": job.get("ready_time", job["release_time"]),
                "due_time": job["due_time"],
                "candidate_machines": op["candidate_machines"],
            }
        )
    return rows

def _summarize_machines(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "machine_id": m["machine_id"],
            "location": m["location"],
            "status": m["status"],
            "available_time": m["available_time"],
            "current_job": m["current_job"],
        }
        for m in state["machines"]
    ]

def _summarize_vehicles(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "vehicle_id": v["vehicle_id"],
            "current_location": v["current_location"],
            "status": v["status"],
            "available_time": v["available_time"],
            "speed": v["speed"],
        }
        for v in state["vehicles"]
    ]

def build_dispatch_prompt(state: Dict[str, Any], strategic_experience: str = "") -> str:
    dispatchable = get_dispatchable_jobs(state)
    strategic_experience = strategic_experience or state.get("strategic_experience", "")

    return f"""
你是柔性作业车间生产-运输一体化调度专家。

你的任务：基于用户动态输入的车间状态，决定下一步应该派工哪一个工件、在哪台机器加工、由哪辆车运输。

重要规则：
1. 只能从当前未完成且已释放、已准备好的工件中选择。
2. machine_id 必须来自该工序的 candidate_machines。
3. 如果工件当前位置和目标机器位置不同，优先分配空闲车辆；如果没有车辆，也可以返回 vehicle_id 为 null。
4. 你的输出必须是 JSON，格式如下：
{{
  "job_id": "J1",
  "op_id": "O11",
  "machine_id": "M1",
  "vehicle_id": "V1",
  "reason": "简要说明为什么这样调度"
}}
5. 不要输出 markdown，不要输出代码块，只输出 JSON。

当前时刻：{state['time']}

可调度工件：
{dispatchable}

全部工件摘要：
{_summarize_jobs(state)}

机器状态：
{_summarize_machines(state)}

车辆状态：
{_summarize_vehicles(state)}

历史经验：
{strategic_experience if strategic_experience else '无'}
""".strip()


def build_reflection_prompt(trajectories):
    return f"""
你是调度优化反思专家。
请根据以下不同启发式规则得到的调度结果，归纳哪种规则在哪些状态下表现更好，并给出可迁移的高层策略建议。

输入轨迹摘要：
{trajectories}

请输出中文分析。
""".strip()