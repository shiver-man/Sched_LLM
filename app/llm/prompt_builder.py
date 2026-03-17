from typing import Dict, Any, List
from app.models.state import get_dispatchable_jobs

def _summarize_jobs(state: Dict[str, Any]) -> str:
    lines = []
    for job in state["jobs"]:
        if job["finished"]:
            lines.append(f"工件 {job['job_id']} (已完成)")
            continue
        op = job["operations"][job["current_op_index"]]
        lines.append(
            f"工件 {job['job_id']} (释放:{job['release_time']}, 准备就绪:{job.get('ready_time', 0.0)}, 交期:{job['due_time']}, 当前位置:{job['current_location']}) -> 工序 {op['op_id']} (候选机器:{[cm['machine_id'] for cm in op['candidate_machines']]})"
        )
    return "\n".join(lines)

def _summarize_machines(state: Dict[str, Any]) -> str:
    lines = []
    for m in state["machines"]:
        lines.append(f"机器 {m['machine_id']} (位置:{m['location']}, 状态:{m['status']}, 可用时间:{m['available_time']})")
    return "\n".join(lines)

def _summarize_vehicles(state: Dict[str, Any]) -> str:
    lines = []
    for v in state["vehicles"]:
        lines.append(f"车辆 {v['vehicle_id']} (位置:{v['current_location']}, 状态:{v['status']}, 可用时间:{v['available_time']})")
    return "\n".join(lines)

def build_dispatch_prompt(state: Dict[str, Any], strategic_experience: str = "") -> str:
    dispatchable = [j["job_id"] for j in get_dispatchable_jobs(state)]
    strategic_experience = strategic_experience or state.get("strategic_experience", "")

    return f"""
你是柔性作业车间生产-运输一体化调度专家。

你的任务：基于用户动态输入的车间状态，决定下一步应该派工哪一个工件、在哪台机器加工、由哪辆车运输。

重要规则：
1. 只能从当前未完成且已释放、已准备好的工件中选择。
2. 必须且只能从以下列表中选择一个 job_id: {dispatchable}
3. machine_id 必须来自所选工件当前工序的 candidate_machines 列表。
4. 如果工件当前位置和目标机器位置不同，必须分配一个空闲车辆；如果没有空闲车辆，也可以返回 vehicle_id 为 null，但之后该任务会被延迟。
5. 你的输出必须是 JSON，格式如下：
{{
  "job_id": "J1",
  "op_id": "O11",
  "machine_id": "M1",
  "vehicle_id": "V1",
  "reason": "简要说明为什么这样调度"
}}
6. 不要输出 markdown，不要输出代码块，只输出 JSON。

当前时刻：{state['time']}

全部工件详情：
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


def build_llm_plan_payload(plan_result: Dict[str, Any]) -> Dict[str, Any]:
    all_rule_results = plan_result.get("all_rule_results", [])
    comparison = [
        {
            "rule": item["rule"],
            "metrics": item["metrics"],
            "num_steps": len(item.get("plan", [])),
        }
        for item in all_rule_results
    ]

    return {
        "task_type": "fjsp_production_transport_scheduling",
        "objective": plan_result.get("objective", "makespan"),
        "best_rule": plan_result.get("best_rule"),
        "best_metrics": plan_result.get("best_metrics", {}),
        "best_schedule_plan": plan_result.get("best_schedule_plan", []),
        "rule_comparison": comparison,
    }


def build_llm_plan_brief(llm_payload: Dict[str, Any]) -> str:
    best_metrics = llm_payload.get("best_metrics", {})
    best_steps = llm_payload.get("best_schedule_plan", [])
    rule_comparison = llm_payload.get("rule_comparison", [])

    lines = [
        "调度问题类型: 柔性作业车间生产-运输一体化调度",
        f"优化目标: {llm_payload.get('objective', 'makespan')}",
        f"推荐规则: {llm_payload.get('best_rule')}",
        (
            "最优指标: "
            f"makespan={best_metrics.get('makespan')}, "
            f"utilization={best_metrics.get('utilization')}, "
            f"total_tardiness={best_metrics.get('total_tardiness')}, "
            f"total_transport_time={best_metrics.get('total_transport_time')}, "
            f"num_events={best_metrics.get('num_events')}"
        ),
        "最优调度步骤:"
    ]

    for step in best_steps:
        lines.append(
            f"Step {step['step']}: {step['job_id']}-{step['op_id']} -> {step['machine_id']}, "
            f"vehicle={step.get('vehicle_id')}, "
            f"start={step['start_time']}, finish={step['finish_time']}, "
            f"transport_time={step.get('transport_time', 0)}"
        )

    lines.append("规则对比:")
    for item in rule_comparison:
        metrics = item.get("metrics", {})
        lines.append(
            f"{item['rule']}: makespan={metrics.get('makespan')}, "
            f"utilization={metrics.get('utilization')}, "
            f"total_tardiness={metrics.get('total_tardiness')}, "
            f"total_transport_time={metrics.get('total_transport_time')}, "
            f"steps={item.get('num_steps')}"
        )

    return "\n".join(lines)
