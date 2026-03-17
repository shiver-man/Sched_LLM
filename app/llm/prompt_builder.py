from typing import List, Dict, Any

def build_dispatch_prompt(state: dict, strategic_experience: str = "") -> str:
    """
    将当前系统状态转换为面向 LLM 的专业调度提示词。
    支持注入“反思经验”以指导决策。
    """
    time_now = state["time"]

    # ... (作业信息格式化逻辑保持不变)
    jobs_text = []
    for job in state["jobs"]:
        if job["finished"]:
            continue
        
        # 获取当前待加工工序
        current_op = job["operations"][job["current_op_index"]]
        candidate_machines = [
            f"{cm['machine_id']}(耗时:{cm['process_time']})"
            for cm in current_op["candidate_machines"]
        ]
        
        jobs_text.append(
            f"- 工件 {job['job_id']}: 释放时间={job['release_time']}, 交期={job['due_time']}, "
            f"当前工序={current_op['op_id']}, 可选机器=[{', '.join(candidate_machines)}]"
        )

    # 格式化机器信息
    machines_text = []
    for m in state["machines"]:
        machines_text.append(
            f"- 机器 {m['machine_id']}: 位置={m['location']}, 状态={m['status']}, "
            f"可用时间={m['available_time']}, 当前工件={m['current_job'] or '无'}"
        )

    # 格式化运输车信息
    vehicles_text = []
    for v in state["vehicles"]:
        vehicles_text.append(
            f"- 运输车 {v['vehicle_id']}: 位置={v['current_location']}, 状态={v['status']}, "
            f"速度={v['speed']}, 可用时间={v['available_time']}"
        )

    # 格式化优化目标
    obj = state["objective"]
    objective_text = f"类型={obj.get('type', '综合')}, 权重={obj.get('weight', 1.0)}"

    # 经验总结部分
    experience_section = ""
    if strategic_experience:
        experience_section = f"\n### 之前的调度反思（专家经验指导）:\n{strategic_experience}\n"

    prompt = f"""
你是一位资深的柔性作业车间调度（FJSP）专家。你的任务是根据当前系统状态，给出一个最优的调度动作建议。
{experience_section}
### 当前系统时间: {time_now}

### 待处理工件（当前工序）:
{chr(10).join(jobs_text) if jobs_text else "所有工件已完成"}

### 机器状态:
{chr(10).join(machines_text)}

### 运输车状态:
{chr(10).join(vehicles_text)}

### 调度目标:
{objective_text}

### 任务要求:
1. 分析当前哪些工件已经到达（当前时间 >= 释放时间）。
2. 分析哪些机器是空闲的（status=idle）。
3. 选择一个最合适的“工件-工序-机器”组合。
4. 如果该工件当前所在位置与目标机器位置不同，请分配一辆空闲运输车进行搬运。
5. 你的决策必须逻辑严密，旨在最小化交期延误或总完工时间。

### 输出格式 (JSON):
请严格按照以下 JSON 格式输出，不要包含任何多余文字：
{{
  "job_id": "工件ID",
  "op_id": "工序ID",
  "machine_id": "目标机器ID",
  "vehicle_id": "运输车ID (若不需要运输则填 null)",
  "reason": "简短的决策理由"
}}
"""
    return prompt


def build_reflection_prompt(trajectories: List[Dict[str, Any]]) -> str:
    """
    ReflecSched 核心：分层反思 (Hierarchical Reflection)。
    通过对比多个调度轨迹（如不同 PDR 规则的结果），让 LLM 总结高层调度经验。
    """
    comparison_text = []
    for i, traj in enumerate(trajectories):
        metrics = traj["metrics"]
        rule_name = traj["rule"]
        comparison_text.append(
            f"轨迹 {i+1} (规则: {rule_name}):\n"
            f"- Makespan: {metrics['makespan']}\n"
            f"- 机器利用率: {metrics['utilization']}\n"
            f"- 总交期延误: {metrics['total_tardiness']}\n"
            f"- 调度历史摘要: {traj['history_summary']}\n"
        )

    prompt = f"""
你是一位工业工程与智能制造领域的专家。你需要通过分析下面几个不同的调度轨迹结果，总结出针对当前车间环境的最优调度策略（Strategic Experience）。

### 调度轨迹对比:
{chr(10).join(comparison_text)}

### 任务要求:
1. **分析差异**: 为什么某些规则（如 SPT）在某些指标上表现更好？
2. **提炼经验**: 总结出 3 条核心的调度原则（例如：优先处理快到期的长工序，或者优先保证瓶颈机器 M1 的利用率）。
3. **输出格式**: 请以简洁的自然语言输出你的“调度经验总结”，这些经验将被用于指导后续的实时决策。

### 输出示例:
- 经验1: 当车间负荷较高时，应优先处理剩余工时最多的工件以降低整体完工时间。
- 经验2: 机器 M2 是当前瓶颈，应尽量减少其空闲时间，运输车应优先向其送料。
- 经验3: ...

请开始你的反思与经验提炼：
"""
    return prompt
