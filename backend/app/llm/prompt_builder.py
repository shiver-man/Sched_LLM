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
    """
    构建具有可解释性 (XAI) 的调度简报。
    符合“可解释性”目标：通过前瞻得分和协同度分析，解释调度决策的底层逻辑。
    """
    best_metrics = llm_payload.get("best_metrics", {})
    best_steps = llm_payload.get("best_schedule_plan", [])
    rule_comparison = llm_payload.get("rule_comparison", [])

    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    valid_candidates = []
    for item in rule_comparison:
        metrics = item.get("metrics", {})
        mk = _to_float(metrics.get("makespan", 0))
        if mk > 0:
            valid_candidates.append(item)

    if _to_float(best_metrics.get("makespan", 0)) <= 0 and valid_candidates:
        best_item = min(valid_candidates, key=lambda x: _to_float(x.get("metrics", {}).get("makespan", 0)))
        best_metrics = best_item.get("metrics", {})
        llm_payload["best_rule"] = best_item.get("rule")

    lines = [
        "**调度概览与诊断**",
        f"优化目标: {llm_payload.get('objective', 'makespan')}",
        f"核心策略: {llm_payload.get('best_rule')} (含时域前瞻自适应模型)",
        (
            "综合指标: "
            f"Makespan={best_metrics.get('makespan')}, "
            f"设备利用率={best_metrics.get('utilization')}, "
            f"运输占比={round(_to_float(best_metrics.get('total_transport_time', 0)) / max(1.0, _to_float(best_metrics.get('makespan', 1))) * 100, 1)}%"
        ),
        "",
        "**策略决策分析 (XAI)**",
    ]

    # 1. 瓶颈与协同度深度分析
    t_time = _to_float(best_metrics.get("total_transport_time", 0))
    mk = _to_float(best_metrics.get("makespan", 1), 1.0)
    if t_time > mk * 0.6:
        lines.append("- ⚠️ 物流瓶颈突出：运输耗时严重拖累生产节奏，建议优化布局或增加搬运资源。")
    elif t_time < mk * 0.2:
        lines.append("- ✅ 物流响应极佳：运输损耗被成功压缩，体现了优异的生产-运输协同性。")

    m_util = _to_float(best_metrics.get("utilization", 0))
    if m_util < 0.4:
        lines.append("- ⚠️ 产能闲置：加工资源未被充分激活，前瞻推演显示这可能与工件到达不均衡有关。")

    # 2. 前瞻决策路径
    lines.append("")
    lines.append("**关键决策路径 (Look-Ahead Insight):**")
    lines.append("注：Score 代表决策的长远收益预估，分值越高表示对后续瓶颈的缓解越有效。")
    for step in best_steps[:12]:
        v_str = f" [AGV:{step.get('vehicle_id')}]" if step.get("vehicle_id") else " [无运输]"
        score_val = step.get("lookahead_score", "N/A")
        lines.append(
            f"T={step['start_time']} | {step['job_id']} → {step['machine_id']}{v_str} | Score: {score_val} | 完工={step['finish_time']}"
        )
    if len(best_steps) > 12:
        lines.append(f"... (其余 {len(best_steps)-12} 个步骤已省略)")

    # 3. 鲁棒性验证
    lines.append("")
    lines.append("**方案鲁棒性对比 (多场景推演):**")
    for item in rule_comparison:
        metrics = item.get("metrics", {})
        mk_item = _to_float(metrics.get("makespan", 0))
        if mk_item <= 0:
            lines.append(f"- {item['rule']}: 无有效计划")
        else:
            lines.append(f"- {item['rule']}: 预期 Makespan = {mk_item}")

    return "\n".join(lines)


def build_ollama_plan_prompt(llm_payload: Dict[str, Any]) -> str:
    objective = llm_payload.get("objective", "makespan")
    best_rule = llm_payload.get("best_rule", "")
    best_metrics = llm_payload.get("best_metrics", {})
    schedule_plan = llm_payload.get("best_schedule_plan", [])
    comparison = llm_payload.get("rule_comparison", [])
    return f"""
你是生产-运输协同调度分析专家。请基于以下结构化结果，输出给前端可直接展示的中文方案说明。

要求：
1) 用简洁中文输出；
2) 必须包含：核心策略、核心指标、关键调度步骤、运输瓶颈与改进建议；
3) 不要输出 JSON，不要输出代码块；
4) 如果某策略 makespan<=0，要明确指出该策略无效，不得当作最优。

优化目标: {objective}
核心策略: {best_rule}
核心指标: {best_metrics}
核心计划步骤: {schedule_plan[:20]}
策略对比: {comparison}

请输出：
- 调度结论
- 关键路径解释
- 风险与瓶颈
- 下一步优化建议
""".strip()
