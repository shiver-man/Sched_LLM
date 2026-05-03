from typing import Dict, Any, List
from app.models.state import get_dispatchable_jobs
from app.core.experience_store import experience_store

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


def build_reflection_prompt(trajectories: List[Dict[str, Any]]):
    # 仅提取核心指标进行反思，避免原始数据干扰
    summaries = []
    for t in trajectories:
        m = t.get("metrics", {})
        summaries.append(f"规则: {t['rule']} | Makespan: {m.get('makespan')} | 利用率: {m.get('utilization')} | 运输占比: {round(m.get('total_transport_time', 0)/max(1, m.get('makespan', 1))*100, 1)}%")

    return f"""
你是一位深耕离散制造与协同物流的【调度经验蒸馏专家】。
你的目标是从多组调度轨迹对比中，提炼出具有普适性、可读性、可迁移的“黄金准则”。

输入轨迹概要（已精简）：
{summaries}

请通过对上述数据的“深度蒸馏”，输出一份【可读性极强的经验简报】：
1. 【核心发现】：哪种规则在当前这种规模和约束下表现出了压倒性优势？其背后的逻辑是什么？
2. 【生产-物流协同规律】：运输占比与完工时间之间呈现出怎样的博弈关系？
3. 【可迁移经验】：如果以后遇到类似的（工件/机器规模）场景，我们应该优先采用什么策略？请给出 1-2 条明确的黄金法则。

要求：
- 严禁堆砌数据！必须转化为人类可理解的战略语言。
- 语气专业且具洞察力。
""".strip()


def build_llm_plan_payload(plan_result: Dict[str, Any]) -> Dict[str, Any]:
    all_rule_results = plan_result.get("all_rule_results", [])
    comparison = []
    for item in all_rule_results:
        # 兼容两种格式：原始 plan 列表 或 精简后的 plan_summary 描述
        num_steps = 0
        if "plan" in item:
            num_steps = len(item["plan"])
        elif "plan_summary" in item:
            import re
            # 尝试从 "共 123 步" 这种字符串中提取数字
            match = re.search(r"(\d+)", str(item["plan_summary"]))
            num_steps = int(match.group(1)) if match else 0
            
        comparison.append({
            "rule": item.get("rule"),
            "metrics": item.get("metrics"),
            "num_steps": num_steps,
        })

    return {
        "task_type": "fjsp_production_transport_scheduling",
        "objective": plan_result.get("objective", "makespan"),
        "best_rule": plan_result.get("best_rule"),
        "best_metrics": plan_result.get("best_metrics", {}),
        "best_schedule_plan": plan_result.get("best_schedule_plan", []),
        "jobs": plan_result.get("jobs", []),
        "machines": plan_result.get("machines", []),
        "vehicles": plan_result.get("vehicles", []),
        "rule_comparison": comparison,
    }


def build_llm_plan_brief(llm_payload: Dict[str, Any]) -> str:
    """
    构建具有可解释性 (XAI) 的调度简报（Python 备选逻辑）。
    """
    best_metrics = llm_payload.get("best_metrics", {})
    best_steps = llm_payload.get("best_schedule_plan", [])
    rule_comparison = llm_payload.get("rule_comparison", [])

    lines = [
        "### 📊 调度方案执行简报",
        f"**最优策略**: {llm_payload.get('best_rule')} | **完工时间**: {best_metrics.get('makespan')}",
        f"**资源利用率**: {best_metrics.get('utilization')} | **运输压力**: {round(best_metrics.get('total_transport_time', 0)/max(1, best_metrics.get('makespan', 1))*100, 1)}%",
        "",
        "#### 核心决策路径：",
    ]
    
    for step in best_steps[:5]:
        v_str = f" [车辆:{step.get('vehicle_id')}]" if step.get("vehicle_id") else " [无需运输]"
        lines.append(f"- T={step['start_time']}：安排工件 {step['job_id']} 至机器 {step['machine_id']}{v_str}")
    
    lines.append("\n*注：完整数据请在「详细方案」中查看。*")
    return "\n".join(lines)


def build_ollama_plan_prompt(llm_payload: Dict[str, Any]) -> str:
    objective = llm_payload.get("objective", "makespan")
    best_rule = llm_payload.get("best_rule", "")
    best_metrics = llm_payload.get("best_metrics", {})
    schedule_plan = llm_payload.get("best_schedule_plan", [])
    comparison = llm_payload.get("rule_comparison", [])
    
    # 检索相似历史经验（实现快速响应与经验复用）
    case_summary = {
        "jobs_count": len(llm_payload.get("jobs", [])),
        "machines_count": len(llm_payload.get("machines", [])),
        "vehicles_count": len(llm_payload.get("vehicles", [])),
    }
    similar_exps = experience_store.search_similar(case_summary, limit=2)
    exp_context = ""
    if similar_exps:
        exp_context = "\n### 📚 调取历史相似案例沉淀的【蒸馏经验】：\n"
        for i, exp in enumerate(similar_exps):
            exp_context += f"- 案例{i+1}：{exp.reflection[:200]}...\n"

    # 为了防止 token 过多导致 Ollama 处理过慢，这里将 steps 进一步精简，只取前5步作为示例，并简化字段
    simplified_plan = []
    for step in schedule_plan[:5]:
        simplified_plan.append({
            "j": step.get("job_id"),
            "m": step.get("machine_id"),
            "v": step.get("vehicle_id"),
            "t": f"{step.get('start_time')}-{step.get('finish_time')}"
        })

    return f"""
你是一位专业的【大模型驱动的智能排产与可解释调度专家】。
你的核心任务是展现大模型在排产系统中的“认知增强、经验蒸馏与交互驱动”三大核心价值，将底层的冷冰冰算法轨迹数据，转化为管理者能看懂的管理洞察。
{exp_context}

---
### 原始数据摘要（仅供参考）：
- 优化目标: {objective}
- 选定核心最优策略: {best_rule}
- 最优策略指标表现: {best_metrics}
- 各种策略横向对比: {comparison}
- 最优策略前置步骤示例: {simplified_plan}

---
请基于以上数据，输出一份面向工厂管理层的【大模型排产深度分析报告】，必须严格按照以下三大模块进行结构化输出：

### 1. 结果的可解释性蒸馏（解释驱动）
分析并解释为什么“{best_rule}”策略在当前工况下被选为最优方案。
结合具体数据（如完工时间、设备利用率、运输等待时间），找出方案成功的关键点。
【必须包含】：指出当前车间中的核心瓶颈是什么（是哪台机器过度繁忙，还是运输资源成为短板，并引用数据证明）。

### 2. 经验的沉淀与复用（知识驱动）
作为大模型，你需要结合上面的【历史相似案例经验】（如果有）以及本次排产的结果进行深度反思。
总结出在面对类似规模的排产任务时，未来应该遵循哪些通用的调度原则。例如：是否应该更倾向于就近加工？是否应该放宽运输优先级？

### 3. 多算法策略的动态评估（评估驱动）
分析其他被淘汰策略（如 {', '.join([x['rule'] for x in comparison if x['rule'] != best_rule])} 等）为什么表现不佳。
它们在权衡什么指标时出现了偏差（比如过度追求单机效率导致了大量的运输损耗）？给出在遇到突发插单或设备故障时，策略切换的建议。

【要求】：
- 禁止直接平铺罗列 JSON 数据或流水账式列出每一步。
- 语气需专业、自信，突出“大模型正在驱动整个决策闭环”的视角。
- 输出必须严格使用上述三个一级标题。
""".strip()
