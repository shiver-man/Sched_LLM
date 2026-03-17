def build_dispatch_prompt(state: dict) -> str:
    time_now = state["time"]

    jobs_text = []
    for job in state["jobs"]:
        jobs_text.append(
            f"工件{job['job_id']} 当前工序索引={job['current_op_index']} "
            f"释放时间={job['release_time']} 交期={job['due_time']} 完成状态={job['finished']}"
        )

    machines_text = []
    for m in state["machines"]:
        machines_text.append(
            f"机器{m['machine_id']} 位置={m['location']} 状态={m['status']} 可用时间={m['available_time']}"
        )

    vehicles_text = []
    for v in state["vehicles"]:
        vehicles_text.append(
            f"运输车{v['vehicle_id']} 位置={v['current_location']} 状态={v['status']} 可用时间={v['available_time']}"
        )

    prompt = f"""
你是柔性作业车间生产-运输一体化调度专家。

当前系统时间: {time_now}

工件状态:
{chr(10).join(jobs_text)}

机器状态:
{chr(10).join(machines_text)}

运输车状态:
{chr(10).join(vehicles_text)}

优化目标:
{state["objective"]}

请输出下一步调度建议，要求：
1. 指定优先调度哪个工件
2. 指定该工序分配到哪台机器
3. 是否需要运输，如果需要，分配哪辆车
4. 输出必须是 JSON 格式

输出格式示例：
{{
  "job_id": "J1",
  "op_id": "O11",
  "machine_id": "M2",
  "vehicle_id": "T1",
  "reason": "该工件最早可执行且运输代价较低"
}}
"""
    return prompt