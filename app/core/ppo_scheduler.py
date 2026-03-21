"""
PPO (Proximal Policy Optimization) 算法实现模块
用于训练和执行强化学习策略，处理任务调度决策
"""
from typing import Dict, Any, List, Optional
import json
import math
from pathlib import Path
import random
import uuid
import networkx as nx

# 导入项目内部模块
from app.models.state import build_initial_state, get_dispatchable_jobs
from app.core.simulator import Simulator
from app.core.dispatcher import Dispatcher
from app.core.evaluator import Evaluator




# 策略存储相关常量
POLICY_STORE: Dict[str, Dict[str, Any]] = {}  # 存储所有策略的字典
POLICY_FILE = Path(__file__).resolve().parents[2] / "data" / "ppo_policies.json"  # 策略文件路径
LAST_POLICY_FILE = Path(__file__).resolve().parents[2] / "data" / "last_policy_id.txt"  # 最后使用的策略ID文件路径


def _load_policy_store() -> None:

    """加载策略存储文件中的所有策略"""
    if POLICY_STORE:  # 如果已经加载过，直接返回
        return
    if not POLICY_FILE.exists():  # 如果策略文件不存在，返回
        return
    data = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    if isinstance(data, dict):  # 验证数据格式并加载
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(v.get("weights"), list):
                POLICY_STORE[k] = v


def _save_policy_store() -> None:

    """保存当前所有策略到文件"""
    POLICY_FILE.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    POLICY_FILE.write_text(json.dumps(POLICY_STORE, ensure_ascii=False), encoding="utf-8")


def _read_last_policy_id() -> Optional[str]:

    """读取最后使用的策略ID"""
    if not LAST_POLICY_FILE.exists():  # 如果文件不存在，返回None
        return None
    policy_id = LAST_POLICY_FILE.read_text(encoding="utf-8").strip()
    return policy_id or None  # 返回有效的policy_id


def _dot(a: List[float], b: List[float]) -> float:
    """计算两个向量的点积"""
    return sum(x * y for x, y in zip(a, b))


def _softmax(logits: List[float]) -> List[float]:

    """计算Softmax概率分布"""
    if not logits:  # 空列表处理
        return []
    max_logit = max(logits)  # 数值稳定性处理
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    if total <= 0:  # 防止除零错误
        return [1.0 / len(logits)] * len(logits)
    return [x / total for x in exps]


def _state_features(state: Dict[str, Any]) -> List[float]:
    """提取状态特征向量"""
    time_now = state["time"]
    jobs = state["jobs"]
    machines = state["machines"]
    vehicles = state["vehicles"]
    unfinished = [j for j in jobs if not j["finished"]]
    dispatchable = get_dispatchable_jobs(state)
    idle_m = [m for m in machines if m["status"] == "idle"]
    down_m = [m for m in machines if m["status"] == "down"]
    idle_v = [v for v in vehicles if v["status"] == "idle"]

    # 计算平均交期松弛时间和等待时间
    avg_due_slack = 0.0
    avg_ready_wait = 0.0
    if unfinished:
        avg_due_slack = sum(j.get("due_time", 1000.0) - time_now for j in unfinished) / len(unfinished)
        avg_ready_wait = sum(max(0.0, time_now - j.get("ready_time", j["release_time"])) for j in unfinished) / len(unfinished)

    # 运输相关特征：空闲车辆到待调度工件的平均距离
    avg_v_dist = 0.0
    if idle_v and dispatchable:
        graph = state["graph"]
        total_dist = 0.0
        for v in idle_v:
            for j in dispatchable:
                try:
                    total_dist += nx.shortest_path_length(graph, v["current_location"], j["current_location"], weight="weight")
                except:
                    total_dist += 50.0
        avg_v_dist = total_dist / (len(idle_v) * len(dispatchable))

    # 返回归一化的特征向量 (维度: 8 -> 9)
    return [
        time_now / 100.0,
        len(unfinished) / max(1, len(jobs)),
        len(dispatchable) / max(1, len(jobs)),
        len(idle_m) / max(1, len(machines)),
        len(down_m) / max(1, len(machines)),
        len(idle_v) / max(1, len(vehicles) if vehicles else 1),
        avg_due_slack / 100.0,
        avg_ready_wait / 100.0,
        avg_v_dist / 50.0,  # 新增：物流响应延迟潜力
    ]


def _transport_time(state: Dict[str, Any], from_loc: str, to_loc: str, vehicle: Optional[Dict[str, Any]]) -> float:

    """计算运输时间"""
    if from_loc == to_loc:  # 同一位置不需要运输
        return 0.0
    if vehicle is None:  # 没有车辆无法运输
        return float("inf")
    graph = state["graph"]
    # 计算携带任务的运输时间
    carrying = nx.shortest_path_length(graph, from_loc, to_loc, weight="weight") / vehicle["speed"]
    # 计算重新定位时间（如果需要）
    reposition = 0.0
    if vehicle["current_location"] != from_loc:
        reposition = (
            nx.shortest_path_length(graph, vehicle["current_location"], from_loc, weight="weight")
            / vehicle["speed"]
        )
    return reposition + carrying + vehicle.get("load_unload_time", 0.0)


def _build_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """构建所有可能的动作"""
    dispatchable_jobs = get_dispatchable_jobs(state)
    idle_machines = [m for m in state["machines"] if m["status"] == "idle"]
    idle_vehicles = [v for v in state["vehicles"] if v["status"] == "idle"]
    actions = []
    time_now = state["time"]

    # 遍历所有可调度任务
    for job in dispatchable_jobs:
        op = job["operations"][job["current_op_index"]]
        remaining_ops = len(job["operations"]) - job["current_op_index"]
        # 遍历所有空闲机器
        for machine in idle_machines:
            cm = next((x for x in op["candidate_machines"] if x["machine_id"] == machine["machine_id"]), None)
            if not cm:  # 机器不是候选机器，跳过
                continue
            
            same_loc = job["current_location"] == machine["location"]
            due_slack = job.get("due_time", 1000.0) - time_now
            
            if same_loc:  # 任务和机器在同一位置
                actions.append(
                    {
                        "decision": {
                            "job_id": job["job_id"],
                            "op_id": op["op_id"],
                            "machine_id": machine["machine_id"],
                            "vehicle_id": None,
                            "reason": "PPO policy",
                        },
                        "features": [
                            cm["process_time"] / 100.0,
                            0.0, # transport_time
                            remaining_ops / 10.0,
                            due_slack / 100.0,
                            1.0, # is_local
                            0.0, # empty_run
                        ],
                    }
                )
            else:  # 任务和机器不在同一位置，需要运输
                for vehicle in idle_vehicles:
                    graph = state["graph"]
                    # 空跑距离 (Repositioning)
                    empty_run = nx.shortest_path_length(graph, vehicle["current_location"], job["current_location"], weight="weight")
                    # 载重距离 (Carrying)
                    carrying = nx.shortest_path_length(graph, job["current_location"], machine["location"], weight="weight")
                    
                    t_time = (empty_run + carrying) / vehicle["speed"] + vehicle.get("load_unload_time", 0.0)
                    
                    actions.append(
                        {
                            "decision": {
                                "job_id": job["job_id"],
                                "op_id": op["op_id"],
                                "machine_id": machine["machine_id"],
                                "vehicle_id": vehicle["vehicle_id"],
                                "reason": "PPO policy",
                            },
                            "features": [
                                cm["process_time"] / 100.0,
                                t_time / 100.0,
                                remaining_ops / 10.0,
                                due_slack / 100.0,
                                0.0, # is_local
                                empty_run / 50.0, # empty_run penalty feature
                            ],
                        }
                    )
    return actions


def _compose_features(state_f: List[float], action_f: List[float]) -> List[float]:
    """组合状态特征和动作特征"""
    cross = [
        state_f[0] * action_f[0],  # 当前时间 × 处理时间
        state_f[1] * action_f[2],  # 未完成任务比例 × 剩余操作数
        state_f[2] * action_f[1],  # 可调度任务比例 × 运输时间
        state_f[6] * action_f[3],  # 平均交期松弛 × 交期松弛
    ]
    return state_f + action_f + cross  # 拼接所有特征


def _sample_action(probs: List[float], rng: random.Random) -> int:
    """根据概率分布采样动作"""
    r = rng.random()
    acc = 0.0
    for idx, p in enumerate(probs):
        acc += p
        if r <= acc:
            return idx
    return len(probs) - 1  # 防止数值误差导致的越界


def _sample_processing_times(state: Dict[str, Any], rng: random.Random, low: float, high: float) -> None:
    """为处理时间添加随机噪声"""
    for job in state["jobs"]:
        for op in job["operations"]:
            for cm in op["candidate_machines"]:
                if "_base_process_time" not in cm:  # 保存基准处理时间
                    cm["_base_process_time"] = cm["process_time"]
                noise = rng.uniform(low, high)  # 添加随机噪声
                cm["process_time"] = cm["_base_process_time"] * noise


def _compute_reward(state: Dict[str, Any]) -> float:
    """强化奖励函数：引导 PPO 降低运输比重、减少协同间隙并平衡物流/加工资源利用率。"""
    if not state["history"]:
        return -0.05
    
    last = state["history"][-1]
    process_duration = last["finish_time"] - last["start_time"]
    transport = last.get("transport_time", 0.0)
    time_now = state.get("time", 0.0)
    
    # 1. 基础惩罚：不仅惩罚时间，还惩罚运输相对于加工的比例
    # 生产-运输协同：运输占比越高，惩罚越重
    transport_ratio_penalty = (transport / (process_duration + 0.1)) * 2.0
    reward = -0.1 * process_duration - 2.0 * transport - transport_ratio_penalty
    
    # 2. 任务完成奖励 (分步奖励 + 终点奖励)
    job = next((j for j in state["jobs"] if j["job_id"] == last["job_id"]), None)
    if job:
        total_ops = len(job["operations"])
        progress = (job["current_op_index"]) / total_ops
        reward += 5.0 * progress # 鼓励向前推进
        if job["finished"]:
            reward += 20.0 # 完成整个任务的重奖
            
    # 3. 机器等待惩罚 (协同性：如果运输太久让机器闲置，惩罚)
    # 寻找该机器上一次完成的时间
    machine_id = last["machine_id"]
    machine_history = [h for h in state["history"][:-1] if h["machine_id"] == machine_id]
    if machine_history:
        prev_finish = machine_history[-1]["finish_time"]
        idle_gap = last["start_time"] - prev_finish
        if idle_gap > 0.1:
            reward -= 0.5 * idle_gap # 惩罚机器闲置等待工件的时间
            
    # 4. 交期压力与逾期惩罚
    if job:
        due_slack = job.get("due_time", 1000.0) - last["finish_time"]
        if due_slack < 0:
            reward -= 2.0 * abs(due_slack) # 严厉惩罚逾期
        elif due_slack < 20:
            reward += 1.0 # 奖励在交期前不久完成 (JIT 倾向)
            
    # 5. 资源负载均衡 (定期全局评估)
    if len(state["history"]) % 5 == 0:
        v_utils = []
        for v in state.get("vehicles", []):
            v_history = [h for h in state["history"] if h.get("vehicle_id") == v["vehicle_id"]]
            v_work = sum(h.get("transport_time", 0.0) for h in v_history)
            v_utils.append(v_work / max(time_now, 1.0))
        
        if v_utils:
            avg_v = sum(v_utils) / len(v_utils)
            v_imbalance = sum(abs(x - avg_v) for x in v_utils) / len(v_utils)
            reward -= 3.0 * v_imbalance # 生产-运输协同：严防物流资源分配极度不均

    return reward


def _plan_from_history(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从历史记录生成计划"""
    return [
        {
            "step": idx + 1,
            "job_id": h["job_id"],
            "op_id": h["op_id"],
            "machine_id": h["machine_id"],
            "vehicle_id": h.get("vehicle_id"),
            "transport_time": h.get("transport_time", 0),
            "start_time": h["start_time"],
            "finish_time": h["finish_time"],
        }
        for idx, h in enumerate(state["history"])
    ]


def _rollout(req, weights: List[float], noise_low: float, noise_high: float, rng: random.Random, sample: bool, max_steps: int):
    """执行一轮策略评估"""
    state = build_initial_state(req)
    transitions: List[Dict[str, Any]] = []
    steps = 0

    # 主循环：直到所有任务完成或达到最大步数
    while not all(j["finished"] for j in state["jobs"]) and steps < max_steps:
        Simulator._release_resources(state)
        _sample_processing_times(state, rng, noise_low, noise_high)
        actions = _build_actions(state)

        if actions:  # 如果有可用动作
            s_feat = _state_features(state)
            phis = [_compose_features(s_feat, a["features"]) for a in actions]
            logits = [_dot(weights, phi) for phi in phis]
            probs = _softmax(logits)
            # 根据采样策略选择动作
            selected_idx = _sample_action(probs, rng) if sample else max(range(len(probs)), key=lambda i: probs[i])

            decision = actions[selected_idx]["decision"]
            old_prob = max(probs[selected_idx], 1e-8)  # 防止概率为0
            try:
                state = Dispatcher.apply_decision(state, decision)
                reward = _compute_reward(state)
            except Exception:  # 动作执行失败
                reward = -2.0
                selected_idx = None

            if selected_idx is not None:
                transitions.append(
                    {
                        "phis": phis,
                        "selected_idx": selected_idx,
                        "old_prob": old_prob,
                        "reward": reward,
                    }
                )
        else:  # 没有可用动作，推进时间
            moved = Simulator._advance_time(state)
            if not moved:  # 时间无法推进，退出
                break
            if transitions:  # 惩励时间推进
                transitions[-1]["reward"] += -0.01

        steps += 1

    metrics = Evaluator.evaluate(state)
    return state, transitions, metrics


def _discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    """计算折扣回报"""
    out = [0.0] * len(rewards)
    running = 0.0
    # 反向计算折扣回报
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def train_ppo_policy(req) -> Dict[str, Any]:
    """训练PPO策略"""
    _load_policy_store()
    rng = random.Random(req.seed)
    dim = 9 + 6 + 4  # 特征维度：状态特征(9) + 动作特征(6) + 交叉特征(4)
    weights = [rng.uniform(-0.02, 0.02) for _ in range(dim)]  # 初始化权重
    reward_history = []

    # 训练循环
    for _ in range(req.episodes):
        _, transitions, _ = _rollout(
            req=req,
            weights=weights,
            noise_low=req.process_time_noise_low,
            noise_high=req.process_time_noise_high,
            rng=rng,
            sample=True,  # 采样探索
            max_steps=req.max_steps,
        )
        if not transitions:  # 没有有效转换，记录低奖励
            reward_history.append(-1.0)
            continue

        # 计算回报和优势
        rewards = [t["reward"] for t in transitions]
        returns = _discounted_returns(rewards, req.gamma)
        mean_return = sum(returns) / len(returns)
        advantages = [r - mean_return for r in returns]
        reward_history.append(sum(rewards))

        # 策略更新
        for _ in range(req.update_epochs):
            grad = [0.0] * dim
            for t, adv in zip(transitions, advantages):
                phis = t["phis"]
                idx = t["selected_idx"]
                logits = [_dot(weights, phi) for phi in phis]
                probs = _softmax(logits)
                new_prob = max(probs[idx], 1e-8)
                ratio = new_prob / t["old_prob"]
                # PPO裁剪
                clipped_ratio = min(max(ratio, 1.0 - req.clip_ratio), 1.0 + req.clip_ratio)

                # 根据优势值决定是否更新
                if adv >= 0 and ratio > 1.0 + req.clip_ratio:
                    continue
                if adv < 0 and ratio < 1.0 - req.clip_ratio:
                    continue

                # 计算梯度
                expected_phi = [0.0] * dim
                for p, phi in zip(probs, phis):
                    for i in range(dim):
                        expected_phi[i] += p * phi[i]
                dlogpi = [phis[idx][i] - expected_phi[i] for i in range(dim)]
                coeff = adv * clipped_ratio
                for i in range(dim):
                    grad[i] += coeff * dlogpi[i]

            # 更新权重
            scale = req.learning_rate / max(1, len(transitions))
            for i in range(dim):
                weights[i] += scale * grad[i]

    # 最终评估
    final_state, _, final_metrics = _rollout(
        req=req,
        weights=weights,
        noise_low=req.process_time_noise_low,
        noise_high=req.process_time_noise_high,
        rng=random.Random(req.seed + 777),  # 使用不同的随机种子
        sample=False,  # 不采样，选择最佳动作
        max_steps=req.max_steps,
    )

    # 保存策略
    policy_id = f"ppo_{uuid.uuid4().hex[:10]}"
    POLICY_STORE[policy_id] = {
        "weights": weights,
        "noise_low": req.process_time_noise_low,
        "noise_high": req.process_time_noise_high,
    }
    _save_policy_store()
    LAST_POLICY_FILE.write_text(policy_id, encoding="utf-8")

    return {
        "policy_id": policy_id,
        "episodes": req.episodes,
        "reward_history": reward_history,
        "final_metrics": final_metrics,
        "final_plan": _plan_from_history(final_state),
    }


def run_ppo_policy(req) -> Dict[str, Any]:
    """执行PPO策略"""
    _load_policy_store()
    policy_id = str(req.policy_id).strip().strip("\"'`")
    placeholders = {"", "ppo_xxxxxxxxxx", "string", "your_policy_id"}
    if policy_id in placeholders:  # 如果是占位符，使用最后使用的策略
        last_policy_id = _read_last_policy_id()
        if last_policy_id:
            policy_id = last_policy_id

    policy = POLICY_STORE.get(policy_id)
    if policy is None:  # 如果策略不存在，尝试使用最后使用的策略
        last_policy_id = _read_last_policy_id()
        if last_policy_id and last_policy_id in POLICY_STORE:
            policy_id = last_policy_id
            policy = POLICY_STORE.get(policy_id)

    if policy is None:  # 如果仍然不存在，报错
        available = list(POLICY_STORE.keys())[-10:]
        raise ValueError(f"policy_id 不存在，请先调用 /simulation/ppo-train，可用 policy_id: {available}")

    # 执行策略
    noise_low = req.process_time_noise_low
    noise_high = req.process_time_noise_high
    final_state, _, metrics = _rollout(
        req=req,
        weights=policy["weights"],
        noise_low=noise_low,
        noise_high=noise_high,
        rng=random.Random(req.seed),
        sample=False,
        max_steps=req.max_steps,
    )

    return {
        "policy_id": policy_id,
        "metrics": metrics,
        "plan": _plan_from_history(final_state),
    }

def get_ppo_decision(state: Dict[str, Any], policy_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """供实时引擎调用的 PPO 单步决策函数。"""
    _load_policy_store()
    
    # 规范化 policy_id
    target_id = str(policy_id).strip().strip("\"'`") if policy_id else None
    placeholders = {"", "ppo_xxxxxxxxxx", "string", "your_policy_id", "None", None}
    
    if target_id in placeholders:
        target_id = _read_last_policy_id()
    
    policy = POLICY_STORE.get(target_id)
    if not policy:
        if POLICY_STORE:
            target_id = list(POLICY_STORE.keys())[-1]
            policy = POLICY_STORE[target_id]
        else:
            return None

    weights = policy["weights"]
    actions = _build_actions(state)
    if not actions:
        return None
    
    s_feat = _state_features(state)
    phis = [_compose_features(s_feat, a["features"]) for a in actions]
    logits = [_dot(weights, phi) for phi in phis]
    probs = _softmax(logits)
    
    selected_idx = max(range(len(probs)), key=lambda i: probs[i])
    return actions[selected_idx]["decision"]
