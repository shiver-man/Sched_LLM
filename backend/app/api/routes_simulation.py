from fastapi import APIRouter, HTTPException
from itertools import combinations
from typing import Any, Dict, List
from datetime import datetime
from app.models.schema import (
    SimulationRuleRequest,
    ScheduleRequest,
    SchedulePlanRequest,
    FailureRecoveryRequest,
    PPOTrainRequest,
    PPOPlanRequest,
    DynamicUncertaintyRequest,
    MultiStrategyResponse,
)
from app.models.state import build_initial_state
from app.core.simulator import Simulator
from app.core.scheduler import PDR
from app.core.evaluator import Evaluator
from app.core.dispatcher import Dispatcher
from app.core.ppo_scheduler import train_ppo_policy, run_ppo_policy, get_ppo_decision
from app.core.engine import EventEngine
from app.core.math_optimizer import MathOptimizer
from app.core.meta_heuristic import GeneticAlgorithm
from app.core.multi_strategy_scheduler import MultiStrategyScheduler
from app.llm.ollama_client import OllamaClient
from app.llm.prompt_builder import build_reflection_prompt, build_llm_plan_payload, build_llm_plan_brief, build_ollama_plan_prompt
from app.config import settings

router = APIRouter()
ollama_client = OllamaClient(model=settings.ollama_model)
SUPPORTED_RULES = ["SPT", "FIFO", "MWKR", "COOP", "COOP_RH"]


@router.post("/run", response_model=MultiStrategyResponse)
async def run_unified_simulation(payload: Dict[str, Any]):
    """
    统一调度实验平台接口。
    基于同一基础数据，同时运行多种策略，生成完整调度方案并进行横向对比。
    """
    mode = payload.get("mode", "compare_all")
    
    try:
        # 1. 规范化输入并构造请求对象
        req_data = _normalize_rich_payload_for_ppo(payload)
        req = ScheduleRequest(**req_data)
        
        # 获取配置参数
        ppo_id = payload.get("dispatching_config", {}).get("ppo_policy_id") or "latest"
        max_steps = int(payload.get("simulation_config", {}).get("ppo_max_steps", 1000))
        
        # 2. 初始化多策略调度平台
        platform = MultiStrategyScheduler(req, max_steps=max_steps)
        
        # 3. 执行多方案生成
        # 即使 mode 为 ppo_plan，也建议返回完整对比以增加实验价值
        # 但这里我们遵循用户逻辑：compare_all 为全量，其余为特定模式（此处简化为全部走全量）
        response = platform.execute_all(ppo_policy_id=ppo_id)
        llm_cfg = payload.get("llm_config", {}) or {}
        use_ollama = bool(llm_cfg.get("use_ollama", True))
        if use_ollama:
            try:
                best = response.summary_comparison[0] if response.summary_comparison else {}
                best_rule = best.get("rule")
                best_scheme = next((s for s in response.detailed_schemes if s.rule == best_rule), None)
                llm_payload = build_llm_plan_payload(
                    {
                        "objective": "makespan",
                        "best_rule": best_rule,
                        "best_metrics": best_scheme.metrics.model_dump() if best_scheme else {},
                        "best_schedule_plan": [x.model_dump() for x in (best_scheme.plan if best_scheme else [])],
                        "all_rule_results": [
                            {
                                "rule": s.rule,
                                "metrics": s.metrics.model_dump(),
                                "plan": [p.model_dump() for p in s.plan],
                            }
                            for s in response.detailed_schemes
                        ],
                    }
                )
                prompt = build_ollama_plan_prompt(llm_payload)
                model_text = ollama_client.generate(prompt)
                if model_text and model_text.strip():
                    response.llm_readable_brief = model_text.strip()
            except Exception:
                pass
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"调度实验平台运行失败: {str(e)}")


@router.post("/ga-transparent")
def run_ga_transparent(payload: Dict[str, Any]):
    try:
        req_data = _normalize_rich_payload_for_ppo(payload)
        req = ScheduleRequest(**req_data)
        state = build_initial_state(req)
        sim_cfg = payload.get("simulation_config", {}) or {}
        pop_size = int(payload.get("ga_pop_size", sim_cfg.get("ga_pop_size", 60)))
        generations = int(payload.get("ga_generations", sim_cfg.get("ga_generations", 30)))
        ga = GeneticAlgorithm(state, pop_size=pop_size, generations=generations, debug=True)
        result = ga.solve()
        return {
            "status": "success" if result.get("status") == "success" else "failed",
            "ga_result": result,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"GA 透明化运行失败: {str(e)}")


@router.post("/run-dynamic-scenarios")
def run_dynamic_scenarios(req: DynamicUncertaintyRequest):
    """
    运行多组代表性动态不确定场景。
    根据参数化模型自动生成扰动。
    """
    try:
        if not req.jobs:
            raise HTTPException(status_code=400, detail="jobs 不能为空")
        if not req.machines:
            raise HTTPException(status_code=400, detail="machines 不能为空")
        results = []
        base_seed = req.seed
        
        for i in range(req.num_scenarios):
            scenario_seed = base_seed + i
            # 复制请求数据并更新 seed
            scenario_payload = req.model_dump()
            scenario_payload["simulation_config"] = scenario_payload.get("simulation_config", {})
            scenario_payload["simulation_config"]["random_seed"] = scenario_seed
            
            # 构建初始状态
            initial_state = build_initial_state(req)
            # 将 uncertainty_config 放入 metadata 供引擎使用
            initial_state["metadata"]["uncertainty_config"] = req.uncertainty_config.model_dump()
            initial_state["metadata"]["simulation_config"] = {"random_seed": scenario_seed}
            initial_state["metadata"]["factory_info"] = req.model_dump().get("factory_info", {})
            
            # 定义策略函数
            if req.policy_type == "PPO":
                def policy_fn(state):
                    decision = get_ppo_decision(state, policy_id=req.policy_id)
                    if decision:
                        return decision
                    return PDR.get_dispatch_action(state, rule="COOP_RH")
            else:
                def policy_fn(state):
                    return PDR.get_dispatch_action(state, rule=req.policy_type)
            
            # 运行引擎
            engine = EventEngine(initial_state, policy_fn=policy_fn)
            engine.max_time = req.model_dump().get("factory_info", {}).get("planning_horizon", 1000.0)
            
            final_state = engine.run(max_steps=req.max_steps)
            
            # 收集结果
            metrics = Evaluator.evaluate(final_state)
            plan = [
                {
                    "step": idx + 1,
                    "job_id": h["job_id"],
                    "op_id": h["op_id"],
                    "machine_id": h["machine_id"],
                    "vehicle_id": h.get("vehicle_id"),
                    "start_time": round(h["start_time"], 2),
                    "finish_time": round(h["finish_time"], 2),
                    "transport_time": round(h.get("transport_time", 2), 2),
                    "process_time": round(h.get("process_time", 2), 2),
                }
                for idx, h in enumerate(final_state["history"])
            ]
            event_trace = final_state.get("event_trace", [])
            timeline_brief = [
                {
                    "time": round(float(e.get("time", 0.0)), 3),
                    "event_type": e.get("event_type"),
                    "job_id": (e.get("payload") or {}).get("job_id"),
                    "machine_id": (e.get("payload") or {}).get("machine_id"),
                    "vehicle_id": (e.get("payload") or {}).get("vehicle_id"),
                }
                for e in event_trace
            ]
            unfinished_jobs = [j["job_id"] for j in final_state["jobs"] if not j["finished"]]
            feasible = len(unfinished_jobs) == 0
            machine_busy = {}
            vehicle_busy = {}
            for h in final_state["history"]:
                machine_busy[h["machine_id"]] = machine_busy.get(h["machine_id"], 0.0) + (h["finish_time"] - h["start_time"])
                if h.get("vehicle_id"):
                    vehicle_busy[h["vehicle_id"]] = vehicle_busy.get(h["vehicle_id"], 0.0) + h.get("transport_time", 0.0)
            machine_bottlenecks = sorted(
                [{"machine_id": k, "busy_time": round(v, 3)} for k, v in machine_busy.items()],
                key=lambda x: x["busy_time"],
                reverse=True,
            )
            vehicle_bottlenecks = sorted(
                [{"vehicle_id": k, "transport_busy_time": round(v, 3)} for k, v in vehicle_busy.items()],
                key=lambda x: x["transport_busy_time"],
                reverse=True,
            )
            
            results.append({
                "scenario_id": i + 1,
                "seed": scenario_seed,
                "metrics": metrics,
                "plan": plan,
                "event_trace": event_trace,
                "timeline_brief": timeline_brief,
                "feasible": feasible,
                "unfinished_jobs": unfinished_jobs,
                "machine_bottlenecks": machine_bottlenecks,
                "vehicle_bottlenecks": vehicle_bottlenecks
            })
            
        # 计算平均指标
        avg_metrics = {
            "makespan": round(sum(r["metrics"]["makespan"] for r in results) / len(results), 2),
            "utilization": round(sum(r["metrics"]["utilization"] for r in results) / len(results), 4),
            "total_tardiness": round(sum(r["metrics"]["total_tardiness"] for r in results) / len(results), 2),
            "total_transport_time": round(sum(r["metrics"]["total_transport_time"] for r in results) / len(results), 2),
        }
        
        # 构建 LLM 可读简报 (选取第一个场景作为代表)
        representative_scenario = results[0]
        llm_payload = build_llm_plan_payload({
            "objective": "makespan",
            "best_rule": req.policy_type,
            "best_metrics": representative_scenario["metrics"],
            "best_schedule_plan": representative_scenario["plan"],
            "all_rule_results": [{"rule": req.policy_type, "metrics": representative_scenario["metrics"], "plan": representative_scenario["plan"]}]
        })
        
        return {
            "status": "success",
            "num_scenarios": req.num_scenarios,
            "policy_type": req.policy_type,
            "average_metrics": avg_metrics,
            "llm_readable_payload": llm_payload,
            "llm_readable_brief": build_llm_plan_brief(llm_payload),
            "scenarios": results
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"动态场景仿真失败: {str(e)}")


def _normalize_rich_payload_for_dynamic(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("machines") and payload.get("layout"):
        sim_cfg = payload.get("simulation_config", {}) or {}
        out = dict(payload)
        out["policy_type"] = str(payload.get("policy_type", "PPO")).upper()
        out["num_scenarios"] = int(payload.get("num_scenarios", 5))
        out["max_steps"] = int(payload.get("max_steps", sim_cfg.get("ppo_max_steps", 1000)))
        out["seed"] = int(payload.get("seed", sim_cfg.get("random_seed", 42)))
        if "policy_id" not in out:
            out["policy_id"] = payload.get("dispatching_config", {}).get("ppo_policy_id")
        return out

    normalized = _normalize_rich_payload_for_ppo(payload)
    sim_cfg = payload.get("simulation_config", {}) or {}
    return {
        "jobs": normalized["jobs"],
        "machines": normalized["machines"],
        "vehicles": normalized["vehicles"],
        "layout": normalized["layout"],
        "current_time": normalized["current_time"],
        "strategic_experience": normalized["strategic_experience"],
        "metadata": normalized["metadata"],
        "uncertainty_config": payload.get("uncertainty_config", {}),
        "num_scenarios": int(payload.get("num_scenarios", 5)),
        "max_steps": int(payload.get("max_steps", sim_cfg.get("ppo_max_steps", 1000))),
        "policy_type": str(payload.get("policy_type", "PPO")).upper(),
        "policy_id": payload.get("policy_id", payload.get("dispatching_config", {}).get("ppo_policy_id")),
        "seed": int(payload.get("seed", sim_cfg.get("random_seed", 42))),
    }


@router.post("/run-dynamic-scenarios-rich")
def run_dynamic_scenarios_rich(payload: Dict[str, Any]):
    try:
        req = DynamicUncertaintyRequest(**_normalize_rich_payload_for_dynamic(payload))
        return run_dynamic_scenarios(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"动态场景 rich 仿真失败: {str(e)}")


def _run_rule_plan(req: ScheduleRequest, rule: str, max_steps: int):
    initial_state = build_initial_state(req)

    def pdr_policy(state):
        return PDR.get_dispatch_action(state, rule=rule)

    final_state = Simulator.run_simulation(initial_state, pdr_policy, max_steps=max_steps)
    metrics = Evaluator.evaluate(final_state)
    plan = [
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
        for idx, h in enumerate(final_state["history"])
    ]

    return {"rule": rule, "metrics": metrics, "plan": plan}


def _to_layout_from_matrix(matrix: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    nodes = list(matrix.keys())
    edges: List[Dict[str, Any]] = []
    for i, src in enumerate(nodes):
        row = matrix.get(src, {})
        for j, dst in enumerate(nodes):
            if i >= j:
                continue
            dist = row.get(dst)
            if dist is None:
                continue
            if dist <= 0:
                continue
            edges.append({"from_node": src, "to_node": dst, "distance": float(dist)})
    return {"nodes": nodes, "edges": edges, "directed": False}


def _extract_global_noise_range(payload: Dict[str, Any]) -> (float, float):
    low = 0.8
    high = 1.2
    machines = payload.get("shop_floor", {}).get("machines", [])
    candidate_lows = []
    candidate_highs = []
    for m in machines:
        u = m.get("processing_time_uncertainty", {}) or {}
        if "min_factor" in u:
            candidate_lows.append(float(u["min_factor"]))
        if "low_factor" in u:
            candidate_lows.append(float(u["low_factor"]))
        if "max_factor" in u:
            candidate_highs.append(float(u["max_factor"]))
        if "high_factor" in u:
            candidate_highs.append(float(u["high_factor"]))
    if candidate_lows:
        low = min(candidate_lows)
    if candidate_highs:
        high = max(candidate_highs)
    if low <= 0:
        low = 0.8
    if high <= 0:
        high = 1.2
    if low >= high:
        high = max(low + 0.1, 1.1)
    return low, high


def _normalize_rich_payload_for_ppo(payload: Dict[str, Any]) -> Dict[str, Any]:
    def _to_float_time(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        try:
            return float(text)
        except Exception:
            pass
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return float(dt.timestamp())
        except Exception:
            return default

    factory = payload.get("factory_info", {}) or {}
    floor = payload.get("shop_floor", {}) or {}
    jobs_raw = payload.get("jobs", []) or []
    machines_raw = floor.get("machines", []) or []
    vehicles_raw = floor.get("vehicles", []) or []

    network = floor.get("transport_network", {}) or {}
    matrix = network.get("travel_time_matrix", {}) or {}
    if matrix:
        layout = _to_layout_from_matrix(matrix)
    else:
        layout = {
            "nodes": network.get("nodes", []),
            "edges": network.get("edges", []),
            "directed": False,
        }

    jobs = []
    for job in jobs_raw:
        initial_location = job.get("initial_location") or job.get("current_location") or "L/U"
        operations = []
        for op in job.get("operations", []):
            cms = []
            for cm in op.get("candidate_machines", []):
                # 兼容更多可能的字段名
                process_time = cm.get("process_time", cm.get("base_processing_time", cm.get("processing_time", 1.0)))
                cms.append(
                    {
                        "machine_id": cm["machine_id"],
                        "process_time": float(process_time),
                    }
                )
            operations.append(
                {
                    "op_id": op.get("op_id", op.get("operation_id", "")),
                    "source_location": job.get("current_location", initial_location),
                    "candidate_machines": cms,
                }
            )
        jobs.append(
            {
                "job_id": job["job_id"],
                "operations": operations,
                "release_time": float(job.get("release_time", 0.0)),
                "due_time": float(job.get("due_time", 10**9)),
                "initial_location": initial_location,
            }
        )

    machines = [
        {
            "machine_id": m["machine_id"],
            "machine_type": m.get("type"),
            "location": m.get("location", m["machine_id"]),
            "status": m.get("status", "idle"),
            "available_time": float(m.get("available_from", 0.0)),
            "current_job": None,
        }
        for m in machines_raw
    ]

    vehicles = [
        {
            "vehicle_id": v["vehicle_id"],
            "current_location": v.get("current_location", v.get("start_location", "L/U")),
            "speed": float(v.get("speed", v.get("speed_m_per_min", 1.0))),
            "capacity": int(v.get("capacity", 1)),
            "load_unload_time": float(v.get("load_unload_time", 0.0)),
            "status": v.get("status", "idle"),
            "available_time": float(v.get("available_time", 0.0)),
            "current_task": None,
        }
        for v in vehicles_raw
    ]

    low, high = _extract_global_noise_range(payload)
    sim_cfg = payload.get("simulation_config", {}) or {}

    return {
        "factory_info": factory,
        "jobs": jobs,
        "machines": machines,
        "vehicles": vehicles,
        "layout": layout,
        "current_time": _to_float_time(factory.get("current_time", 0.0), 0.0),
        "strategic_experience": "rich_json_dynamic_scheduling",
        "metadata": {
            "factory_id": factory.get("factory_id"),
            "factory_name": factory.get("factory_name"),
            "factory_info": factory,
            "dynamic_events": payload.get("dynamic_events", []),
            "dispatching_config": payload.get("dispatching_config", {}),
            "simulation_config": sim_cfg,
            "objectives": payload.get("objectives", {}),
            "uncertainty_model": payload.get("uncertainty_model", {}),
            "_raw_machines": machines_raw,
        },
        "episodes": int(sim_cfg.get("ppo_episodes", 200)),
        "max_steps": int(sim_cfg.get("ppo_max_steps", 1000)),
        "gamma": float(sim_cfg.get("ppo_gamma", 0.99)),
        "clip_ratio": float(sim_cfg.get("ppo_clip_ratio", 0.2)),
        "learning_rate": float(sim_cfg.get("ppo_learning_rate", 0.01)),
        "update_epochs": int(sim_cfg.get("ppo_update_epochs", 4)),
        "process_time_noise_low": float(sim_cfg.get("ppo_noise_low", low)),
        "process_time_noise_high": float(sim_cfg.get("ppo_noise_high", high)),
        "seed": int(sim_cfg.get("random_seed", 42)),
    }


def _pick_best(results, objective: str):
    objective = objective.lower()
    if objective == "utilization":
        return max(results, key=lambda x: x["metrics"]["utilization"])
    if objective == "total_tardiness":
        return min(results, key=lambda x: x["metrics"]["total_tardiness"])
    if objective == "total_transport_time":
        return min(results, key=lambda x: x["metrics"]["total_transport_time"])
    if objective == "num_events":
        return min(results, key=lambda x: x["metrics"]["num_events"])
    return min(results, key=lambda x: x["metrics"]["makespan"])


def _pick_best_with_feasibility(results, objective: str):
    feasible = [x for x in results if x.get("feasible", True)]
    target = feasible if feasible else results
    return _pick_best(target, objective)


def _to_plan_result_like_objective(scenario_result: dict, objective: str):
    return {
        "status": "success",
        "objective": objective,
        "evaluated_rules": [x["rule"] for x in scenario_result["all_rule_results"]],
        "best_rule": scenario_result["best_rule"],
        "best_metrics": scenario_result["best_metrics"],
        "best_schedule_plan": scenario_result["best_schedule_plan"],
        "all_rule_results": [
            {
                "rule": x["rule"],
                "metrics": x["metrics"],
                "plan": x["plan"],
            }
            for x in scenario_result["all_rule_results"]
        ],
    }


def _collect_future_times(state):
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
    return future_times


def _apply_machine_failures(state, failed_machine_ids):
    failed_set = set(failed_machine_ids)
    for machine in state["machines"]:
        if machine["machine_id"] not in failed_set:
            continue
        if machine["status"] == "busy":
            machine["down_after_busy"] = True
            continue
        machine["status"] = "down"
        machine["current_job"] = None


def _run_rule_plan_with_failures(req: ScheduleRequest, rule: str, max_steps: int, fault_time: float, failed_machine_ids):
    state = build_initial_state(req)
    step = 0
    failures_applied = False

    while not all(j["finished"] for j in state["jobs"]) and step < max_steps:
        if not failures_applied and state["time"] >= fault_time:
            _apply_machine_failures(state, failed_machine_ids)
            failures_applied = True

        Simulator._release_resources(state)
        if failures_applied:
            _apply_machine_failures(state, failed_machine_ids)

        decision = PDR.get_dispatch_action(state, rule=rule)
        if decision:
            state = Dispatcher.apply_decision(state, decision)
            step += 1
            continue

        if not failures_applied and state["time"] < fault_time:
            future_times = _collect_future_times(state)
            if future_times and min(future_times) > fault_time:
                state["time"] = fault_time
                _apply_machine_failures(state, failed_machine_ids)
                failures_applied = True
                continue

        moved = Simulator._advance_time(state)
        if not moved:
            break
        step += 1

    if not failures_applied and state["time"] >= fault_time:
        _apply_machine_failures(state, failed_machine_ids)
    Simulator._release_resources(state)
    if failures_applied:
        _apply_machine_failures(state, failed_machine_ids)

    final_finish = 0.0
    if state["history"]:
        final_finish = max(h["finish_time"] for h in state["history"])
    state["time"] = max(state["time"], final_finish)

    metrics = Evaluator.evaluate(state)
    plan = [
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
    unfinished_jobs = [j["job_id"] for j in state["jobs"] if not j["finished"]]
    feasible = len(unfinished_jobs) == 0

    return {
        "rule": rule,
        "metrics": metrics,
        "plan": plan,
        "feasible": feasible,
        "unfinished_jobs": unfinished_jobs,
    }


def run_trajectory(req: SimulationRuleRequest):
    try:
        result = _run_rule_plan(req, req.rule, req.max_steps)

        return {
            "rule": result["rule"],
            "metrics": result["metrics"],
            "history_summary": result["plan"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_schedule_plan(req: SchedulePlanRequest):
    try:
        requested_rules = [rule.upper() for rule in req.rules]
        supported_rules = SUPPORTED_RULES
        rules = [rule for rule in requested_rules if rule in supported_rules]
        if not rules:
            rules = supported_rules

        results = [_run_rule_plan(req, rule, req.max_steps) for rule in rules]
        best = _pick_best(results, req.objective)
        raw_result = {
            "status": "success",
            "objective": req.objective,
            "evaluated_rules": rules,
            "best_rule": best["rule"],
            "best_metrics": best["metrics"],
            "best_schedule_plan": best["plan"],
            "all_rule_results": results,
        }
        llm_payload = build_llm_plan_payload(raw_result)
        raw_result["llm_readable_payload"] = llm_payload
        raw_result["llm_readable_brief"] = build_llm_plan_brief(llm_payload)
        return raw_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调度方案生成失败: {str(e)}")


def generate_failure_recovery_plan(req: FailureRecoveryRequest):
    try:
        machine_ids = [m.machine_id for m in req.machines]
        candidate_ids = req.candidate_failed_machines or machine_ids
        candidate_ids = [m for m in candidate_ids if m in machine_ids]
        if not candidate_ids:
            raise ValueError("candidate_failed_machines 为空或不在 machines 列表中")

        requested_rules = [rule.upper() for rule in req.rules]
        rules = [rule for rule in requested_rules if rule in SUPPORTED_RULES]
        if not rules:
            rules = SUPPORTED_RULES

        max_failed = min(req.max_failed_machines, len(candidate_ids))
        scenarios = []
        if req.include_no_failure:
            scenarios.append([])
        for k in range(1, max_failed + 1):
            for combo in combinations(candidate_ids, k):
                scenarios.append(list(combo))
                if len(scenarios) >= req.max_scenarios:
                    break
            if len(scenarios) >= req.max_scenarios:
                break

        scenario_results = []
        for failed in scenarios:
            rule_results = [
                _run_rule_plan_with_failures(
                    req=req,
                    rule=rule,
                    max_steps=req.max_steps,
                    fault_time=req.fault_time,
                    failed_machine_ids=failed,
                )
                for rule in rules
            ]
            best = _pick_best_with_feasibility(rule_results, req.objective)
            scenario_results.append(
                {
                    "failed_machines": failed,
                    "feasible": best["feasible"],
                    "best_rule": best["rule"],
                    "best_metrics": best["metrics"],
                    "best_schedule_plan": best["plan"],
                    "unfinished_jobs": best["unfinished_jobs"],
                    "all_rule_results": rule_results,
                }
            )
            scenario_plan_like = _to_plan_result_like_objective(scenario_results[-1], req.objective)
            scenario_llm_payload = build_llm_plan_payload(scenario_plan_like)
            scenario_results[-1]["llm_readable_payload"] = scenario_llm_payload
            scenario_results[-1]["llm_readable_brief"] = build_llm_plan_brief(scenario_llm_payload)

        best_scenarios = [x for x in scenario_results if x["feasible"]]
        global_best = None
        if best_scenarios:
            global_best = _pick_best(
                [{"rule": str(x["failed_machines"]), "metrics": x["best_metrics"], "data": x} for x in best_scenarios],
                req.objective,
            )["data"]

        result = {
            "status": "success",
            "objective": req.objective,
            "fault_time": req.fault_time,
            "evaluated_rules": rules,
            "evaluated_scenarios": len(scenario_results),
            "scenario_results": scenario_results,
            "global_best_scenario": global_best,
        }
        llm_source = global_best if global_best else (scenario_results[0] if scenario_results else None)
        if llm_source:
            llm_plan_like = _to_plan_result_like_objective(llm_source, req.objective)
            llm_payload = build_llm_plan_payload(llm_plan_like)
            result["llm_readable_payload"] = llm_payload
            result["llm_readable_brief"] = build_llm_plan_brief(llm_payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"故障恢复仿真失败: {str(e)}")


@router.post("/ppo-train")
def ppo_train(req: PPOTrainRequest):
    try:
        result = train_ppo_policy(req)
        return {
            "status": "success",
            "algorithm": "PPO",
            "policy_id": result["policy_id"],
            "episodes": result["episodes"],
            "final_metrics": result["final_metrics"],
            "final_plan": result["final_plan"],
            "reward_history": result["reward_history"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PPO 训练失败: {str(e)}")


@router.post("/ppo-train-rich")
def ppo_train_rich(payload: Dict[str, Any]):
    try:
        normalized = _normalize_rich_payload_for_ppo(payload)
        req = PPOTrainRequest(**normalized)
        return ppo_train(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PPO rich 训练失败: {str(e)}")


@router.post("/ppo-plan")
def ppo_plan(req: PPOPlanRequest):
    try:
        result = run_ppo_policy(req)
        plan_result = {
            "status": "success",
            "objective": "makespan",
            "evaluated_rules": ["PPO"],
            "best_rule": "PPO",
            "best_metrics": result["metrics"],
            "best_schedule_plan": result["plan"],
            "all_rule_results": [
                {
                    "rule": "PPO",
                    "metrics": result["metrics"],
                    "plan": result["plan"],
                }
            ],
            "policy_id": result["policy_id"],
        }
        llm_payload = build_llm_plan_payload(plan_result)
        plan_result["llm_readable_payload"] = llm_payload
        plan_result["llm_readable_brief"] = build_llm_plan_brief(llm_payload)
        return plan_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PPO 调度失败: {str(e)}")


@router.post("/ppo-plan-rich")
def ppo_plan_rich(payload: Dict[str, Any]):
    """使用 Rich JSON payload 运行 PPO 调度。"""
    try:
        req_data = _normalize_rich_payload_for_ppo(payload)
        req = PPOPlanRequest(**req_data)
        return run_ppo_policy(req)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PPO Rich 调度失败: {str(e)}")


@router.post("/realtime-engine")
def run_realtime_simulation(payload: Dict[str, Any]):
    """
    实时事件驱动调度引擎。
    支持：
    1. 动态事件注入 (故障、新订单)
    2. 加工时间波动 (VPT)
    3. PPO 实时重调度
    """
    try:
        # 1. 规范化输入
        req_data = _normalize_rich_payload_for_ppo(payload)
        req = ScheduleRequest(**req_data)
        initial_state = build_initial_state(req)
        
        # 2. 定义调度策略 (使用 PPO)
        policy_id = payload.get("dispatching_config", {}).get("ppo_policy_id")
        fallback_rule = str(payload.get("dispatching_config", {}).get("fallback_rule", "COOP_RH")).upper()
        
        def ppo_policy_wrapper(state):
            decision = get_ppo_decision(state, policy_id=policy_id)
            if decision:
                return decision
            rule = fallback_rule if fallback_rule in SUPPORTED_RULES else "COOP_RH"
            return PDR.get_dispatch_action(state, rule=rule)
            
        # 3. 启动引擎
        engine = EventEngine(initial_state, policy_fn=ppo_policy_wrapper)
        
        # 设置仿真参数
        sim_cfg = payload.get("simulation_config", {})
        max_steps = int(sim_cfg.get("ppo_max_steps", 1000))
        engine.max_time = float(payload.get("factory_info", {}).get("planning_horizon", 200.0))
        
        final_state = engine.run(max_steps=max_steps)
        
        # 4. 评估与输出
        metrics = Evaluator.evaluate(final_state)
        plan = [
            {
                "step": idx + 1,
                "job_id": h["job_id"],
                "op_id": h["op_id"],
                "machine_id": h["machine_id"],
                "vehicle_id": h.get("vehicle_id"),
                "start_time": round(h["start_time"], 2),
                "finish_time": round(h["finish_time"], 2),
                "transport_time": round(h.get("transport_time", 0), 2),
                "process_time": round(h.get("process_time", 0), 2),
            }
            for idx, h in enumerate(final_state["history"])
        ]
        
        plan_result = {
            "status": "success",
            "objective": payload.get("objectives", {}).get("primary", "makespan"),
            "best_rule": "PPO-Realtime",
            "best_metrics": metrics,
            "best_schedule_plan": plan,
            "all_rule_results": [{"rule": "PPO-Realtime", "metrics": metrics, "plan": plan}]
        }
        llm_payload = build_llm_plan_payload(plan_result)
        
        return {
            "status": "success",
            "engine": "DiscreteEventEngine",
            "metrics": metrics,
            "schedule_plan": plan,
            "llm_readable_payload": llm_payload,
            "llm_readable_brief": build_llm_plan_brief(llm_payload)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"实时引擎运行失败: {str(e)}")


def run_comparative_study(payload: Dict[str, Any]):
    """
    运行多方案调度生成系统。
    基于同一基础数据，同时生成并输出多种调度策略的完整排产方案。
    """
    try:
        # 1. 规范化输入并构造请求对象
        req_data = _normalize_rich_payload_for_ppo(payload)
        req = ScheduleRequest(**req_data)
        
        # 2. 初始化多策略调度器
        scheduler = MultiStrategyScheduler(req)
        
        # 获取配置参数
        ppo_policy_id = payload.get("dispatching_config", {}).get("ppo_policy_id") or "latest"
        
        # 3. 批量生成所有调度方案 (详细层)
        schemes = scheduler.generate_all_schemes(ppo_policy_id=ppo_policy_id)
        
        if not schemes:
            raise HTTPException(status_code=500, detail="未能生成任何有效的调度方案")
            
        # 4. 生成汇总对比数据 (汇总层)
        summary = scheduler.get_summary_comparison(schemes)
        
        # 5. 构建 LLM 简报 (基于最优方案)
        best_scheme = schemes[0] if schemes else None
        llm_payload = build_llm_plan_payload({
            "objective": "makespan",
            "best_rule": best_scheme.rule if best_scheme else "N/A",
            "best_metrics": best_scheme.metrics.model_dump() if best_scheme else {},
            "best_schedule_plan": [step.model_dump() for step in best_scheme.plan] if best_scheme else [],
            "all_rule_results": [
                {
                    "rule": s.rule,
                    "metrics": s.metrics.model_dump(),
                    "plan": [step.model_dump() for step in s.plan]
                }
                for s in schemes
            ]
        })
        
        # 6. 返回双层结构结果
        return MultiStrategyResponse(
            status="success",
            detailed_schemes=schemes,
            summary_comparison=summary,
            llm_readable_brief=build_llm_plan_brief(llm_payload)
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"多方案调度生成失败: {str(e)}")


def reflect_on_trajectories(req: ScheduleRequest):
    try:
        trajectories = []
        for rule in ["SPT", "FIFO", "MWKR"]:
            tmp_req = SimulationRuleRequest(**req.model_dump(), rule=rule, max_steps=1000)
            result = run_trajectory(tmp_req)
            trajectories.append(result)

        prompt = build_reflection_prompt(trajectories)
        reflection_result = ollama_client.generate(prompt)
        return {
            "trajectories": trajectories,
            "reflection": reflection_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"反思失败: {str(e)}")
