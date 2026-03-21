import heapq
import random
import json
import copy
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import networkx as nx

from app.models.state import build_initial_state, get_dispatchable_jobs
from app.core.dispatcher import Dispatcher
from app.core.evaluator import Evaluator
from app.core.scheduler import PDR
from app.core.transport_scheduler import TransportScheduler

class EventType(Enum):
    JOB_RELEASE = "job_release"
    MACHINE_BREAKDOWN = "machine_breakdown"
    MACHINE_RECOVERY = "machine_recovery"
    TRANSPORT_FINISH = "transport_finish"
    PROCESS_FINISH = "process_finish"
    AGV_LOW_BATTERY = "agv_low_battery"
    AGV_RECHARGE_FINISH = "agv_recharge_finish"

class Event:
    def __init__(self, time: float, event_type: EventType, payload: Dict[str, Any]):
        self.time = time
        self.type = event_type
        self.payload = payload

    def __lt__(self, other):
        return self.time < other.time

def sample_value(base: float, low: float, high: float, dist: str, rng: random.Random) -> float:
    """根据波动范围和分布类型采样。"""
    if dist == "normal":
        # 假设 [low, high] 覆盖了 95% 的概率 (2 sigma)
        mean = (low + high) / 2.0
        std = (high - low) / 4.0
        factor = rng.gauss(mean, std)
    else: # uniform
        factor = rng.uniform(low, high)
    return base * max(0.1, factor)

class UncertaintyGenerator:
    """产生不确定性事件：机器故障、随机加工时间波动、新订单到达。"""

    @staticmethod
    def generate_events(state: Dict[str, Any], metadata: Dict[str, Any]) -> List[Event]:
        events = []
        seed = metadata.get("simulation_config", {}).get("random_seed", 42)
        rng = random.Random(seed)
        planning_horizon = metadata.get("factory_info", {}).get("planning_horizon", 200.0)
        
        # 获取不确定性配置 (从新的 schema 结构中获取)
        unc_cfg = metadata.get("uncertainty_config", {})
        breakdown_cfg = unc_cfg.get("breakdown", {})
        
        # 1. 处理显式预设的事件 (如果有)
        dynamic_events = metadata.get("dynamic_events", [])
        for de in dynamic_events:
            if de.get("event_type") == "machine_breakdown":
                details = de.get("details", {})
                events.append(Event(
                    time=de["time"],
                    event_type=EventType.MACHINE_BREAKDOWN,
                    payload={
                        "machine_id": details.get("machine_id"),
                        "repair_duration": details.get("repair_duration", 10.0)
                    }
                ))
            elif de.get("event_type") == "new_order_arrival":
                events.append(Event(
                    time=de["time"],
                    event_type=EventType.JOB_RELEASE,
                    payload={"job_data": de.get("details", {}).get("job")}
                ))

        # 2. 自动生成：机器随机故障 (基于 MachineBreakdownModel)
        if breakdown_cfg.get("enabled", False):
            for machine in state["machines"]:
                # 按照概率决定是否发生故障
                if rng.random() < breakdown_cfg.get("breakdown_probability", 0.1):
                    mtbf = breakdown_cfg.get("mean_time_to_failure", 150.0)
                    mttr = breakdown_cfg.get("mean_repair_time", 20.0)
                    
                    # 生成故障序列
                    current_t = rng.expovariate(1.0 / mtbf)
                    while current_t < planning_horizon:
                        events.append(Event(
                            time=current_t,
                            event_type=EventType.MACHINE_BREAKDOWN,
                            payload={
                                "machine_id": machine["machine_id"],
                                "repair_duration": max(5.0, rng.gauss(mttr, mttr * 0.2))
                            }
                        ))
                        current_t += mttr + rng.expovariate(1.0 / mtbf)

        # 3. 自动生成：基于泊松分布的新订单到达 (保留原有逻辑，但可以后续参数化)
        arrival_cfg = unc_cfg.get("order_arrival", {})
        if arrival_cfg.get("enabled", False) and state.get("machines") and not any(de.get("event_type") == "new_order_arrival" for de in dynamic_events):
            arrival_rate = float(arrival_cfg.get("arrival_rate", 1.0 / 60.0))
            current_t = rng.expovariate(arrival_rate)
            while current_t < planning_horizon:
                new_job_id = f"J_AUTO_{int(current_t)}"
                events.append(Event(
                    time=current_t,
                    event_type=EventType.JOB_RELEASE,
                    payload={
                        "job_data": {
                            "job_id": new_job_id,
                            "release_time": current_t,
                            "due_time": current_t + rng.uniform(40, 100),
                            "initial_location": "L/U",
                            "operations": [
                                {
                                    "operation_id": f"{new_job_id}_O1",
                                    "candidate_machines": [
                                        {
                                            "machine_id": rng.choice(state["machines"])["machine_id"],
                                            "base_processing_time": rng.uniform(5, 15)
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                ))
                current_t += rng.expovariate(arrival_rate)

        return sorted(events, key=lambda x: x.time)

class StateManager:
    """管理车间状态的更新逻辑。"""
    
    def __init__(self, state: Dict[str, Any]):
        self.state = state

    def update_machine_status(self, machine_id: str, status: str):
        for m in self.state["machines"]:
            if m["machine_id"] == machine_id:
                m["status"] = status
                break

    def update_job_location(self, job_id: str, location: str):
        for j in self.state["jobs"]:
            if j["job_id"] == job_id:
                j["current_location"] = location
                break

    def mark_job_finished(self, job_id: str):
        for j in self.state["jobs"]:
            if j["job_id"] == job_id:
                j["finished"] = True
                break

    def add_history(self, entry: Dict[str, Any]):
        self.state["history"].append(entry)

class EventEngine:
    """离散事件驱动引擎 (DES Core)。"""

    def __init__(self, initial_state: Dict[str, Any], policy_fn: Optional[callable] = None):
        self.state = initial_state
        self.manager = StateManager(self.state)
        self.event_queue = []
        self.policy_fn = policy_fn
        if "event_trace" not in self.state:
            self.state["event_trace"] = []
        
        metadata = self.state.get("metadata", {})
        seed = metadata.get("simulation_config", {}).get("random_seed", 42)
        self.rng = random.Random(seed)
        
        raw_horizon = metadata.get("factory_info", {}).get("planning_horizon", 1000.0)
        try:
            parsed_horizon = float(raw_horizon)
        except Exception:
            parsed_horizon = 1000.0
        self.max_time = parsed_horizon if parsed_horizon > 0 else 1000.0
        transport_rule = metadata.get("dispatching_config", {}).get("transport_rule", "NEAREST_VEHICLE")
        self.transport_scheduler = TransportScheduler(self.state, strategy=transport_rule)
        
        # 初始化事件队列
        initial_events = UncertaintyGenerator.generate_events(self.state, metadata)
        for e in initial_events:
            heapq.heappush(self.event_queue, e)

    def schedule_event(self, event: Event):
        heapq.heappush(self.event_queue, event)

    def run(self, max_steps: int = 2000) -> Dict[str, Any]:
        step = 0
        while step < max_steps and self.state["time"] < self.max_time:
            queued = self._try_assign_transport_queue()
            if queued:
                step += 1
                continue
            # 1. 尝试派工 (只有在资源空闲且有待办工件时)
            if self._can_dispatch() and self.policy_fn:
                decision = self.policy_fn(self.state)
                if decision:
                    # 前瞻评估
                    lookahead_score = self._lookahead_eval(decision)
                    decision["lookahead_score"] = lookahead_score
                    
                    applied = self._apply_dispatch(decision)
                    if applied:
                        self.state["event_trace"].append(
                            {
                                "time": self.state["time"],
                                "event_type": "dispatch_decision",
                                "payload": decision,
                            }
                        )
                        step += 1
                        continue
            
            # 2. 推进时间到下一个事件
            if self.event_queue:
                event = heapq.heappop(self.event_queue)
                # 确保时间不倒退，且推进到下一个有意义的时刻
                self.state["time"] = max(self.state["time"], event.time)
                self._handle_event(event)
                step += 1
            else:
                # 如果没有待处理事件，检查是否所有任务已完成
                if all(j.get("finished") for j in self.state["jobs"]):
                    break
                self._track_idle_reasons()
                
                # 如果任务没完成且没事件，尝试强行释放可能已到期的资源
                idle_before = len([m for m in self.state["machines"] if m["status"] == "idle"])
                self._force_release_resources()
                idle_after = len([m for m in self.state["machines"] if m["status"] == "idle"])
                
                if idle_after > idle_before:
                    continue # 释放了新资源，重新尝试派工
                
                # 实在没辙了（死锁或配置错误）
                break

        return self.state

    def _try_assign_transport_queue(self) -> bool:
        if not self.state.get("transport_queue"):
            return False
        queued = self.transport_scheduler.pop_assignable_queue()
        if not queued:
            return False
        job = next((j for j in self.state["jobs"] if j["job_id"] == queued["job_id"]), None)
        machine = next((m for m in self.state["machines"] if m["machine_id"] == queued["machine_id"]), None)
        if not job or not machine or job.get("finished") or job.get("locked"):
            return False
        if machine.get("status") != "idle":
            self.transport_scheduler.enqueue_transport(queued)
            return False
        vehicle_id = self.transport_scheduler.assign_vehicle(
            queued["job_id"], queued["op_id"], job["current_location"], machine["location"]
        )
        if not vehicle_id:
            self.transport_scheduler.enqueue_transport(queued)
            return False
        wait = max(0.0, self.state["time"] - float(queued.get("enqueued_time", self.state["time"])))
        self.state["transport_stats"]["total_wait_time"] += wait
        queued_decision = {
            "job_id": queued["job_id"],
            "op_id": queued["op_id"],
            "machine_id": queued["machine_id"],
            "vehicle_id": vehicle_id,
            "reason": "transport-queue-dispatch",
        }
        return self._apply_dispatch(queued_decision)

    def _track_idle_reasons(self):
        stats = self.state.setdefault("idle_reason_stats", {"waiting_transport": 0, "waiting_machine": 0, "waiting_repair": 0})
        dispatchable = [j for j in get_dispatchable_jobs(self.state) if not j.get("locked", False)]
        idle_machines = [m for m in self.state["machines"] if m["status"] == "idle"]
        down_machines = [m for m in self.state["machines"] if m["status"] == "down"]
        idle_vehicles = [v for v in self.state["vehicles"] if v.get("status") == "idle"]
        if down_machines:
            stats["waiting_repair"] += 1
        if dispatchable and not idle_machines:
            stats["waiting_machine"] += 1
        if dispatchable and idle_machines and (self.state.get("transport_queue") or not idle_vehicles):
            stats["waiting_transport"] += 1

    def _force_release_resources(self):
        """强制检查并释放所有已到达可用时间的资源。"""
        now = self.state["time"]
        for m in self.state["machines"]:
            if m["status"] == "busy" and m.get("available_time", 0) <= now:
                m["status"] = "idle"
        for v in self.state["vehicles"]:
            if v["status"] == "busy" and v.get("available_time", 0) <= now:
                v["status"] = "idle"

    def _can_dispatch(self) -> bool:
        """检查当前是否有可派工的工件和空闲机器。"""
        dispatchable = [j for j in get_dispatchable_jobs(self.state) if not j.get("locked", False)]
        idle_machines = [m for m in self.state["machines"] if m["status"] == "idle"]
        return len(dispatchable) > 0 and len(idle_machines) > 0

    def _handle_event(self, event: Event):
        t = event.type
        p = event.payload
        self.state["event_trace"].append(
            {
                "time": event.time,
                "event_type": t.value,
                "payload": p,
            }
        )
        
        if t == EventType.MACHINE_BREAKDOWN:
            self.manager.update_machine_status(p["machine_id"], "down")
            # 如果机器正在忙，可能需要处理中断逻辑（此处简化为等待其完成）
            # 生成恢复事件
            recovery_time = self.state["time"] + p["repair_duration"]
            self.schedule_event(Event(recovery_time, EventType.MACHINE_RECOVERY, {"machine_id": p["machine_id"]}))
            
        elif t == EventType.MACHINE_RECOVERY:
            self.manager.update_machine_status(p["machine_id"], "idle")
            
        elif t == EventType.PROCESS_FINISH:
            # 工序加工完成
            job_id = p["job_id"]
            machine_id = p["machine_id"]
            
            # 记录历史
            self.manager.add_history(p["history_entry"])
            
            # 更新机器状态
            self.manager.update_machine_status(machine_id, "idle")
            
            # 更新工件状态
            for j in self.state["jobs"]:
                if j["job_id"] == job_id:
                    j["locked"] = False
                    j["current_op_index"] += 1
                    j["current_location"] = p["location"]
                    if j["current_op_index"] >= len(j["operations"]):
                        j["finished"] = True
                    else:
                        j["ready_time"] = self.state["time"]
                    break
                    
        elif t == EventType.TRANSPORT_FINISH:
            # 运输完成
            job_id = p["job_id"]
            vehicle_id = p["vehicle_id"]
            
            # 更新车辆状态
            for v in self.state["vehicles"]:
                if v["vehicle_id"] == vehicle_id:
                    v["status"] = "idle"
                    v["current_location"] = p["destination"]
                    v["available_time"] = self.state["time"]
                    self.transport_scheduler.release_vehicle_task(
                        vehicle_id=v["vehicle_id"],
                        transport_time=max(0.0, p.get("transport_time", 0.0)),
                        from_loc=p.get("source", ""),
                        to_loc=p.get("destination", "")
                    )
                    break
            
            # 更新工件位置
            for j in self.state["jobs"]:
                if j["job_id"] == job_id:
                    j["current_location"] = p["destination"]
                    j["ready_time"] = self.state["time"]
                    break

        elif t == EventType.JOB_RELEASE:
            # 新工件到达
            job_data = p["job_data"]
            if job_data:
                # 将新工件加入 state
                new_job = {
                    "job_id": job_data["job_id"],
                    "operations": [
                        {
                            "op_id": op["operation_id"],
                            "candidate_machines": [
                                {"machine_id": cm["machine_id"], "process_time": cm["base_processing_time"]}
                                for cm in op["candidate_machines"]
                            ]
                        }
                        for op in job_data["operations"]
                    ],
                    "release_time": job_data.get("release_time", self.state["time"]),
                    "due_time": job_data.get("due_time", 999),
                    "initial_location": job_data.get("initial_location", "L/U"),
                    "current_location": job_data.get("initial_location", "L/U"),
                    "current_op_index": 0,
                    "finished": False,
                    "ready_time": self.state["time"],
                    "locked": False
                }
                self.state["jobs"].append(new_job)

    def _lookahead_eval(self, decision: Dict[str, Any], steps: int = 12) -> float:
        """
        前瞻评估：通过快速启发式规则 (COOP) 预演未来决策序列。
        符合“动态自适应”目标：在动态环境下，通过多步推演选择最优路径。
        """
        try:
            # 1. 性能优化：轻量化克隆状态 (排除 graph 对象以加快 deepcopy)
            temp_state = copy.deepcopy({k: v for k, v in self.state.items() if k != "graph"})
            temp_state["graph"] = self.state["graph"]
            
            # 2. 模拟当前决策的初步影响
            job = next((j for j in temp_state["jobs"] if j["job_id"] == decision["job_id"]), None)
            machine = next((m for m in temp_state["machines"] if m["machine_id"] == decision["machine_id"]), None)
            
            if not job or not machine or job.get("locked"):
                return -1.0
            
            current_op = job["operations"][job["current_op_index"]]
            base_pt = next((cm["process_time"] for cm in current_op["candidate_machines"] if cm["machine_id"] == machine["machine_id"]), 10.0)
            
            # 模拟执行
            machine["status"] = "busy"
            machine["available_time"] = temp_state["time"] + base_pt
            job["locked"] = True
            job["current_location"] = machine["location"]
            
            # 3. 启发式多步推演
            score = 0.0
            for _ in range(steps):
                # 尝试派工
                next_action = PDR.get_dispatch_action(temp_state, rule="COOP")
                if next_action:
                    j2 = next(j for j in temp_state["jobs"] if j["job_id"] == next_action["job_id"])
                    m2 = next(m for m in temp_state["machines"] if m["machine_id"] == next_action["machine_id"])
                    op2 = j2["operations"][j2["current_op_index"]]
                    pt2 = next((cm["process_time"] for cm in op2["candidate_machines"] if cm["machine_id"] == m2["machine_id"]), 10.0)
                    
                    # 动态估算运输开销 (生产-运输协同)
                    v2 = next((v for v in temp_state["vehicles"] if v["vehicle_id"] == next_action.get("vehicle_id")), None)
                    t_time = PDR._transport_time(temp_state, j2["current_location"], m2["location"], v2)
                    
                    finish_t = temp_state["time"] + t_time + pt2
                    m2["status"] = "busy"
                    m2["available_time"] = finish_t
                    j2["locked"] = True
                    j2["current_location"] = m2["location"]
                    
                    # 协同评分：综合考虑加工效率、运输比重与任务交期
                    # 协同评分：(加工时间 / (运输时间 + 1)) * 任务紧急度
                    urgency = 1.0 + max(0, (100 - (j2.get("due_time", 200) - finish_t)) / 100)
                    score += (pt2 / (t_time + 1.0)) * urgency
                else:
                    # 推进虚拟时间
                    busy_times = [m["available_time"] for m in temp_state["machines"] if m["status"] == "busy"]
                    if not busy_times: break
                    temp_state["time"] = min(busy_times)
                    for m in temp_state["machines"]:
                        if m["available_time"] <= temp_state["time"]:
                            m["status"] = "idle"
                    for j in temp_state["jobs"]:
                        if j.get("locked") and not any(m["available_time"] > temp_state["time"] for m in temp_state["machines"]):
                             j["locked"] = False # 简化处理
                             
            return round(score, 3)
        except Exception:
            return 0.0

    def _apply_dispatch(self, decision: Dict[str, Any]):
        """执行派工动作并产生后续事件。"""
        job_id = decision["job_id"]
        machine_id = decision["machine_id"]
        vehicle_id = decision.get("vehicle_id")
        
        job = next(j for j in self.state["jobs"] if j["job_id"] == job_id)
        machine = next(m for m in self.state["machines"] if m["machine_id"] == machine_id)
        if job.get("finished"):
            return False
        if job.get("locked", False):
            return False
        if machine.get("status") != "idle":
            return False
        if job["current_op_index"] >= len(job["operations"]):
            return False
        current_op = job["operations"][job["current_op_index"]]
        if decision.get("op_id") != current_op.get("op_id"):
            return False
        if not any(cm["machine_id"] == machine_id for cm in current_op["candidate_machines"]):
            return False
        
        # 获取不确定性配置
        unc_cfg = self.state.get("metadata", {}).get("uncertainty_config", {})
        trans_cfg = unc_cfg.get("transport", {})
        delay_cfg = unc_cfg.get("vehicle_delay", {})
        proc_cfg = unc_cfg.get("processing", {})

        # 1. 处理运输
        transport_time = 0.0
        if job["current_location"] != machine["location"]:
            if not vehicle_id:
                vehicle_id = self.transport_scheduler.assign_vehicle(
                    job_id, decision["op_id"], job["current_location"], machine["location"]
                )
                if not vehicle_id:
                    self.transport_scheduler.enqueue_transport(decision)
                    return False
                decision["vehicle_id"] = vehicle_id
            vehicle = next(v for v in self.state["vehicles"] if v["vehicle_id"] == vehicle_id)
            if vehicle.get("status") != "idle":
                self.transport_scheduler.enqueue_transport(decision)
                return False
            
            graph = self.state["graph"]
            dist_to_job = nx.shortest_path_length(graph, vehicle["current_location"], job["current_location"], weight="weight")
            dist_to_machine = nx.shortest_path_length(graph, job["current_location"], machine["location"], weight="weight")
            
            # 基础运输时间
            base_transport_time = (dist_to_job + dist_to_machine) / vehicle["speed"]
            
            # 采样不确定运输时间 (TransportTimeUncertainty)
            transport_time = sample_value(
                base_transport_time, 
                trans_cfg.get("fluctuation_low", 0.95), 
                trans_cfg.get("fluctuation_high", 1.15), 
                trans_cfg.get("distribution_type", "uniform"), 
                self.rng
            )
            
            # 采样随机延迟 (VehicleDelayModel)
            if delay_cfg.get("enabled", True):
                if self.rng.random() < delay_cfg.get("delay_probability", 0.05):
                    delay_min, delay_max = delay_cfg.get("delay_range", [2.0, 10.0])
                    delay = self.rng.uniform(delay_min, delay_max)
                    transport_time += delay
            
            # 锁定资源
            vehicle["status"] = "busy"
            vehicle["available_time"] = self.state["time"] + transport_time
            vehicle["current_task"] = f"{job_id}->{machine_id}"
            self.transport_scheduler.reserve_path(job["current_location"], machine["location"], self.state["time"] + transport_time)
            
            self.schedule_event(Event(
                time=self.state["time"] + transport_time,
                event_type=EventType.TRANSPORT_FINISH,
                payload={
                    "job_id": job_id,
                    "vehicle_id": vehicle_id,
                    "source": job["current_location"],
                    "destination": machine["location"],
                    "transport_time": transport_time
                }
            ))
        
        # 2. 处理加工
        start_time = max(self.state["time"] + transport_time, machine["available_time"])
        
        op = current_op
        base_pt = next(cm["process_time"] for cm in op["candidate_machines"] if cm["machine_id"] == machine_id)
        
        # 采样不确定加工时间 (ProcessingTimeUncertainty)
        actual_pt = sample_value(
            base_pt, 
            proc_cfg.get("fluctuation_low", 0.9), 
            proc_cfg.get("fluctuation_high", 1.1), 
            proc_cfg.get("distribution_type", "uniform"), 
            self.rng
        )
        
        finish_time = start_time + actual_pt
        
        # 锁定机器
        machine["status"] = "busy"
        machine["available_time"] = finish_time
        job["locked"] = True
        
        # 产生加工完成事件
        history_entry = {
            "job_id": job_id,
            "op_id": decision["op_id"],
            "machine_id": machine_id,
            "vehicle_id": vehicle_id,
            "start_time": start_time,
            "finish_time": finish_time,
            "transport_time": transport_time,
            "process_time": actual_pt,
            "location": machine["location"],
            "lookahead_score": decision.get("lookahead_score")
        }
        
        self.schedule_event(Event(
            time=finish_time,
            event_type=EventType.PROCESS_FINISH,
            payload={
                "job_id": job_id,
                "machine_id": machine_id,
                "location": machine["location"],
                "history_entry": history_entry
            }
        ))
        return True
