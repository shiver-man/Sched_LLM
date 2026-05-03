"""
基于遗传算法 (GA) 的元启发式调度模块
"""
import random
import copy
from typing import Dict, Any, List, Tuple
from app.core.engine import EventEngine
from app.core.evaluator import Evaluator

class GeneticAlgorithm:
    def __init__(
        self,
        state: Dict[str, Any],
        pop_size: int = 50,
        generations: int = 20,
        debug: bool = False,
        use_tabu_seed: bool = False,
        tabu_iters: int = 30,
        tabu_tenure: int = 12,
    ):
        # 深度复制状态，防止外部修改
        self.base_state = copy.deepcopy(state)
        self.pop_size = pop_size
        self.generations = generations
        self.jobs = self.base_state["jobs"]
        self.machines = self.base_state["machines"]
        self.vehicles = self.base_state["vehicles"]
        self.debug = debug
        self.use_tabu_seed = use_tabu_seed
        self.tabu_iters = tabu_iters
        self.tabu_tenure = tabu_tenure
        
        # 统计总工序数和候选机器范围
        self.op_list = []
        self.ms_ranges = []
        for job in self.jobs:
            for op in job["operations"]:
                self.op_list.append((job["job_id"], op["op_id"]))
                self.ms_ranges.append(len(op["candidate_machines"]))
        self.num_total_ops = len(self.op_list)
        self.eval_max_steps = max(1000, self.num_total_ops * 80)
        self.job_key = {self._norm(j["job_id"]): idx for idx, j in enumerate(self.jobs)}
        self.generation_history: List[Dict[str, Any]] = []
        self.num_vehicles = max(1, len(self.vehicles))
        
    def _create_chromosome(self) -> Tuple[List[int], List[int], List[int]]:
        """
        创建染色体：OS (Operation Sequence) 和 MS (Machine Selection)
        """
        # OS: 用 job_id 的索引填充，重复次数等于其工序数，然后随机打乱
        os = []
        for j_idx, job in enumerate(self.jobs):
            os.extend([j_idx] * len(job["operations"]))
        random.shuffle(os)
        
        # MS: 每一位代表对应工序选择候选机器列表中的第几个机器
        ms = [random.randint(0, r - 1) for r in self.ms_ranges]
        
        # VS: 每一位代表运输优先车辆索引（仅在需要运输时生效）
        vs = [random.randint(0, self.num_vehicles - 1) for _ in range(self.num_total_ops)]
        return os, ms, vs

    def _create_seed_chromosome(self) -> Tuple[List[int], List[int], List[int]]:
        ordered = sorted(
            list(enumerate(self.jobs)),
            key=lambda x: (x[1].get("release_time", 0.0), x[0])
        )
        os = []
        for j_idx, job in ordered:
            os.extend([j_idx] * len(job["operations"]))

        ms = []
        for job in self.jobs:
            for op in job["operations"]:
                best_idx = 0
                best_pt = op["candidate_machines"][0]["process_time"]
                for idx, cm in enumerate(op["candidate_machines"]):
                    if cm["process_time"] < best_pt:
                        best_idx = idx
                        best_pt = cm["process_time"]
                ms.append(best_idx)
        vs = [i % self.num_vehicles for i in range(self.num_total_ops)]
        return os, ms, vs

    @staticmethod
    def _norm(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().casefold()

    def _decode_ms_map(self, ms: List[int]) -> Dict[Tuple[str, str], str]:
        ms_map = {}
        ms_idx = 0
        for job in self.jobs:
            for op in job["operations"]:
                candidate = op["candidate_machines"]
                choose_idx = ms[ms_idx] if ms_idx < len(ms) else 0
                if choose_idx < 0 or choose_idx >= len(candidate):
                    choose_idx = 0
                ms_map[(self._norm(job["job_id"]), self._norm(op["op_id"]))] = candidate[choose_idx]["machine_id"]
                ms_idx += 1
        return ms_map

    def _decode_vs_map(self, vs: List[int]) -> Dict[Tuple[str, str], int]:
        vs_map = {}
        idx = 0
        for job in self.jobs:
            for op in job["operations"]:
                choose_idx = vs[idx] if idx < len(vs) else 0
                if choose_idx < 0 or choose_idx >= self.num_vehicles:
                    choose_idx = 0
                vs_map[(self._norm(job["job_id"]), self._norm(op["op_id"]))] = choose_idx
                idx += 1
        return vs_map

    def _tabu_optimize_seed(self, os: List[int], ms: List[int]) -> Tuple[List[int], List[int]]:
        """
        TS 预优化：仅优化生产编码 OS+MS，产出高质量 seed。
        """
        current_os = copy.deepcopy(os)
        current_ms = copy.deepcopy(ms)
        current_vs = [0 for _ in range(self.num_total_ops)]
        current_fit = self._evaluate((current_os, current_ms, current_vs))
        best_os, best_ms, best_fit = copy.deepcopy(current_os), copy.deepcopy(current_ms), current_fit

        tabu_list: Dict[str, int] = {}
        rng = random.Random(42)

        for it in range(self.tabu_iters):
            candidates: List[Tuple[float, Tuple[str, int, int], List[int], List[int]]] = []

            # OS 邻域：随机交换两个位置
            for _ in range(6):
                if len(current_os) < 2:
                    break
                i, j = rng.sample(range(len(current_os)), 2)
                n_os = copy.deepcopy(current_os)
                n_os[i], n_os[j] = n_os[j], n_os[i]
                move = ("os_swap", min(i, j), max(i, j))
                move_key = f"{move[0]}:{move[1]}:{move[2]}"
                fit = self._evaluate((n_os, current_ms, current_vs))
                if move_key not in tabu_list or fit > best_fit:
                    candidates.append((fit, move, n_os, current_ms))

            # MS 邻域：随机改一个工序的机器选择
            for _ in range(6):
                if not current_ms:
                    break
                idx = rng.randint(0, len(current_ms) - 1)
                range_max = self.ms_ranges[idx]
                if range_max <= 1:
                    continue
                new_val = rng.randint(0, range_max - 1)
                if new_val == current_ms[idx]:
                    continue
                n_ms = copy.deepcopy(current_ms)
                n_ms[idx] = new_val
                move = ("ms_change", idx, new_val)
                move_key = f"{move[0]}:{move[1]}:{move[2]}"
                fit = self._evaluate((current_os, n_ms, current_vs))
                if move_key not in tabu_list or fit > best_fit:
                    candidates.append((fit, move, current_os, n_ms))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[0], reverse=True)
            best_neighbor_fit, best_move, best_neighbor_os, best_neighbor_ms = candidates[0]
            current_os = copy.deepcopy(best_neighbor_os)
            current_ms = copy.deepcopy(best_neighbor_ms)
            current_fit = best_neighbor_fit

            if current_fit > best_fit:
                best_fit = current_fit
                best_os = copy.deepcopy(current_os)
                best_ms = copy.deepcopy(current_ms)

            # 维护 tabu tenure
            for key in list(tabu_list.keys()):
                tabu_list[key] -= 1
                if tabu_list[key] <= 0:
                    del tabu_list[key]
            move_key = f"{best_move[0]}:{best_move[1]}:{best_move[2]}"
            tabu_list[move_key] = self.tabu_tenure

        return best_os, best_ms

    def _evaluate(self, chromosome: Tuple[List[int], List[int], List[int]]) -> float:
        """
        适应度评估：运行离散事件引擎
        """
        os, ms, vs = chromosome
        
        # 构建一个基于染色体的策略函数
        # GA 的 OS/MS 指导调度逻辑
        # 预先映射 MS 到每个工序的机器
        ms_map = self._decode_ms_map(ms)
        vs_map = self._decode_vs_map(vs)

        def ga_policy(state):
            # 获取当前可派工任务
            from app.models.state import get_dispatchable_jobs
            dispatchable = get_dispatchable_jobs(state)
            if not dispatchable:
                return None
            
            # 找到 OS 中当前最优先的、且在可派工列表中的任务
            dispatchable_ids = {self._norm(d["job_id"]) for d in dispatchable}
            for j_idx in os:
                if j_idx >= len(self.jobs):
                    continue
                target_job_id = self.jobs[j_idx]["job_id"]
                target_job_key = self._norm(target_job_id)
                # 检查这个 job 的当前待办工序是否在可派工列表中
                job_in_state = next((j for j in state["jobs"] if self._norm(j["job_id"]) == target_job_key), None)
                if not job_in_state or job_in_state["finished"] or job_in_state.get("locked"):
                    continue
                
                # 检查是否满足工序顺序
                if target_job_key in dispatchable_ids:
                    # 确定机器
                    current_op = job_in_state["operations"][job_in_state["current_op_index"]]
                    preferred_m = ms_map.get((target_job_key, self._norm(current_op["op_id"])), current_op["candidate_machines"][0]["machine_id"])
                    candidate_order = [preferred_m] + [
                        cm["machine_id"] for cm in current_op["candidate_machines"] if cm["machine_id"] != preferred_m
                    ]
                    for m_id in candidate_order:
                        machine = next((m for m in state["machines"] if m["machine_id"] == m_id), None)
                        if not machine or machine["status"] != "idle":
                            continue
                        vehicle_id = None
                        if job_in_state["current_location"] != machine["location"]:
                            idle_vehicles = [v for v in state["vehicles"] if v["status"] == "idle"]
                            if not idle_vehicles:
                                continue
                            pref_v_idx = vs_map.get((target_job_key, self._norm(current_op["op_id"])), 0)
                            pref_vehicle_id = self.vehicles[pref_v_idx % self.num_vehicles]["vehicle_id"]
                            pref_vehicle = next((v for v in idle_vehicles if v["vehicle_id"] == pref_vehicle_id), None)
                            if pref_vehicle is not None:
                                vehicle_id = pref_vehicle["vehicle_id"]
                            else:
                                from app.core.scheduler import PDR
                                vehicle_id = min(
                                    idle_vehicles,
                                    key=lambda v: PDR._transport_time(
                                        state, job_in_state["current_location"], machine["location"], v
                                    ),
                                )["vehicle_id"]
                        return {
                            "job_id": target_job_id,
                            "op_id": current_op["op_id"],
                            "machine_id": m_id,
                            "vehicle_id": vehicle_id,
                            "reason": "GA meta-heuristic decision"
                        }
            from app.core.scheduler import PDR
            return PDR.get_dispatch_action(state, rule="COOP_RH")

        # 运行引擎 (不含随机性，用于评估确定性质量)
        temp_state = copy.deepcopy(self.base_state)
        engine = EventEngine(temp_state, policy_fn=ga_policy)
        engine.rng = random.Random(42) # 固定种子评估
        final_state = engine.run(max_steps=self.eval_max_steps)
        
        metrics = Evaluator.evaluate(final_state)
        total_ops = max(1, metrics.get("total_ops_expected", self.num_total_ops))
        progress = metrics.get("num_events", 0) / total_ops
        if metrics["makespan"] == 0:
            return -1e6 + (progress * 1e4)
        if not metrics.get("is_complete", False):
            missing_ops = max(0, metrics.get("total_ops_expected", 0) - metrics.get("num_events", 0))
            return -1e5 - (missing_ops * 1e3) - metrics["makespan"] + (progress * 5e3)
        return -metrics["makespan"] + (metrics.get("utilization", 0) * 20.0)

    def solve(self) -> Dict[str, Any]:
        """
        运行遗传算法
        """
        # 初始化种群
        population = [self._create_chromosome() for _ in range(max(0, self.pop_size - 2))]
        seed_os, seed_ms, seed_vs = self._create_seed_chromosome()
        if self.use_tabu_seed:
            seed_os, seed_ms = self._tabu_optimize_seed(seed_os, seed_ms)
        seed = (seed_os, seed_ms, seed_vs)
        population.append(seed)
        population.append(self._mutate(copy.deepcopy(seed)))
        
        best_chrom = None
        best_fitness = -1e12
        
        for gen in range(self.generations):
            # 评估
            fitness_scores = [self._evaluate(chrom) for chrom in population]
            if self.debug and fitness_scores:
                best_idx_gen = max(range(len(population)), key=lambda i: fitness_scores[i])
                self.generation_history.append(
                    {
                        "generation": gen + 1,
                        "best_fitness": round(float(max(fitness_scores)), 6),
                        "avg_fitness": round(float(sum(fitness_scores) / len(fitness_scores)), 6),
                        "worst_fitness": round(float(min(fitness_scores)), 6),
                        "best_chromosome_os_head": population[best_idx_gen][0][: min(20, len(population[best_idx_gen][0]))],
                        "best_chromosome_ms_head": population[best_idx_gen][1][: min(20, len(population[best_idx_gen][1]))],
                    }
                )
            
            # 记录最优
            for i, score in enumerate(fitness_scores):
                if score > best_fitness:
                    best_fitness = score
                    best_chrom = copy.deepcopy(population[i])
            
            ranked_idx = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)
            elites = [copy.deepcopy(population[ranked_idx[0]])]
            if len(ranked_idx) > 1:
                elites.append(copy.deepcopy(population[ranked_idx[1]]))

            new_pop = elites
            while len(new_pop) < self.pop_size:
                # 选两个亲本
                p1 = self._tournament_select(population, fitness_scores)
                p2 = self._tournament_select(population, fitness_scores)
                
                # 交叉
                c1, c2 = self._crossover(p1, p2)
                
                # 变异
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)
            population = new_pop[:self.pop_size]
            
        # 最终结果
        if best_chrom:
            # 重新运行一遍获取完整计划
            os, ms, vs = best_chrom
            ms_map = self._decode_ms_map(ms)
            vs_map = self._decode_vs_map(vs)

            def final_ga_policy(state):
                from app.models.state import get_dispatchable_jobs
                dispatchable = get_dispatchable_jobs(state)
                if not dispatchable: return None
                dispatchable_ids = {self._norm(d["job_id"]) for d in dispatchable}
                for j_idx in os:
                    if j_idx >= len(self.jobs):
                        continue
                    target_job_id = self.jobs[j_idx]["job_id"]
                    target_job_key = self._norm(target_job_id)
                    job_in_state = next((j for j in state["jobs"] if self._norm(j["job_id"]) == target_job_key), None)
                    if not job_in_state or job_in_state["finished"] or job_in_state.get("locked"): continue
                    if target_job_key in dispatchable_ids:
                        current_op = job_in_state["operations"][job_in_state["current_op_index"]]
                        preferred_m = ms_map.get((target_job_key, self._norm(current_op["op_id"])), current_op["candidate_machines"][0]["machine_id"])
                        candidate_order = [preferred_m] + [
                            cm["machine_id"] for cm in current_op["candidate_machines"] if cm["machine_id"] != preferred_m
                        ]
                        for m_id in candidate_order:
                            machine = next((m for m in state["machines"] if m["machine_id"] == m_id), None)
                            if not machine or machine["status"] != "idle":
                                continue
                            vehicle_id = None
                            if job_in_state["current_location"] != machine["location"]:
                                idle_vehicles = [v for v in state["vehicles"] if v["status"] == "idle"]
                                if not idle_vehicles:
                                    continue
                                pref_v_idx = vs_map.get((target_job_key, self._norm(current_op["op_id"])), 0)
                                pref_vehicle_id = self.vehicles[pref_v_idx % self.num_vehicles]["vehicle_id"]
                                pref_vehicle = next((v for v in idle_vehicles if v["vehicle_id"] == pref_vehicle_id), None)
                                if pref_vehicle is not None:
                                    vehicle_id = pref_vehicle["vehicle_id"]
                                else:
                                    from app.core.scheduler import PDR
                                    vehicle_id = min(
                                        idle_vehicles,
                                        key=lambda v: PDR._transport_time(
                                            state, job_in_state["current_location"], machine["location"], v
                                        ),
                                    )["vehicle_id"]
                            return {
                                "job_id": target_job_id,
                                "op_id": current_op["op_id"],
                                "machine_id": m_id,
                                "vehicle_id": vehicle_id,
                                "reason": "GA Optimized"
                            }
                from app.core.scheduler import PDR
                return PDR.get_dispatch_action(state, rule="COOP_RH")

            temp_state = copy.deepcopy(self.base_state)
            engine = EventEngine(temp_state, policy_fn=final_ga_policy)
            engine.rng = random.Random(42)
            final_state = engine.run(max_steps=self.eval_max_steps)
            
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
                    "transport_time": round(h.get("transport_time", 0), 2)
                }
                for idx, h in enumerate(final_state["history"])
            ]
            if not plan or metrics.get("makespan", 0) <= 0:
                return {"status": "failed", "rule": "Meta-Heuristic (Genetic Algorithm)", "detail": "empty_or_invalid_plan"}
            diagnostics = None
            if self.debug:
                dispatch_trace = []
                for e in final_state.get("event_trace", []):
                    if e.get("event_type") == "dispatch_decision":
                        p = e.get("payload", {})
                        dispatch_trace.append(
                            {
                                "time": round(float(e.get("time", 0.0)), 4),
                                "job_id": p.get("job_id"),
                                "op_id": p.get("op_id"),
                                "machine_id": p.get("machine_id"),
                                "vehicle_id": p.get("vehicle_id"),
                                "reason": p.get("reason"),
                                "lookahead_score": p.get("lookahead_score"),
                            }
                        )
                best_ms_map = {}
                for job in self.jobs:
                    for op in job["operations"]:
                        key = f"{job['job_id']}::{op['op_id']}"
                        best_ms_map[key] = ms_map.get((self._norm(job["job_id"]), self._norm(op["op_id"])))
                diagnostics = {
                    "best_fitness": round(float(best_fitness), 6),
                    "best_chromosome_os": os,
                    "best_chromosome_ms": ms,
                    "best_chromosome_vs": vs,
                    "best_machine_mapping": best_ms_map,
                    "generation_history": self.generation_history,
                    "dispatch_trace": dispatch_trace,
                    "event_trace_size": len(final_state.get("event_trace", [])),
                }
            return {
                "status": "success",
                "rule": "Meta-Heuristic (Genetic Algorithm)",
                "metrics": metrics,
                "plan": plan,
                "ga_diagnostics": diagnostics
            }
            
        return {"status": "failed", "rule": "GA"}

    def _tournament_select(self, pop, scores, k=3):
        k = min(max(1, k), len(pop))
        idx = random.sample(range(len(pop)), k)
        best_idx = idx[0]
        for i in idx:
            if scores[i] > scores[best_idx]:
                best_idx = i
        return pop[best_idx]

    def _crossover(self, p1, p2):
        os1, ms1, vs1 = p1
        os2, ms2, vs2 = p2
        
        # OS Crossover: POX (Precedence Operation Crossover)
        # 简化：随机切片交换并修复重复
        cut = random.randint(1, len(os1) - 1)
        # 这里使用一种简单且合法的 OS 交叉：保留一部分，另一部分按顺序填充
        def pox(parent1, parent2):
            child = [-1] * len(parent1)
            # 随机选一些 job_id 保留位置
            jobs_to_keep = random.sample(range(len(self.jobs)), len(self.jobs) // 2)
            for i, v in enumerate(parent1):
                if v in jobs_to_keep:
                    child[i] = v
            # 剩余位置按 parent2 的顺序填充
            ptr = 0
            for v in parent2:
                if v not in jobs_to_keep:
                    while child[ptr] != -1: ptr += 1
                    child[ptr] = v
            return child
        
        c_os1 = pox(os1, os2)
        c_os2 = pox(os2, os1)
        
        # MS Crossover: 单点交叉
        cut_ms = random.randint(1, len(ms1) - 1)
        c_ms1 = ms1[:cut_ms] + ms2[cut_ms:]
        c_ms2 = ms2[:cut_ms] + ms1[cut_ms:]
        
        # VS Crossover: 单点交叉
        cut_vs = random.randint(1, len(vs1) - 1) if len(vs1) > 1 else 1
        c_vs1 = vs1[:cut_vs] + vs2[cut_vs:]
        c_vs2 = vs2[:cut_vs] + vs1[cut_vs:]

        return (c_os1, c_ms1, c_vs1), (c_os2, c_ms2, c_vs2)

    def _mutate(self, chrom):
        os, ms, vs = chrom
        # OS Mutation: 交换两个位置
        if random.random() < 0.1 and len(os) > 1:
            idx1, idx2 = random.sample(range(len(os)), 2)
            os[idx1], os[idx2] = os[idx2], os[idx1]
        
        # MS Mutation: 随机改变一个工序的机器
        if random.random() < 0.1 and len(ms) > 0:
            idx = random.randint(0, len(ms) - 1)
            # 找到该工序对应的候选机器数量
            range_max = self.ms_ranges[idx]
            ms[idx] = random.randint(0, range_max - 1)
        
        # VS Mutation: 随机改变一个工序偏好车辆
        if random.random() < 0.1 and len(vs) > 0:
            idx = random.randint(0, len(vs) - 1)
            vs[idx] = random.randint(0, self.num_vehicles - 1)

        return os, ms, vs
