"""
基于 OR-Tools CP-SAT 的数学优化调度模块
实现了完整的生产-运输协同调度建模 (FJSPT)，包含：
1. 工序先后顺序约束
2. 机器资源互斥约束
3. 运输时间约束 (取决于机器位置)
4. 运输资源约束 (AGV 数量限制)
"""
from typing import Dict, Any, List
import copy
import networkx as nx
from ortools.sat.python import cp_model
from app.core.evaluator import Evaluator

class MathOptimizer:
    @staticmethod
    def solve_fjspt(state: Dict[str, Any], time_limit_seconds: float = 30.0) -> Dict[str, Any]:
        """
        使用 CP-SAT 模型求解柔性作业车间生产-运输一体化调度问题 (FJSPT)。
        """
        model = cp_model.CpModel()
        jobs = state["jobs"]
        machines = state["machines"]
        vehicles = state["vehicles"]
        graph = state["graph"]
        
        # 1. 预处理
        num_jobs = len(jobs)
        num_machines = len(machines)
        num_vehicles = len(vehicles) if vehicles else 1
        
        # 计算 Horizon (必须是整数)
        horizon_val = 0
        max_transport = 0
        if graph.nodes:
            try:
                # 寻找图中最长最短路径作为单次运输的最大可能时间
                path_lengths = dict(nx.all_pairs_shortest_path_length(graph, weight="weight"))
                for src in path_lengths:
                    for dst in path_lengths[src]:
                        max_transport = max(max_transport, path_lengths[src][dst])
            except:
                max_transport = 100
        
        for j in jobs:
            job_time = 0
            for op in j["operations"]:
                job_time += max(cm["process_time"] for cm in op["candidate_machines"])
                job_time += max_transport # 每道工序预留一次最大可能的运输时间
            horizon_val += job_time
        
        # 加上一个安全余量
        horizon = int(horizon_val * 1.2) + 100

        # 预先计算机器间的运输距离矩阵
        def get_dist(loc1, loc2):
            try:
                return int(nx.shortest_path_length(graph, loc1, loc2, weight="weight"))
            except:
                return 50 # 默认降级

        # 2. 变量定义
        # p_starts[j][o], p_ends[j][o] -> 加工开始/结束
        # t_starts[j][o], t_ends[j][o] -> 运输开始/结束 (指运送工件到该工序所在机器的运输)
        p_starts = {}
        p_ends = {}
        t_starts = {}
        t_ends = {}
        p_intervals = {}
        t_intervals = {}
        m_choices = {} # (j, o, m) -> bool
        v_choices = {} # (j, o, v) -> bool
        
        for j_idx, job in enumerate(jobs):
            for o_idx, op in enumerate(job["operations"]):
                # 加工变量
                p_start = model.NewIntVar(0, horizon, f'p_start_{j_idx}_{o_idx}')
                p_end = model.NewIntVar(0, horizon, f'p_end_{j_idx}_{o_idx}')
                p_starts[(j_idx, o_idx)] = p_start
                p_ends[(j_idx, o_idx)] = p_end
                
                # 运输变量 (工序开始前的搬运)
                t_start = model.NewIntVar(0, horizon, f't_start_{j_idx}_{o_idx}')
                t_end = model.NewIntVar(0, horizon, f't_end_{j_idx}_{o_idx}')
                t_starts[(j_idx, o_idx)] = t_start
                t_ends[(j_idx, o_idx)] = t_end
                
                # 机器分配
                machine_literals = []
                for cm in op["candidate_machines"]:
                    m_idx = next(i for i, m in enumerate(machines) if m["machine_id"] == cm["machine_id"])
                    b = model.NewBoolVar(f'm_{j_idx}_{o_idx}_{m_idx}')
                    machine_literals.append((m_idx, b, int(cm["process_time"])))
                    m_choices[(j_idx, o_idx, m_idx)] = b
                
                model.AddExactlyOne([b for _, b, _ in machine_literals])
                
                # 加工时长约束
                p_dur = model.NewIntVar(0, horizon, f'p_dur_{j_idx}_{o_idx}')
                for _, b, pt in machine_literals:
                    model.Add(p_dur == pt).OnlyEnforceIf(b)
                
                p_intervals[(j_idx, o_idx)] = model.NewIntervalVar(p_start, p_dur, p_end, f'p_iv_{j_idx}_{o_idx}')

                # 运输资源分配 (选择哪个 AGV)
                v_literals = []
                for v_idx in range(num_vehicles):
                    b = model.NewBoolVar(f'v_{j_idx}_{o_idx}_{v_idx}')
                    v_literals.append(b)
                    v_choices[(j_idx, o_idx, v_idx)] = b
                model.AddExactlyOne(v_literals)

        # 3. 约束条件
        # (1) 先后顺序：前序加工结束 -> 本序运输开始 -> 本序运输结束 -> 本序加工开始
        for j_idx, job in enumerate(jobs):
            for o_idx in range(len(job["operations"])):
                # 约束：运输结束才能加工
                model.Add(p_starts[(j_idx, o_idx)] >= t_ends[(j_idx, o_idx)])
                
                if o_idx == 0:
                    # 第一道工序从初始位置开始运输
                    init_loc = job.get("initial_location", "LOAD_STATION")
                    model.Add(t_starts[(j_idx, o_idx)] >= int(job.get("release_time", 0)))
                    for m_idx, b, _ in machine_choices_of_op(j_idx, o_idx, m_choices, num_machines):
                        dist = get_dist(init_loc, machines[m_idx]["location"])
                        model.Add(t_ends[(j_idx, o_idx)] >= t_starts[(j_idx, o_idx)] + dist).OnlyEnforceIf(b)
                else:
                    # 后续工序：前序结束 -> 运输开始
                    model.Add(t_starts[(j_idx, o_idx)] >= p_ends[(j_idx, o_idx - 1)])
                    # 运输耗时取决于前序机器和本序机器
                    for m_prev_idx in range(num_machines):
                        b_prev = m_choices.get((j_idx, o_idx - 1, m_prev_idx))
                        if b_prev is None: continue
                        for m_curr_idx in range(num_machines):
                            b_curr = m_choices.get((j_idx, o_idx, m_curr_idx))
                            if b_curr is None: continue
                            
                            dist = get_dist(machines[m_prev_idx]["location"], machines[m_curr_idx]["location"])
                            # 当两序机器同时确定时，约束运输时长
                            both = model.NewBoolVar(f'both_{j_idx}_{o_idx}_{m_prev_idx}_{m_curr_idx}')
                            model.AddBoolAnd([b_prev, b_curr]).OnlyEnforceIf(both)
                            model.AddBoolOr([b_prev.Not(), b_curr.Not()]).OnlyEnforceIf(both.Not())
                            model.Add(t_ends[(j_idx, o_idx)] >= t_starts[(j_idx, o_idx)] + dist).OnlyEnforceIf(both)

        # (2) 机器互斥：同一台机器在同一时间只能加工一个工件
        for m_idx in range(num_machines):
            m_ivs = []
            for j_idx, job in enumerate(jobs):
                for o_idx in range(len(job["operations"])):
                    b = m_choices.get((j_idx, o_idx, m_idx))
                    if b is not None:
                        # 创建可选区间
                        start = p_starts[(j_idx, o_idx)]
                        end = p_ends[(j_idx, o_idx)]
                        # 需要一个确定的持续时间变量
                        dur = model.NewIntVar(0, horizon, f'd_{j_idx}_{o_idx}_{m_idx}')
                        pt = next(cm["process_time"] for cm in job["operations"][o_idx]["candidate_machines"] if cm["machine_id"] == machines[m_idx]["machine_id"])
                        model.Add(dur == int(pt)).OnlyEnforceIf(b)
                        model.Add(dur == 0).OnlyEnforceIf(b.Not())
                        
                        m_ivs.append(model.NewOptionalIntervalVar(start, dur, end, b, f'opt_m_{j_idx}_{o_idx}_{m_idx}'))
            model.AddNoOverlap(m_ivs)

        # (3) 运输资源约束 (AGV 数量限制)
        # 简化建模：所有运输任务作为一个整体，限制并行数量 <= AGV 数
        all_t_ivs = []
        for (j_idx, o_idx), t_start in t_starts.items():
            t_end = t_ends[(j_idx, o_idx)]
            t_dur = model.NewIntVar(0, horizon, f't_dur_{j_idx}_{o_idx}')
            model.Add(t_dur == t_end - t_start)
            iv = model.NewIntervalVar(t_start, t_dur, t_end, f't_iv_{j_idx}_{o_idx}')
            all_t_ivs.append(iv)
        
        # 使用 Cumulative 约束限制 AGV 占用
        model.AddCumulative(all_t_ivs, [1]*len(all_t_ivs), num_vehicles)

        # 4. 目标函数：最小化 Makespan
        makespan = model.NewIntVar(0, horizon, 'makespan')
        last_ends = []
        for j_idx, job in enumerate(jobs):
            last_ends.append(p_ends[(j_idx, len(job["operations"]) - 1)])
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        # 5. 求解
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            history = []
            for j_idx, job in enumerate(jobs):
                for o_idx, op in enumerate(job["operations"]):
                    start_val = solver.Value(p_starts[(j_idx, o_idx)])
                    end_val = solver.Value(p_ends[(j_idx, o_idx)])
                    t_start_val = solver.Value(t_starts[(j_idx, o_idx)])
                    t_end_val = solver.Value(t_ends[(j_idx, o_idx)])
                    
                    selected_m = ""
                    for m_idx in range(num_machines):
                        if (j_idx, o_idx, m_idx) in m_choices and solver.Value(m_choices[(j_idx, o_idx, m_idx)]):
                            selected_m = machines[m_idx]["machine_id"]
                            break
                    
                    selected_v = ""
                    for v_idx in range(num_vehicles):
                        if solver.Value(v_choices[(j_idx, o_idx, v_idx)]):
                            selected_v = vehicles[v_idx]["vehicle_id"] if vehicles else "V1"
                            break

                    history.append({
                        "job_id": job["job_id"],
                        "op_id": op["op_id"],
                        "machine_id": selected_m,
                        "vehicle_id": selected_v,
                        "start_time": float(start_val),
                        "finish_time": float(end_val),
                        "transport_time": float(t_end_val - t_start_val),
                        "transport_start": float(t_start_val)
                    })
            
            history.sort(key=lambda x: x["finish_time"])
            for idx, h in enumerate(history):
                h["step"] = idx + 1
            
            res_state = copy.deepcopy(state)
            res_state["history"] = history
            metrics = Evaluator.evaluate(res_state)
            
            return {
                "status": "success",
                "rule": "CP-SAT (Production-Transport Coordinated)",
                "metrics": metrics,
                "plan": history
            }
        
        return {"status": "failed", "detail": solver.StatusName(status)}

def machine_choices_of_op(j_idx, o_idx, m_choices, num_machines):
    res = []
    for m_idx in range(num_machines):
        b = m_choices.get((j_idx, o_idx, m_idx))
        if b is not None:
            res.append((m_idx, b, 0)) # pt 不在这里用
    return res
