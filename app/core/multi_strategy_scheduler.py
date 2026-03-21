"""
多策略调度实验平台核心引擎 (Multi-Strategy Scheduling Experiment Platform)
支持同一基础数据下多种策略的并行生成、统一建模与横向对比。
"""
import copy
from typing import Dict, Any, List, Type, Callable

from app.models.schema import (
    ScheduleRequest, 
    ScheduleScheme, 
    ScheduleStep, 
    ScheduleMetrics,
    MultiStrategyResponse,
    PPOPlanRequest
)
from app.models.state import build_initial_state
from app.core.scheduler import PDR
from app.core.simulator import Simulator
from app.core.evaluator import Evaluator
from app.core.ppo_scheduler import run_ppo_policy
from app.core.math_optimizer import MathOptimizer
from app.core.meta_heuristic import GeneticAlgorithm
from app.llm.prompt_builder import build_llm_plan_payload, build_llm_plan_brief

class MultiStrategyScheduler:
    def __init__(self, request: ScheduleRequest, max_steps: int = 1000):
        self.request = request
        # 预先构建初始状态，确保所有策略起点一致
        self.initial_state = build_initial_state(request)
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.max_steps = max_steps
        self._register_default_strategies()

    def _register_default_strategies(self):
        """注册系统内置的所有调度策略"""
        # 1. 规则调度 (Rule-based)
        self.register_strategy("SPT", "Rule-based", self._run_pdr_strategy)
        self.register_strategy("FIFO", "Rule-based", self._run_pdr_strategy)
        self.register_strategy("MWKR", "Rule-based", self._run_pdr_strategy)
        
        # 2. 启发式调度 (Heuristic)
        self.register_strategy("COOP", "Heuristic", self._run_pdr_strategy)
        self.register_strategy("COOP_RH", "Heuristic", self._run_pdr_strategy)
        
        # 3. 元启发式调度 (Meta-heuristic)
        self.register_strategy("GA", "Meta-heuristic", self._run_ga_strategy)
        
        # 4. 数学优化调度 (Math-Optimization)
        self.register_strategy("CP-SAT", "Math-Optimization", self._run_math_strategy)
        
        # 5. 强化学习调度 (Reinforcement-Learning)
        self.register_strategy("PPO", "Reinforcement-Learning", self._run_ppo_strategy)

    def register_strategy(self, name: str, category: str, func: Callable):
        """允许动态注册新策略"""
        self.strategies[name] = {
            "category": category,
            "func": func
        }

    def execute_all(self, ppo_policy_id: str = "latest") -> MultiStrategyResponse:
        """执行所有已注册的策略并生成统一结果"""
        detailed_schemes: List[ScheduleScheme] = []
        
        for name, info in self.strategies.items():
            try:
                # 传入策略名称以便区分 PDR 规则
                scheme = info["func"](name, info["category"], ppo_policy_id)
                if scheme:
                    detailed_schemes.append(scheme)
            except Exception as e:
                print(f"策略 {name} 执行失败: {str(e)}")
                continue

        # 生成汇总对比表
        summary = self._generate_summary(detailed_schemes)
        
        # 生成 LLM 简报
        brief = self._generate_llm_brief(detailed_schemes)
        
        return MultiStrategyResponse(
            status="success",
            detailed_schemes=detailed_schemes,
            summary_comparison=summary,
            llm_readable_brief=brief
        )

    def _run_pdr_strategy(self, name: str, category: str, ppo_policy_id: str) -> ScheduleScheme:
        """运行基于 PDR 规则的策略"""
        state_copy = copy.deepcopy(self.initial_state)
        def policy(s):
            return PDR.get_dispatch_action(s, rule=name)
        
        final_state = Simulator.run_simulation(state_copy, policy, max_steps=self.max_steps)
        return self._convert_to_scheme(category, name, final_state)

    def _run_ga_strategy(self, name: str, category: str, ppo_policy_id: str) -> ScheduleScheme:
        """运行遗传算法策略"""
        # GA 内部已经处理了仿真引擎
        ga = GeneticAlgorithm(self.initial_state, pop_size=50, generations=20)
        result = ga.solve()
        if result["status"] == "success":
            metrics = result.get("metrics", {})
            plan = result.get("plan", [])
            if not plan or metrics.get("makespan", 0) <= 0:
                return None
            return ScheduleScheme(
                category=category,
                rule=name,
                metrics=ScheduleMetrics(**metrics),
                plan=[ScheduleStep(**step) for step in plan]
            )
        return None

    def _run_math_strategy(self, name: str, category: str, ppo_policy_id: str) -> ScheduleScheme:
        """运行数学优化策略"""
        result = MathOptimizer.solve_fjspt(self.initial_state, time_limit_seconds=30.0)
        if result["status"] == "success":
            return ScheduleScheme(
                category=category,
                rule=name,
                metrics=ScheduleMetrics(**result["metrics"]),
                plan=[ScheduleStep(**step) for step in result["plan"]]
            )
        return None

    def _run_ppo_strategy(self, name: str, category: str, ppo_policy_id: str) -> ScheduleScheme:
        """运行 PPO 策略"""
        ppo_req = PPOPlanRequest(
            **self.request.model_dump(),
            policy_id=ppo_policy_id
        )
        try:
            result = run_ppo_policy(ppo_req)
            return ScheduleScheme(
                category=category,
                rule=name,
                metrics=ScheduleMetrics(**result["metrics"]),
                plan=[ScheduleStep(**step) for step in result["plan"]]
            )
        except Exception:
            return None

    def _convert_to_scheme(self, category: str, rule: str, state: Dict[str, Any]) -> ScheduleScheme:
        """将仿真状态转换为标准输出结构"""
        metrics_dict = Evaluator.evaluate(state)
        plan = []
        for idx, h in enumerate(state["history"]):
            plan.append(ScheduleStep(
                step=idx + 1,
                job_id=h["job_id"],
                op_id=h["op_id"],
                machine_id=h["machine_id"],
                vehicle_id=h.get("vehicle_id"),
                transport_time=h.get("transport_time", 0.0),
                start_time=h["start_time"],
                finish_time=h["finish_time"],
                lookahead_score=h.get("lookahead_score")
            ))
            
        return ScheduleScheme(
            category=category,
            rule=rule,
            metrics=ScheduleMetrics(**metrics_dict),
            plan=plan
        )

    def _generate_summary(self, schemes: List[ScheduleScheme]) -> List[Dict[str, Any]]:
        """生成指标对比表，优先筛选完整方案"""
        summary = []
        for s in schemes:
            summary.append({
                "category": s.category,
                "rule": s.rule,
                "makespan": s.metrics.makespan,
                "utilization": s.metrics.utilization,
                "tardiness": s.metrics.total_tardiness,
                "transport_time": s.metrics.total_transport_time,
                "vehicle_utilization": s.metrics.vehicle_utilization,
                "transport_wait_time": s.metrics.transport_wait_time,
                "busiest_vehicle": s.metrics.busiest_vehicle,
                "busiest_path": s.metrics.busiest_path,
                "path_conflicts": s.metrics.path_conflicts,
                "machine_idle_reasons": s.metrics.machine_idle_reasons,
                "events": s.metrics.num_events,
                "is_complete": s.metrics.is_complete,
                "total_ops_expected": s.metrics.total_ops_expected
            })
        
        # 排序逻辑：
        # 1. is_complete (True 优先，即 not x["is_complete"] 为 0)
        # 2. num_events (越大越好，对于不完整的方案，完成越多越靠前)
        # 3. makespan (越小越好)
        return sorted(summary, key=lambda x: (not x["is_complete"], -x["events"], x["makespan"]))

    def _generate_llm_brief(self, schemes: List[ScheduleScheme]) -> str:
        """生成 LLM 可解释性简报"""
        if not schemes: return "❌ 错误：没有任何调度策略成功执行。请检查输入数据中的机器、工件和布局配置。"

        valid_schemes = [s for s in schemes if s.metrics.makespan > 0 and s.metrics.num_events > 0]
        if not valid_schemes:
            return f"❌ 严重警告：所有调度策略生成的计划均为空 (0 步)。\n原因分析：可能存在资源死锁（如所有工序都需要运输但没有空闲车辆）、所有机器初始状态为故障、或工件释放时间晚于仿真截止时间。"

        complete_valid = [s for s in valid_schemes if s.metrics.is_complete]
        target_pool = complete_valid if complete_valid else valid_schemes
        best_scheme = min(target_pool, key=lambda s: s.metrics.makespan)

        status_msg = ""
        if not best_scheme.metrics.is_complete:
            status_msg = f"⚠️ 注意：当前没有任何策略能完成全部任务。以下分析基于执行进度最快的方案 {best_scheme.rule} ({best_scheme.metrics.num_events}/{best_scheme.metrics.total_ops_expected} 步)。\n\n"

        llm_payload = build_llm_plan_payload({
            "objective": "makespan",
            "best_rule": best_scheme.rule,
            "best_metrics": best_scheme.metrics.model_dump(),
            "best_schedule_plan": [step.model_dump() for step in best_scheme.plan],
            "is_complete": best_scheme.metrics.is_complete,
            "all_rule_results": [
                {
                    "rule": s.rule,
                    "metrics": s.metrics.model_dump(),
                    "plan": [step.model_dump() for step in s.plan]
                }
                for s in schemes
            ]
        })
        
        brief = build_llm_plan_brief(llm_payload)
        return status_msg + brief
