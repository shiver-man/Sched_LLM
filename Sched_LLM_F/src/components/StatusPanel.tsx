import type { PlanEndpoint, ScheduleBaseData, ScheduleMetrics } from "../types";

interface StatusPanelProps {
  metrics?: ScheduleMetrics;
  latestPlan?: unknown;
  latestBaseData?: ScheduleBaseData;
  endpoint: PlanEndpoint;
}

function toDisplay(value: number | undefined, fixed = 2): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(fixed);
}

export default function StatusPanel({ metrics, latestPlan, endpoint, latestBaseData }: StatusPanelProps) {
  return (
    <aside className="status-panel">
      <h3>调度状态</h3>
      <div className="tag-row">
        <span className="endpoint-tag">{endpoint}</span>
      </div>
      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-label">Makespan</div>
          <div className="metric-value">{toDisplay(metrics?.makespan, 1)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Vehicle Utilization</div>
          <div className="metric-value">{toDisplay(metrics?.vehicleUtilization, 2)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Transport Wait Time</div>
          <div className="metric-value">{toDisplay(metrics?.transportWaitTime, 1)}</div>
        </div>
      </div>
      <h4>GA / PPO 计划</h4>
      <pre className="plan-box">{latestPlan ? JSON.stringify(latestPlan, null, 2) : "暂无返回计划数据"}</pre>
      <h4>基础数据 JSON</h4>
      <pre className="base-data-box">
        {latestBaseData ? JSON.stringify(latestBaseData, null, 2) : "发送文本后将自动提取关键字段并生成基础数据"}
      </pre>
    </aside>
  );
}
