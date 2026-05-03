import type { PlanEndpoint, ScheduleBaseData, ScheduleMetrics } from "../types";

interface StatusPanelProps {
  metrics?: ScheduleMetrics;
  latestPlan?: unknown;
  summaryComparison?: unknown;
  ganttImage?: string;
  latestBaseData?: ScheduleBaseData;
  loading?: boolean;
}

function toDisplay(value: number | undefined, fixed = 2, loading = false): string {
  if (loading) {
    return "...";
  }
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(fixed);
}

export default function StatusPanel({
  metrics,
  latestPlan,
  summaryComparison,
  ganttImage,
  latestBaseData,
  loading = false
}: StatusPanelProps) {
  return (
    <aside className="status-panel">
      <h3>调度状态</h3>
      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-label">Makespan</div>
          <div className="metric-value">{toDisplay(metrics?.makespan, 1, loading)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Vehicle Utilization</div>
          <div className="metric-value">{toDisplay(metrics?.vehicleUtilization, 2, loading)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Transport Wait Time</div>
          <div className="metric-value">{toDisplay(metrics?.transportWaitTime, 1, loading)}</div>
        </div>
      </div>

      {loading && !ganttImage && (
        <div className="gantt-loading">
          <div className="spinner"></div>
          <span>正在生成甘特图...</span>
        </div>
      )}

      {ganttImage && (
        <div className="gantt-section">
          <h4>排产甘特图</h4>
          <div className="gantt-container">
            <img src={ganttImage} alt="排产甘特图" className="gantt-img" />
          </div>
        </div>
      )}

      {summaryComparison && (
        <>
          <h4>策略排行榜 (对比实验)</h4>
          <pre className="plan-box">
            {JSON.stringify(summaryComparison, null, 2)}
          </pre>
          {Array.isArray(summaryComparison) && summaryComparison.find((x: any) => x.rule === "TS_GA") && (
            <div className="special-highlight">
              <strong>TS_GA 核心指标:</strong>
              <pre>{JSON.stringify(summaryComparison.find((x: any) => x.rule === "TS_GA"), null, 2)}</pre>
            </div>
          )}
        </>
      )}

      <h4>最优调度计划</h4>
      <pre className="plan-box">
        {loading ? "正在计算中，请稍候..." : (latestPlan ? JSON.stringify(latestPlan, null, 2) : "暂无返回计划数据")}
      </pre>

      <h4>基础数据 JSON</h4>
      <pre className="base-data-box">
        {latestBaseData ? JSON.stringify(latestBaseData, null, 2) : "发送文本后将自动提取关键字段并生成基础数据"}
      </pre>
    </aside>
  );
}
