import { useEffect, useMemo, useRef, useState } from "react";
import ChatWindow from "./components/ChatWindow";
import MessageInput from "./components/MessageInput";
import StatusPanel from "./components/StatusPanel";
import { sendTextToSchedule, uploadImage } from "./api/schedule";
import { initialMessages, initialMetrics } from "./data/mock";
import { extractScheduleBaseData } from "./utils/extractor";
import type { ChatMessage, PlanEndpoint, ScheduleBaseData, ScheduleMetrics, UnifiedRunMode } from "./types";

function createMessage(role: "user" | "assistant", content: string, contentType: "text" | "image"): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    contentType,
    createdAt: Date.now()
  };
}

async function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(new Error("读取图片失败"));
    reader.readAsDataURL(file);
  });
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const endpoint: PlanEndpoint = "/run";
  const [runMode, setRunMode] = useState<UnifiedRunMode>("compare_all");
  const [ppoPolicyId, setPpoPolicyId] = useState("latest");
  const [useLlm, setUseLlm] = useState(true);
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState<ScheduleMetrics | undefined>(initialMetrics);
  const [latestPlan, setLatestPlan] = useState<unknown>(undefined);
  const [summaryComparison, setSummaryComparison] = useState<unknown>(undefined);
  const [detailedSchemes, setDetailedSchemes] = useState<unknown>(undefined);
  const [ganttImage, setGanttImage] = useState<string | undefined>(undefined);
  const [latestBaseData, setLatestBaseData] = useState<ScheduleBaseData | undefined>(undefined);

  const handleSendText = async (text: string) => {
    setMessages((prev) => [...prev, createMessage("user", text, "text")]);
    const baseData = extractScheduleBaseData(text);
    setLatestBaseData(baseData);
    setMetrics(undefined); // 清理旧指标
    setLatestPlan(undefined); // 清理旧计划
    setSummaryComparison(undefined); // 清理旧对比
    setDetailedSchemes(undefined); // 清理旧轨迹
    setGanttImage(undefined); // 清理旧甘特图
    setLoading(true);
    try {
      const result = await sendTextToSchedule(endpoint, text, baseData, {
        mode: runMode,
        ppoPolicyId,
        useLlm
      });
      if (result.metrics) {
        setMetrics(result.metrics);
      }
      if (result.plan) {
        setLatestPlan(result.plan);
      }
      if (result.summary_comparison) {
        setSummaryComparison(result.summary_comparison);
      }
      if (result.detailed_schemes) {
        setDetailedSchemes(result.detailed_schemes);
      }
      if (result.gantt_chart_base64) {
        setGanttImage(result.gantt_chart_base64);
      }
      setMessages((prev) => [...prev, createMessage("assistant", result.replyText, "text")]);
    } catch (error) {
      const textError = error instanceof Error ? error.message : "未知错误";
      setMessages((prev) => [...prev, createMessage("assistant", `请求失败：${textError}`, "text")]);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadImage = async (file: File) => {
    const imageDataUrl = await fileToDataUrl(file);
    setMessages((prev) => [...prev, createMessage("user", imageDataUrl, "image")]);
    setMetrics(undefined); // 清理旧指标
    setLatestPlan(undefined); // 清理旧计划
    setSummaryComparison(undefined); // 清理旧对比
    setDetailedSchemes(undefined); // 清理旧轨迹
    setGanttImage(undefined); // 清理旧甘特图
    setLoading(true);
    try {
      const result = await uploadImage(file);
      if (result.metrics) {
        setMetrics(result.metrics);
      }
      if (result.plan) {
        setLatestPlan(result.plan);
      }
      if (result.summary_comparison) {
        setSummaryComparison(result.summary_comparison);
      }
      if (result.detailed_schemes) {
        setDetailedSchemes(result.detailed_schemes);
      }
      if (result.gantt_chart_base64) {
        setGanttImage(result.gantt_chart_base64);
      }
      setMessages((prev) => [...prev, createMessage("assistant", result.replyText, "text")]);
    } catch (error) {
      const textError = error instanceof Error ? error.message : "未知错误";
      setMessages((prev) => [...prev, createMessage("assistant", `上传失败：${textError}`, "text")]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="main-panel">
        <header className="top-bar">
          <h1>Sched_LLM Desktop</h1>
          <div className="toolbar">
            <label className="field-label">
              运行模式
              <select value={runMode} onChange={(event) => setRunMode(event.target.value as UnifiedRunMode)}>
                <option value="compare_all">多策略对比 (推荐)</option>
                <option value="ga_only">仅 GA 算法</option>
                <option value="ppo_only">仅 PPO 算法</option>
              </select>
            </label>
            <label className="field-label">
              PPO Policy
              <input
                className="inline-input"
                value={ppoPolicyId}
                onChange={(event) => setPpoPolicyId(event.target.value)}
                placeholder="latest"
              />
            </label>
            <label className="switch-label">
              <input type="checkbox" checked={useLlm} onChange={(event) => setUseLlm(event.target.checked)} />
              大模型分析
            </label>
          </div>
        </header>
        <ChatWindow messages={messages} loading={loading} />
        <MessageInput onSendText={handleSendText} onUploadImage={handleUploadImage} disabled={loading} />
      </section>
      <StatusPanel
        metrics={metrics}
        latestPlan={latestPlan}
        summaryComparison={summaryComparison}
        detailedSchemes={detailedSchemes}
        ganttImage={ganttImage}
        endpoint={endpoint}
        latestBaseData={latestBaseData}
        loading={loading}
      />
    </main>
  );
}
