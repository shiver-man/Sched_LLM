import { useEffect, useMemo, useRef, useState } from "react";
import ChatWindow from "./components/ChatWindow";
import MessageInput from "./components/MessageInput";
import StatusPanel from "./components/StatusPanel";
import { createProgressSocket, sendTextToSchedule, uploadImage } from "./api/schedule";
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
  const [endpoint, setEndpoint] = useState<PlanEndpoint>("/run");
  const [runMode, setRunMode] = useState<UnifiedRunMode>("compare_all");
  const [ppoPolicyId, setPpoPolicyId] = useState("latest");
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState<ScheduleMetrics | undefined>(initialMetrics);
  const [latestPlan, setLatestPlan] = useState<unknown>(undefined);
  const [latestBaseData, setLatestBaseData] = useState<ScheduleBaseData | undefined>(undefined);
  const [socketEnabled, setSocketEnabled] = useState(false);
  const [socketFeed, setSocketFeed] = useState<string[]>([]);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!socketEnabled) {
      socketRef.current?.close();
      socketRef.current = null;
      return;
    }
    const socket = createProgressSocket((value) => {
      setSocketFeed((prev) => [...prev.slice(-8), value]);
    });
    socketRef.current = socket;
    return () => socket.close();
  }, [socketEnabled]);

  const socketPreview = useMemo(() => socketFeed.join("\n"), [socketFeed]);

  const handleSendText = async (text: string) => {
    setMessages((prev) => [...prev, createMessage("user", text, "text")]);
    const baseData = extractScheduleBaseData(text);
    setLatestBaseData(baseData);
    setLoading(true);
    try {
      const result = await sendTextToSchedule(endpoint, text, baseData, {
        mode: runMode,
        ppoPolicyId
      });
      if (result.metrics) {
        setMetrics(result.metrics);
      }
      if (result.plan) {
        setLatestPlan(result.plan);
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
    setLoading(true);
    try {
      const result = await uploadImage(file);
      if (result.metrics) {
        setMetrics(result.metrics);
      }
      if (result.plan) {
        setLatestPlan(result.plan);
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
              调度接口
              <select value={endpoint} onChange={(event) => setEndpoint(event.target.value as PlanEndpoint)}>
                <option value="/run">/run (统一多策略实验平台)</option>
                <option value="/schedule/run">/schedule/run (GA)</option>
                <option value="/simulation/ppo-plan">/simulation/ppo-plan (PPO)</option>
              </select>
            </label>
            <label className="field-label">
              运行模式
              <select value={runMode} onChange={(event) => setRunMode(event.target.value as UnifiedRunMode)}>
                <option value="compare_all">compare_all</option>
                <option value="ppo_plan">ppo_plan</option>
                <option value="ga_plan">ga_plan</option>
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
              <input
                type="checkbox"
                checked={socketEnabled}
                onChange={(event) => setSocketEnabled(event.target.checked)}
              />
              WebSocket 进度
            </label>
          </div>
        </header>
        <ChatWindow messages={messages} loading={loading} />
        <MessageInput onSendText={handleSendText} onUploadImage={handleUploadImage} disabled={loading} />
        {socketEnabled ? <pre className="socket-feed">{socketPreview || "等待后端进度消息..."}</pre> : null}
      </section>
      <StatusPanel metrics={metrics} latestPlan={latestPlan} endpoint={endpoint} latestBaseData={latestBaseData} />
    </main>
  );
}
