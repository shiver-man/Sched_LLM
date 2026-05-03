import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types";
import MessageBubble from "./MessageBubble";

interface ChatWindowProps {
  messages: ChatMessage[];
  loading: boolean;
}

export default function ChatWindow({ messages, loading }: ChatWindowProps) {
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!listRef.current) {
      return;
    }
    listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages, loading]);

  return (
    <section className="chat-window" ref={listRef}>
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      {loading ? (
        <div className="message-row assistant-row">
          <div className="message-bubble assistant loading-bubble">
            <div className="typing-dots">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
            <div className="loading-text">大模型正在深度分析排产策略中，请稍候...</div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
