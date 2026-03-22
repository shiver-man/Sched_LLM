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
          <div className="message-bubble assistant">
            <span className="typing-dot" />
            <span className="typing-dot" />
            <span className="typing-dot" />
          </div>
        </div>
      ) : null}
    </section>
  );
}
