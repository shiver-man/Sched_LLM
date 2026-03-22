import type { ChatMessage } from "../types";

interface MessageBubbleProps {
  message: ChatMessage;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const bubbleClass = isUser ? "message-bubble user" : "message-bubble assistant";

  return (
    <div className={isUser ? "message-row user-row" : "message-row assistant-row"}>
      <div className={bubbleClass}>
        {message.contentType === "image" ? (
          <img src={message.content} alt="用户上传图像" className="message-image" />
        ) : (
          <pre className="message-text">{message.content}</pre>
        )}
      </div>
    </div>
  );
}
