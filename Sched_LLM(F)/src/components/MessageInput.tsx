import { useState, type KeyboardEvent } from "react";
import ImageUploader from "./ImageUploader";

interface MessageInputProps {
  onSendText: (text: string) => Promise<void>;
  onUploadImage: (file: File) => Promise<void>;
  disabled?: boolean;
}

export default function MessageInput({ onSendText, onUploadImage, disabled = false }: MessageInputProps) {
  const [text, setText] = useState("");

  const send = async () => {
    const value = text.trim();
    if (!value || disabled) {
      return;
    }
    setText("");
    await onSendText(value);
  };

  const onKeyDown = async (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      await send();
    }
  };

  return (
    <section className="input-area">
      <textarea
        className="text-input"
        value={text}
        disabled={disabled}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder="输入任务描述或调度命令，例如：请给我 PPO 计划，目标最小化 makespan"
      />
      <div className="input-actions">
        <ImageUploader onUpload={onUploadImage} disabled={disabled} />
        <button className="send-btn" type="button" onClick={send} disabled={disabled || !text.trim()}>
          发送
        </button>
      </div>
    </section>
  );
}
