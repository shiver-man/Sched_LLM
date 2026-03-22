import { useRef, useState, type DragEvent } from "react";

interface ImageUploaderProps {
  onUpload: (file: File) => Promise<void>;
  disabled?: boolean;
}

export default function ImageUploader({ onUpload, disabled = false }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const submitFile = async (file: File | undefined) => {
    if (!file || disabled) {
      return;
    }
    await onUpload(file);
  };

  const handleDrop = async (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    await submitFile(file);
  };

  return (
    <>
      <input
        ref={inputRef}
        className="hidden-file-input"
        type="file"
        accept="image/*"
        onChange={async (event) => {
          const file = event.target.files?.[0];
          event.currentTarget.value = "";
          await submitFile(file);
        }}
      />
      <button
        className={dragging ? "upload-btn dragging" : "upload-btn"}
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) {
            setDragging(true);
          }
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        上传图片
      </button>
    </>
  );
}
