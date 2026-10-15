import { useEffect, useRef, type FormEvent, type KeyboardEvent } from "react";

import { classNames } from "../../lib/format";
import { SendIcon } from "../ui/Icons";

export const MAX_QUESTION_CHARS = 2000;

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({ value, onChange, onSubmit, disabled, placeholder }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const canSend = !disabled && value.trim().length > 0 && value.length <= MAX_QUESTION_CHARS;

  // Grow with the content, up to a few lines.
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [value]);

  function submit(event?: FormEvent): void {
    event?.preventDefault();
    if (canSend) {
      onSubmit();
    }
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): void {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <form onSubmit={submit} className="mx-auto w-full max-w-3xl px-4 pb-4">
      <div className="flex items-end gap-2 rounded-2xl border border-slate-300 bg-white p-2 shadow-sm focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-200">
        <label htmlFor="chat-question" className="sr-only">
          Your question
        </label>
        <textarea
          id="chat-question"
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={placeholder ?? "Ask about your documents…"}
          className="max-h-[200px] min-h-[40px] flex-1 resize-none border-0 bg-transparent px-2 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-0"
        />
        <button
          type="submit"
          disabled={!canSend}
          className={classNames(
            "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl transition-colors",
            canSend
              ? "bg-brand-600 text-white hover:bg-brand-700"
              : "cursor-not-allowed bg-slate-100 text-slate-400",
          )}
          aria-label="Send"
        >
          <SendIcon width={18} height={18} />
        </button>
      </div>
      <div className="mt-1 flex justify-between px-2 text-[11px] text-slate-400">
        <span>Enter to send · Shift + Enter for a new line</span>
        {value.length > MAX_QUESTION_CHARS * 0.8 ? (
          <span className={value.length > MAX_QUESTION_CHARS ? "text-red-600" : undefined}>
            {value.length} / {MAX_QUESTION_CHARS}
          </span>
        ) : null}
      </div>
    </form>
  );
}
