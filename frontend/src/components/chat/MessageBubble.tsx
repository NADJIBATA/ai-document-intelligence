import { useMemo } from "react";

import type { Source } from "../../api/types";
import { classNames, formatDuration } from "../../lib/format";
import { AlertIcon, SparklesIcon } from "../ui/Icons";
import { AnswerText } from "./AnswerText";
import { SourceList } from "./SourceList";
import type { UiMessage } from "./types";

interface MessageBubbleProps {
  message: UiMessage;
  onSelectSource: (source: Source) => void;
  selectedChunkId?: string | null;
}

export function MessageBubble({ message, onSelectSource, selectedChunkId }: MessageBubbleProps) {
  const sourcesByRef = useMemo(
    () => new Map(message.sources.map((source) => [source.ref, source])),
    [message.sources],
  );
  const knownRefs = useMemo(() => new Set(sourcesByRef.keys()), [sourcesByRef]);

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-brand-600 px-4 py-2.5 text-sm leading-relaxed text-white shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      <div
        className={classNames(
          "mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          message.failed ? "bg-red-100 text-red-600" : "bg-slate-900 text-white",
        )}
        aria-hidden="true"
      >
        {message.failed ? (
          <AlertIcon width={16} height={16} />
        ) : (
          <SparklesIcon width={16} height={16} />
        )}
      </div>
      <div
        className={classNames(
          "min-w-0 max-w-[85%] rounded-2xl rounded-tl-md border px-4 py-3 text-sm shadow-sm",
          message.failed
            ? "border-red-200 bg-red-50 text-red-800"
            : message.answered === false
              ? "border-slate-200 bg-slate-50 text-slate-600"
              : "border-slate-200 bg-white text-slate-800",
        )}
      >
        {message.pending ? (
          <div className="flex items-center gap-2 text-slate-500" role="status">
            <span className="flex gap-1" aria-hidden="true">
              <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400" />
            </span>
            Searching your documents…
          </div>
        ) : message.failed ? (
          <p className="break-words">{message.content}</p>
        ) : (
          <>
            <AnswerText
              text={message.content}
              knownRefs={knownRefs}
              onCitationClick={(ref) => {
                const source = sourcesByRef.get(ref);
                if (source) {
                  onSelectSource(source);
                }
              }}
            />
            <SourceList
              sources={message.sources}
              onSelect={onSelectSource}
              selectedChunkId={selectedChunkId}
            />
            {message.metadata ? (
              <p className="mt-2 text-[11px] text-slate-400">
                {message.metadata.model ? `${message.metadata.model} · ` : ""}
                {formatDuration(message.metadata.total_ms)}
                {message.metadata.degraded.length > 0
                  ? ` · degraded search (${message.metadata.degraded.join(", ")} unavailable)`
                  : ""}
              </p>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}
