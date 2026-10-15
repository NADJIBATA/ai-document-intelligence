import { useEffect, useRef } from "react";

import type { Source } from "../../api/types";
import { MessageBubble } from "./MessageBubble";
import type { UiMessage } from "./types";

interface MessageListProps {
  messages: UiMessage[];
  onSelectSource: (source: Source) => void;
  selectedChunkId?: string | null;
}

export function MessageList({ messages, onSelectSource, selectedChunkId }: MessageListProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const lastMessage = messages[messages.length - 1];

  // Keep the latest message in view when one is added or when the answer arrives.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages.length, lastMessage?.id, lastMessage?.pending]);

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-4 py-6">
      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          message={message}
          onSelectSource={onSelectSource}
          selectedChunkId={selectedChunkId}
        />
      ))}
      <div ref={endRef} />
    </div>
  );
}
