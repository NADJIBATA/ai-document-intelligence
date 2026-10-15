import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { askQuestion, deleteConversation, getConversation } from "../api/chat";
import { listDocuments } from "../api/documents";
import type { ConversationSummary, Message, PassageRef, Source } from "../api/types";
import { ChatInput } from "../components/chat/ChatInput";
import { ConversationList } from "../components/chat/ConversationList";
import { DocumentFilter } from "../components/chat/DocumentFilter";
import { MessageList } from "../components/chat/MessageList";
import { SourcePanel } from "../components/chat/SourcePanel";
import type { UiMessage } from "../components/chat/types";
import { Alert } from "../components/ui/Alert";
import { Button } from "../components/ui/Button";
import { ChatIcon, PlusIcon } from "../components/ui/Icons";
import { Modal } from "../components/ui/Modal";
import { Spinner } from "../components/ui/Spinner";
import { useToast } from "../components/ui/toast-context";
import { useApi } from "../hooks/useApi";
import { useConversations } from "../hooks/useConversations";
import { errorMessage, isAbortError, isNotFound } from "../lib/errors";

const SUGGESTIONS = [
  "What are the main obligations of each party?",
  "What is the notice period to terminate?",
  "Summarize the payment terms.",
];

function toUiMessage(message: Message): UiMessage {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    sources: message.sources,
    answered: message.answered,
  };
}

let localId = 0;
function nextLocalId(prefix: string): string {
  localId += 1;
  return `${prefix}-${localId}`;
}

export function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const { notify } = useToast();
  const conversations = useConversations();
  const indexedDocuments = useApi(
    (signal) => listDocuments({ status: "indexed", limit: 100 }, signal),
    [],
  );

  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [passage, setPassage] = useState<PassageRef | null>(null);
  const [pendingDelete, setPendingDelete] = useState<ConversationSummary | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);

  // Conversation shown right now (null = new chat): answers arriving for another one
  // must not be appended to the messages on screen.
  const currentConversation = useRef<string | null>(conversationId ?? null);
  currentConversation.current = conversationId ?? null;
  // Set when a new conversation has just been created from this page: its messages are
  // already on screen, there is no need to fetch them again.
  const alreadyLoaded = useRef<string | null>(null);

  useEffect(() => {
    setPassage(null);
    setLoadError(null);
    if (alreadyLoaded.current !== null && alreadyLoaded.current === conversationId) {
      return; // just created here: its messages are on screen
    }
    alreadyLoaded.current = null;
    if (!conversationId) {
      setMessages([]);
      setLoadingConversation(false);
      return;
    }
    const controller = new AbortController();
    setMessages([]);
    setLoadingConversation(true);
    getConversation(conversationId, controller.signal)
      .then((detail) => {
        if (!controller.signal.aborted) {
          setMessages(detail.messages.map(toUiMessage));
          setLoadingConversation(false);
        }
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted || isAbortError(error)) {
          return;
        }
        setLoadingConversation(false);
        setLoadError(
          isNotFound(error) ? "This conversation does not exist any more." : errorMessage(error),
        );
      });
    return () => controller.abort();
  }, [conversationId]);

  async function send(): Promise<void> {
    const question = input.trim();
    if (!question || sending) {
      return;
    }
    const origin = conversationId ?? null;
    const questionId = nextLocalId("question");
    const pendingId = nextLocalId("pending");
    setMessages((current) => [
      ...current.filter((message) => !message.failed),
      { id: questionId, role: "user", content: question, sources: [], answered: null },
      { id: pendingId, role: "assistant", content: "", sources: [], answered: null, pending: true },
    ]);
    setInput("");
    setSending(true);
    try {
      const response = await askQuestion({
        question,
        conversationId: origin,
        documentIds: selectedDocumentIds,
      });
      if (currentConversation.current === origin) {
        setMessages((current) =>
          current.map((message) =>
            message.id === pendingId
              ? {
                  id: response.message_id,
                  role: "assistant",
                  content: response.answer,
                  sources: response.sources,
                  answered: response.answered,
                  metadata: response.metadata,
                }
              : message,
          ),
        );
        if (origin === null) {
          alreadyLoaded.current = response.conversation_id;
          navigate(`/chat/${response.conversation_id}`, { replace: true });
        }
      }
      conversations.reload();
    } catch (error) {
      if (currentConversation.current === origin) {
        // Nothing was saved: the question goes back to the input for another try.
        setMessages((current) => [
          ...current.filter((message) => message.id !== questionId && message.id !== pendingId),
          {
            id: nextLocalId("error"),
            role: "assistant",
            content: errorMessage(error),
            sources: [],
            answered: null,
            failed: true,
          },
        ]);
        setInput(question);
      } else {
        notify(errorMessage(error), "error");
      }
    } finally {
      setSending(false);
    }
  }

  async function confirmDelete(): Promise<void> {
    if (pendingDelete === null) {
      return;
    }
    setDeleting(true);
    try {
      await deleteConversation(pendingDelete.id);
      if (pendingDelete.id === conversationId) {
        navigate("/chat", { replace: true });
      }
      setPendingDelete(null);
      conversations.reload();
    } catch (error) {
      notify(errorMessage(error), "error");
    } finally {
      setDeleting(false);
    }
  }

  function selectSource(source: Source): void {
    setPassage(source);
  }

  const documents = indexedDocuments.data?.items ?? [];
  const noDocuments = indexedDocuments.data !== undefined && documents.length === 0;
  const title =
    conversations.data?.items.find((conversation) => conversation.id === conversationId)?.title ??
    (conversationId ? "Conversation" : "New conversation");

  const history = (
    <>
      <div className="flex items-center justify-between gap-2 border-b border-slate-100 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-900">Conversations</h2>
        <Link
          to="/chat"
          onClick={() => setHistoryOpen(false)}
          className="inline-flex items-center gap-1 rounded-lg bg-brand-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-brand-700"
        >
          <PlusIcon width={14} height={14} />
          New chat
        </Link>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {conversations.loading ? (
          <div className="flex justify-center py-6 text-slate-400">
            <Spinner size="sm" />
          </div>
        ) : conversations.error ? (
          <p className="px-3 py-4 text-sm text-red-600">{errorMessage(conversations.error)}</p>
        ) : (
          <ConversationList
            conversations={conversations.data?.items ?? []}
            onDelete={setPendingDelete}
            onNavigate={() => setHistoryOpen(false)}
          />
        )}
      </div>
    </>
  );

  return (
    <div className="flex h-full">
      <aside className="hidden w-72 shrink-0 flex-col border-r border-slate-200 bg-white md:flex">
        {history}
      </aside>

      {historyOpen ? (
        <div className="fixed inset-0 z-40 md:hidden">
          <div
            className="absolute inset-0 bg-slate-900/50"
            onClick={() => setHistoryOpen(false)}
            aria-hidden="true"
          />
          <div className="relative flex h-full w-72 flex-col bg-white shadow-xl">{history}</div>
        </div>
      ) : null}

      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-2.5">
          <div className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              onClick={() => setHistoryOpen(true)}
              className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 md:hidden"
            >
              <ChatIcon width={14} height={14} />
              History
            </button>
            <h1 className="truncate text-sm font-semibold text-slate-900" title={title}>
              {title}
            </h1>
          </div>
          <DocumentFilter
            documents={documents}
            selectedIds={selectedDocumentIds}
            onChange={setSelectedDocumentIds}
          />
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {loadingConversation ? (
            <div className="flex h-full items-center justify-center text-slate-400">
              <Spinner showLabel label="Loading conversation" />
            </div>
          ) : loadError ? (
            <div className="mx-auto max-w-3xl px-4 py-6">
              <Alert
                action={
                  <Link to="/chat" className="text-sm font-medium underline">
                    New chat
                  </Link>
                }
              >
                {loadError}
              </Alert>
            </div>
          ) : messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center px-4 text-center">
              <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-brand-50 text-brand-600">
                <ChatIcon width={28} height={28} />
              </div>
              <h2 className="text-lg font-semibold text-slate-900">
                Ask a question about your documents
              </h2>
              <p className="mt-1 max-w-md text-sm text-slate-500">
                Answers are written only from your documents, and every statement links to the
                passage it comes from.
              </p>
              {noDocuments ? (
                <Alert tone="info" className="mt-6 max-w-md text-left">
                  You have no indexed document yet.{" "}
                  <Link to="/documents" className="font-medium underline">
                    Upload a document
                  </Link>{" "}
                  to get answers.
                </Alert>
              ) : (
                <div className="mt-6 flex max-w-xl flex-wrap justify-center gap-2">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      onClick={() => setInput(suggestion)}
                      className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700 hover:border-brand-300 hover:bg-brand-50"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <MessageList
              messages={messages}
              onSelectSource={selectSource}
              selectedChunkId={passage?.chunk_id}
            />
          )}
        </div>

        <ChatInput
          value={input}
          onChange={setInput}
          onSubmit={() => void send()}
          disabled={sending || loadingConversation || loadError !== null}
        />
      </section>

      <SourcePanel passage={passage} onClose={() => setPassage(null)} docked />

      <Modal
        open={pendingDelete !== null}
        title="Delete this conversation?"
        onClose={() => setPendingDelete(null)}
        footer={
          <>
            <Button variant="secondary" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button variant="danger" loading={deleting} onClick={() => void confirmDelete()}>
              Delete
            </Button>
          </>
        }
      >
        <p>
          <span className="font-medium text-slate-900">{pendingDelete?.title}</span> and its
          messages will be permanently removed.
        </p>
      </Modal>
    </div>
  );
}
