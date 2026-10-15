import { useCallback, useMemo, useRef, useState, type ReactNode } from "react";

import { classNames } from "../../lib/format";
import { AlertIcon, CheckIcon, CloseIcon } from "./Icons";
import { ToastContext, type ToastApi, type ToastTone } from "./toast-context";

interface ToastItem {
  id: number;
  message: string;
  tone: ToastTone;
}

const DURATION_MS = 5000;

const TONES: Record<ToastTone, string> = {
  success: "border-emerald-200 bg-white text-slate-800",
  error: "border-red-200 bg-white text-slate-800",
  info: "border-slate-200 bg-white text-slate-800",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const notify = useCallback(
    (message: string, tone: ToastTone = "info") => {
      const id = nextId.current++;
      setToasts((current) => [...current.slice(-3), { id, message, tone }]);
      window.setTimeout(() => dismiss(id), DURATION_MS);
    },
    [dismiss],
  );

  const api = useMemo<ToastApi>(() => ({ notify }), [notify]);

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div
        className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-full max-w-sm flex-col gap-2"
        aria-live="polite"
      >
        {toasts.map((toast) => (
          <div
            key={toast.id}
            role={toast.tone === "error" ? "alert" : "status"}
            className={classNames(
              "pointer-events-auto flex items-start gap-3 rounded-lg border px-4 py-3 text-sm shadow-lg",
              TONES[toast.tone],
            )}
          >
            <span
              className={classNames(
                "mt-0.5 shrink-0",
                toast.tone === "error" && "text-red-600",
                toast.tone === "success" && "text-emerald-600",
                toast.tone === "info" && "text-brand-600",
              )}
            >
              {toast.tone === "success" ? (
                <CheckIcon width={18} height={18} />
              ) : (
                <AlertIcon width={18} height={18} />
              )}
            </span>
            <p className="min-w-0 flex-1 break-words">{toast.message}</p>
            <button
              type="button"
              onClick={() => dismiss(toast.id)}
              className="shrink-0 rounded p-0.5 text-slate-400 hover:text-slate-600"
              aria-label="Dismiss"
            >
              <CloseIcon width={16} height={16} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
