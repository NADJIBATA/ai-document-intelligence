import type { ReactNode } from "react";

import { classNames } from "../../lib/format";
import { AlertIcon } from "./Icons";

interface AlertProps {
  tone?: "error" | "warning" | "info";
  children: ReactNode;
  action?: ReactNode;
  className?: string;
}

const TONES = {
  error: "border-red-200 bg-red-50 text-red-800",
  warning: "border-amber-200 bg-amber-50 text-amber-900",
  info: "border-brand-200 bg-brand-50 text-brand-900",
} as const;

export function Alert({ tone = "error", children, action, className }: AlertProps) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={classNames(
        "flex items-start gap-3 rounded-lg border px-4 py-3 text-sm",
        TONES[tone],
        className,
      )}
    >
      <AlertIcon className="mt-0.5 shrink-0" width={18} height={18} />
      <div className="min-w-0 flex-1">{children}</div>
      {action}
    </div>
  );
}
