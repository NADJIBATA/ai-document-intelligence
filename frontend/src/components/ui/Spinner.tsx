import { classNames } from "../../lib/format";

interface SpinnerProps {
  /** Text announced to screen readers, and displayed when `showLabel` is set. */
  label?: string;
  showLabel?: boolean;
  size?: "sm" | "md";
  className?: string;
}

export function Spinner({ label = "Loading", showLabel = false, size = "md", className }: SpinnerProps) {
  return (
    <span role="status" className={classNames("inline-flex items-center gap-2", className)}>
      <span
        className={classNames(
          "inline-block animate-spin rounded-full border-2 border-current border-r-transparent",
          size === "sm" ? "h-4 w-4" : "h-6 w-6",
        )}
        aria-hidden="true"
      />
      <span className={showLabel ? "text-sm" : "sr-only"}>{label}</span>
    </span>
  );
}
