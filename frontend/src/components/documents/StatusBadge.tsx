import type { DocumentStatus, JobStatus } from "../../api/types";
import { Badge, type BadgeTone } from "../ui/Badge";
import { Spinner } from "../ui/Spinner";

const LABELS: Record<DocumentStatus | JobStatus, { label: string; tone: BadgeTone; busy: boolean }> =
  {
    uploaded: { label: "Uploaded", tone: "neutral", busy: true },
    pending: { label: "Pending", tone: "neutral", busy: true },
    processing: { label: "Processing", tone: "amber", busy: true },
    indexed: { label: "Indexed", tone: "green", busy: false },
    completed: { label: "Completed", tone: "green", busy: false },
    failed: { label: "Failed", tone: "red", busy: false },
  };

export function StatusBadge({ status }: { status: DocumentStatus | JobStatus }) {
  const { label, tone, busy } = LABELS[status];
  return (
    <Badge tone={tone}>
      {busy ? <Spinner size="sm" label="" className="scale-75" /> : null}
      {label}
    </Badge>
  );
}
