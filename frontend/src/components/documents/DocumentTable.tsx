import { Link } from "react-router-dom";

import type { DocumentItem } from "../../api/types";
import { formatBytes, formatDateTime } from "../../lib/format";
import { Button } from "../ui/Button";
import { DownloadIcon, RefreshIcon, SparklesIcon, TrashIcon } from "../ui/Icons";
import { StatusBadge } from "./StatusBadge";

interface DocumentTableProps {
  documents: DocumentItem[];
  onDelete: (document: DocumentItem) => void;
  onReprocess: (document: DocumentItem) => void;
  onDownload: (document: DocumentItem) => void;
  /** Id of the document whose action is running (buttons disabled). */
  busyId?: string | null;
}

export function DocumentTable({
  documents,
  onDelete,
  onReprocess,
  onDownload,
  busyId,
}: DocumentTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-slate-200">
        <thead className="bg-slate-50">
          <tr>
            <th scope="col" className="table-header">
              Filename
            </th>
            <th scope="col" className="table-header">
              Status
            </th>
            <th scope="col" className="table-header">
              Upload date
            </th>
            <th scope="col" className="table-header">
              Size
            </th>
            <th scope="col" className="table-header text-right">
              Actions
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 bg-white">
          {documents.map((document) => {
            const busy = busyId === document.id;
            return (
              <tr key={document.id} className="hover:bg-slate-50/60">
                <td className="table-cell max-w-xs">
                  <p className="truncate font-medium text-slate-900" title={document.filename}>
                    {document.filename}
                  </p>
                  <p className="text-xs text-slate-500">
                    {document.extension.toUpperCase()}
                    {document.page_count ? ` · ${document.page_count} pages` : ""}
                    {document.status === "indexed" ? ` · ${document.chunk_count} passages` : ""}
                  </p>
                  {document.status === "failed" && document.error_message ? (
                    <p className="mt-1 whitespace-normal text-xs text-red-600">
                      {document.error_message}
                    </p>
                  ) : null}
                </td>
                <td className="table-cell whitespace-nowrap">
                  <StatusBadge status={document.status} />
                </td>
                <td className="table-cell whitespace-nowrap">
                  {formatDateTime(document.created_at)}
                </td>
                <td className="table-cell whitespace-nowrap">{formatBytes(document.size_bytes)}</td>
                <td className="table-cell whitespace-nowrap text-right">
                  <div className="flex justify-end gap-1">
                    {document.status === "indexed" ? (
                      <Link
                        to={`/intelligence/${document.id}`}
                        className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium text-brand-700 hover:bg-brand-50"
                      >
                        <SparklesIcon width={16} height={16} />
                        Insights
                      </Link>
                    ) : null}
                    {document.status === "failed" ? (
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={busy}
                        icon={<RefreshIcon width={16} height={16} />}
                        onClick={() => onReprocess(document)}
                      >
                        Retry
                      </Button>
                    ) : null}
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={busy}
                      icon={<DownloadIcon width={16} height={16} />}
                      onClick={() => onDownload(document)}
                      aria-label={`Download ${document.filename}`}
                    >
                      Download
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={busy}
                      className="text-red-600 hover:bg-red-50"
                      icon={<TrashIcon width={16} height={16} />}
                      onClick={() => onDelete(document)}
                      aria-label={`Delete ${document.filename}`}
                    >
                      Delete
                    </Button>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
