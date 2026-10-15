import { useState } from "react";

import { deleteDocument, reprocessDocument } from "../api/documents";
import type { DocumentItem, DocumentStatus } from "../api/types";
import { DocumentTable } from "../components/documents/DocumentTable";
import { UploadButton } from "../components/documents/UploadButton";
import { PageContainer, PageHeader } from "../components/layout/PageHeader";
import { Alert } from "../components/ui/Alert";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { EmptyState } from "../components/ui/EmptyState";
import { DocumentIcon } from "../components/ui/Icons";
import { Modal } from "../components/ui/Modal";
import { Pagination } from "../components/ui/Pagination";
import { Spinner } from "../components/ui/Spinner";
import { useToast } from "../components/ui/toast-context";
import { useDocuments } from "../hooks/useDocuments";
import { errorMessage } from "../lib/errors";
import { downloadDocumentFile } from "../lib/files";
import { classNames } from "../lib/format";

const PAGE_SIZE = 20;

const FILTERS: Array<{ value: DocumentStatus | undefined; label: string }> = [
  { value: undefined, label: "All" },
  { value: "indexed", label: "Indexed" },
  { value: "processing", label: "Processing" },
  { value: "failed", label: "Failed" },
];

export function DocumentsPage() {
  const { notify } = useToast();
  const [status, setStatus] = useState<DocumentStatus | undefined>(undefined);
  const [offset, setOffset] = useState(0);
  const [pendingDelete, setPendingDelete] = useState<DocumentItem | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const documents = useDocuments({ limit: PAGE_SIZE, offset, status });

  function changeFilter(value: DocumentStatus | undefined): void {
    setStatus(value);
    setOffset(0);
  }

  async function confirmDelete(): Promise<void> {
    if (pendingDelete === null) {
      return;
    }
    const target = pendingDelete;
    setBusyId(target.id);
    try {
      await deleteDocument(target.id);
      notify(`"${target.filename}" was deleted.`, "success");
      setPendingDelete(null);
      // Deleting the last row of a page goes back to the previous one.
      if (documents.data && documents.data.items.length === 1 && offset > 0) {
        setOffset(Math.max(0, offset - PAGE_SIZE));
      } else {
        documents.reload();
      }
    } catch (error) {
      notify(errorMessage(error), "error");
    } finally {
      setBusyId(null);
    }
  }

  async function reprocess(document: DocumentItem): Promise<void> {
    setBusyId(document.id);
    try {
      await reprocessDocument(document.id);
      notify(`Processing of "${document.filename}" restarted.`, "success");
      documents.reload();
    } catch (error) {
      notify(errorMessage(error), "error");
    } finally {
      setBusyId(null);
    }
  }

  async function download(document: DocumentItem): Promise<void> {
    setBusyId(document.id);
    try {
      await downloadDocumentFile(document.id, document.filename);
    } catch (error) {
      notify(errorMessage(error), "error");
    } finally {
      setBusyId(null);
    }
  }

  const page = documents.data;

  return (
    <PageContainer>
      <PageHeader
        title="Documents"
        description="PDF, DOCX and TXT files. They are indexed in the background after the upload."
        actions={<UploadButton onUploaded={documents.reload} />}
      />

      <div className="mb-4 flex flex-wrap gap-2" role="group" aria-label="Filter by status">
        {FILTERS.map((filter) => (
          <button
            key={filter.label}
            type="button"
            onClick={() => changeFilter(filter.value)}
            aria-pressed={status === filter.value}
            className={classNames(
              "rounded-full px-3 py-1 text-sm font-medium transition-colors",
              status === filter.value
                ? "bg-slate-900 text-white"
                : "bg-white text-slate-600 ring-1 ring-inset ring-slate-200 hover:bg-slate-50",
            )}
          >
            {filter.label}
          </button>
        ))}
      </div>

      <Card className="overflow-hidden">
        {documents.loading ? (
          <div className="flex justify-center py-16 text-slate-400">
            <Spinner />
          </div>
        ) : documents.error && !page ? (
          <div className="p-5">
            <Alert
              action={
                <Button variant="secondary" size="sm" onClick={documents.reload}>
                  Retry
                </Button>
              }
            >
              {errorMessage(documents.error)}
            </Alert>
          </div>
        ) : page && page.items.length > 0 ? (
          <>
            <DocumentTable
              documents={page.items}
              busyId={busyId}
              onDelete={setPendingDelete}
              onReprocess={(document) => void reprocess(document)}
              onDownload={(document) => void download(document)}
            />
            <Pagination
              total={page.total}
              limit={page.limit}
              offset={page.offset}
              onChange={setOffset}
            />
          </>
        ) : (
          <EmptyState
            icon={<DocumentIcon />}
            title={status ? "No document with this status" : "No document yet"}
            description={
              status
                ? "Try another filter."
                : "Upload your first document: it will be searchable in a few moments."
            }
            action={
              status ? undefined : (
                <UploadButton onUploaded={documents.reload} variant="secondary" />
              )
            }
          />
        )}
      </Card>

      <Modal
        open={pendingDelete !== null}
        title="Delete this document?"
        onClose={() => setPendingDelete(null)}
        footer={
          <>
            <Button variant="secondary" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              loading={busyId !== null && busyId === pendingDelete?.id}
              onClick={() => void confirmDelete()}
            >
              Delete
            </Button>
          </>
        }
      >
        <p>
          <span className="font-medium text-slate-900">{pendingDelete?.filename}</span> and its
          indexed passages will be permanently removed. Past answers keep the excerpts they cited.
        </p>
      </Modal>
    </PageContainer>
  );
}
