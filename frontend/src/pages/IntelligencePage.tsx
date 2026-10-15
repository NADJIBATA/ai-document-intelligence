import { useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError } from "../api/client";
import { getDocument, getInsights, listDocuments, regenerateInsights } from "../api/documents";
import type { PassageRef } from "../api/types";
import { SourcePanel } from "../components/chat/SourcePanel";
import { StatusBadge } from "../components/documents/StatusBadge";
import { InsightsView } from "../components/intelligence/InsightsView";
import { PageContainer, PageHeader } from "../components/layout/PageHeader";
import { Alert } from "../components/ui/Alert";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { EmptyState } from "../components/ui/EmptyState";
import { RefreshIcon, SparklesIcon } from "../components/ui/Icons";
import { Spinner } from "../components/ui/Spinner";
import { useToast } from "../components/ui/toast-context";
import { useApi } from "../hooks/useApi";
import { useDocumentStatus } from "../hooks/useDocumentStatus";
import { usePolling } from "../hooks/usePolling";
import { errorMessage, isNotFound } from "../lib/errors";
import { formatBytes } from "../lib/format";

function isMissingInsights(error: unknown): boolean {
  return error instanceof ApiError && error.code === "insights_not_found";
}

export function IntelligencePage() {
  const { documentId } = useParams();
  const navigate = useNavigate();
  const { notify } = useToast();
  const [passage, setPassage] = useState<PassageRef | null>(null);
  const [starting, setStarting] = useState(false);

  const indexedDocuments = useApi(
    (signal) => listDocuments({ status: "indexed", limit: 100 }, signal),
    [],
  );
  const documentInfo = useApi(
    documentId ? (signal) => getDocument(documentId, signal) : null,
    [documentId],
  );
  // A document opened right after its upload may still be processing.
  const liveStatus = useDocumentStatus(documentId ?? null);
  const status = liveStatus.data?.status ?? documentInfo.data?.status;
  const indexed = status === "indexed";

  const insights = useApi(
    documentId && indexed ? (signal) => getInsights(documentId, signal) : null,
    [documentId, indexed],
  );
  const extracting =
    insights.data?.status === "pending" || insights.data?.status === "processing";
  usePolling(insights.reload, 2000, extracting);

  async function extract(): Promise<void> {
    if (!documentId) {
      return;
    }
    setStarting(true);
    try {
      insights.setData(await regenerateInsights(documentId));
      insights.reload();
    } catch (error) {
      notify(errorMessage(error), "error");
    } finally {
      setStarting(false);
    }
  }

  const options = indexedDocuments.data?.items ?? [];
  const selector = (
    <div>
      <label htmlFor="intelligence-document" className="sr-only">
        Document
      </label>
      <select
        id="intelligence-document"
        className="field-input min-w-[16rem] max-w-xs"
        value={documentId ?? ""}
        onChange={(event) => {
          setPassage(null);
          navigate(event.target.value ? `/intelligence/${event.target.value}` : "/intelligence");
        }}
      >
        <option value="">Select a document…</option>
        {documentId && documentInfo.data && !options.some((item) => item.id === documentId) ? (
          <option value={documentId}>{documentInfo.data.filename}</option>
        ) : null}
        {options.map((item) => (
          <option key={item.id} value={item.id}>
            {item.filename}
          </option>
        ))}
      </select>
    </div>
  );

  let content: ReactNode;
  if (!documentId) {
    content = (
      <Card>
        {indexedDocuments.loading ? (
          <div className="flex justify-center py-16 text-slate-400">
            <Spinner />
          </div>
        ) : options.length === 0 ? (
          <EmptyState
            icon={<SparklesIcon />}
            title="No indexed document yet"
            description="Upload a document: its title, summary, parties, dates and amounts are extracted automatically."
            action={
              <Link
                to="/documents"
                className="inline-flex rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
              >
                Go to documents
              </Link>
            }
          />
        ) : (
          <ul className="divide-y divide-slate-100">
            {options.map((item) => (
              <li key={item.id}>
                <Link
                  to={`/intelligence/${item.id}`}
                  className="flex items-center justify-between gap-4 px-5 py-3 hover:bg-slate-50"
                >
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-medium text-slate-900">
                      {item.filename}
                    </span>
                    <span className="block text-xs text-slate-500">
                      {item.page_count ? `${item.page_count} pages · ` : ""}
                      {formatBytes(item.size_bytes)}
                    </span>
                  </span>
                  <span className="shrink-0 text-sm font-medium text-brand-700">View insights</span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </Card>
    );
  } else if (documentInfo.loading) {
    content = (
      <div className="flex justify-center py-16 text-slate-400">
        <Spinner />
      </div>
    );
  } else if (documentInfo.error) {
    content = (
      <Alert>
        {isNotFound(documentInfo.error)
          ? "This document does not exist any more."
          : errorMessage(documentInfo.error)}
      </Alert>
    );
  } else if (status === "failed") {
    content = (
      <Alert>
        This document could not be processed
        {documentInfo.data?.error_message ? `: ${documentInfo.data.error_message}` : "."} Retry it from
        the Documents page.
      </Alert>
    );
  } else if (!indexed) {
    content = (
      <Card>
        <EmptyState
          icon={<Spinner />}
          title="This document is being processed"
          description="Its insights will appear here as soon as it is indexed."
        />
      </Card>
    );
  } else if (insights.loading) {
    content = (
      <div className="flex justify-center py-16 text-slate-400">
        <Spinner />
      </div>
    );
  } else if (isMissingInsights(insights.error)) {
    content = (
      <Card>
        <EmptyState
          icon={<SparklesIcon />}
          title="No insights for this document yet"
          description="Extract its title, summary, parties, dates, amounts and key information."
          action={
            <Button loading={starting} onClick={() => void extract()}>
              Extract information
            </Button>
          }
        />
      </Card>
    );
  } else if (insights.error && !insights.data) {
    content = <Alert>{errorMessage(insights.error)}</Alert>;
  } else if (insights.data && extracting) {
    content = (
      <Card>
        <EmptyState
          icon={<Spinner />}
          title="Analysing the document…"
          description="The language model is reading the document. This usually takes a few seconds."
        />
      </Card>
    );
  } else if (insights.data?.status === "failed") {
    content = (
      <Alert
        action={
          <Button variant="secondary" size="sm" loading={starting} onClick={() => void extract()}>
            Try again
          </Button>
        }
      >
        {insights.data.error_message ?? "The analysis failed."}
      </Alert>
    );
  } else if (insights.data && documentInfo.data) {
    content = (
      <InsightsView
        insights={insights.data}
        documentName={documentInfo.data.filename}
        onOpenPassage={setPassage}
      />
    );
  } else {
    content = null;
  }

  return (
    <PageContainer>
      <PageHeader
        title="Document Intelligence"
        description="Structured information extracted from a document by the language model."
        actions={
          <>
            {selector}
            {documentId && insights.data?.status === "completed" ? (
              <Button
                variant="secondary"
                loading={starting}
                icon={<RefreshIcon width={16} height={16} />}
                onClick={() => void extract()}
              >
                Regenerate
              </Button>
            ) : null}
          </>
        }
      />

      {documentId && documentInfo.data ? (
        <div className="mb-4 flex flex-wrap items-center gap-3 text-sm text-slate-600">
          <span className="font-medium text-slate-900">{documentInfo.data.filename}</span>
          {status ? <StatusBadge status={status} /> : null}
          <span>{formatBytes(documentInfo.data.size_bytes)}</span>
          {documentInfo.data.page_count ? <span>{documentInfo.data.page_count} pages</span> : null}
        </div>
      ) : null}

      {content}

      <SourcePanel passage={passage} onClose={() => setPassage(null)} />
    </PageContainer>
  );
}
