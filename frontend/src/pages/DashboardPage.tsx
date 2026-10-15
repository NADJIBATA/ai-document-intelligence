import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import { getDashboardStats } from "../api/dashboard";
import { useAuth } from "../auth/context";
import { StatusBadge } from "../components/documents/StatusBadge";
import { UploadButton } from "../components/documents/UploadButton";
import { PageContainer, PageHeader } from "../components/layout/PageHeader";
import { Alert } from "../components/ui/Alert";
import { Card, CardHeader } from "../components/ui/Card";
import { EmptyState } from "../components/ui/EmptyState";
import {
  ArrowRightIcon,
  ChatIcon,
  CheckIcon,
  CompareIcon,
  DocumentIcon,
  RefreshIcon,
} from "../components/ui/Icons";
import { Spinner } from "../components/ui/Spinner";
import { useApi } from "../hooks/useApi";
import { useDocuments } from "../hooks/useDocuments";
import { usePolling } from "../hooks/usePolling";
import { errorMessage } from "../lib/errors";
import { classNames, formatBytes, formatDateTime, formatNumber } from "../lib/format";

interface StatCardProps {
  label: string;
  value: number | undefined;
  icon: ReactNode;
  tone: string;
  hint?: string;
  to: string;
}

function StatCard({ label, value, icon, tone, hint, to }: StatCardProps) {
  return (
    <Link to={to} className="group block">
      <Card className="p-5 transition-shadow group-hover:shadow-md">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm font-medium text-slate-500">{label}</p>
            <p className="mt-2 text-3xl font-semibold tracking-tight text-slate-900">
              {value === undefined ? "–" : formatNumber(value)}
            </p>
          </div>
          <div className={classNames("flex h-10 w-10 items-center justify-center rounded-lg", tone)}>
            {icon}
          </div>
        </div>
        {hint ? <p className="mt-2 text-xs text-slate-500">{hint}</p> : null}
      </Card>
    </Link>
  );
}

export function DashboardPage() {
  const { user } = useAuth();
  const stats = useApi((signal) => getDashboardStats(signal), []);
  const recent = useDocuments({ limit: 5 });

  const documents = stats.data?.documents;
  const inProgress = documents ? documents.uploaded + documents.processing : undefined;
  // Counters follow the processing of documents without reloading the page.
  usePolling(stats.reload, 3000, (inProgress ?? 0) > 0);

  function refresh(): void {
    stats.reload();
    recent.reload();
  }

  return (
    <PageContainer>
      <PageHeader
        title="Dashboard"
        description={`Welcome${user?.full_name ? `, ${user.full_name}` : ""}. Here is the state of your workspace.`}
        actions={<UploadButton onUploaded={refresh} />}
      />

      {stats.error ? <Alert className="mb-6">{errorMessage(stats.error)}</Alert> : null}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Documents"
          value={documents?.total}
          icon={<DocumentIcon />}
          tone="bg-brand-50 text-brand-700"
          hint={
            stats.data
              ? `${formatBytes(stats.data.storage_bytes)} stored${
                  documents && documents.failed > 0 ? ` · ${documents.failed} failed` : ""
                }`
              : undefined
          }
          to="/documents"
        />
        <StatCard
          label="Documents indexed"
          value={documents?.indexed}
          icon={<CheckIcon />}
          tone="bg-emerald-50 text-emerald-700"
          hint={
            stats.data ? `${formatNumber(stats.data.passages)} searchable passages` : undefined
          }
          to="/documents"
        />
        <StatCard
          label="Documents processing"
          value={inProgress}
          icon={<RefreshIcon />}
          tone="bg-amber-50 text-amber-700"
          hint={inProgress ? "Indexing in progress…" : "Nothing in the queue"}
          to="/documents"
        />
        <StatCard
          label="Conversations"
          value={stats.data?.conversations}
          icon={<ChatIcon />}
          tone="bg-violet-50 text-violet-700"
          hint={
            stats.data
              ? `${stats.data.comparisons} comparison${stats.data.comparisons === 1 ? "" : "s"}`
              : undefined
          }
          to="/chat"
        />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Recent documents"
            action={
              <Link
                to="/documents"
                className="inline-flex items-center gap-1 text-sm font-medium text-brand-700 hover:underline"
              >
                View all <ArrowRightIcon width={16} height={16} />
              </Link>
            }
          />
          {recent.loading ? (
            <div className="flex justify-center py-10 text-slate-400">
              <Spinner />
            </div>
          ) : recent.error ? (
            <div className="p-5">
              <Alert>{errorMessage(recent.error)}</Alert>
            </div>
          ) : recent.data && recent.data.items.length > 0 ? (
            <ul className="divide-y divide-slate-100">
              {recent.data.items.map((document) => (
                <li key={document.id} className="flex items-center justify-between gap-4 px-5 py-3">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-slate-900">
                      {document.filename}
                    </p>
                    <p className="text-xs text-slate-500">
                      {formatDateTime(document.created_at)} · {formatBytes(document.size_bytes)}
                    </p>
                  </div>
                  <StatusBadge status={document.status} />
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState
              icon={<DocumentIcon />}
              title="No document yet"
              description="Upload a PDF, DOCX or TXT file to start asking questions."
              action={<UploadButton onUploaded={refresh} variant="secondary" />}
            />
          )}
        </Card>

        <Card>
          <CardHeader title="Get started" />
          <ul className="divide-y divide-slate-100">
            {[
              {
                to: "/chat",
                icon: <ChatIcon />,
                title: "Ask a question",
                text: "Answers cite the exact passages they come from.",
              },
              {
                to: "/intelligence",
                icon: <DocumentIcon />,
                title: "Document Intelligence",
                text: "Summary, parties, dates and amounts of a document.",
              },
              {
                to: "/compare",
                icon: <CompareIcon />,
                title: "Compare two documents",
                text: "Prices, dates, clauses and obligations side by side.",
              },
            ].map((item) => (
              <li key={item.to}>
                <Link to={item.to} className="flex items-start gap-3 px-5 py-4 hover:bg-slate-50">
                  <span className="mt-0.5 text-slate-400">{item.icon}</span>
                  <span>
                    <span className="block text-sm font-medium text-slate-900">{item.title}</span>
                    <span className="block text-sm text-slate-500">{item.text}</span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </PageContainer>
  );
}
