import type { ReactNode } from "react";

import type { Entity, EntityType, Insights, PassageRef } from "../../api/types";
import { formatAmount, formatDate, formatDateTime, humanize, pageLabel } from "../../lib/format";
import { Alert } from "../ui/Alert";
import { Badge } from "../ui/Badge";
import { Card, CardHeader } from "../ui/Card";

interface InsightsViewProps {
  insights: Insights;
  documentName: string;
  /** Open the passage where a date or an amount was found. */
  onOpenPassage: (passage: PassageRef) => void;
}

const ENTITY_GROUPS: Array<{ type: EntityType; label: string }> = [
  { type: "organization", label: "Organizations" },
  { type: "person", label: "People" },
  { type: "location", label: "Locations" },
  { type: "other", label: "Other" },
];

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="mt-1 text-sm text-slate-900">{children}</dd>
    </div>
  );
}

function PageLink({
  page,
  chunkId,
  onClick,
}: {
  page: number | null;
  chunkId: string | null;
  onClick: () => void;
}) {
  if (chunkId === null) {
    return <span className="text-slate-400">—</span>;
  }
  return (
    <button
      type="button"
      onClick={onClick}
      className="font-medium text-brand-700 hover:underline"
      title="Show the passage"
    >
      {pageLabel(page) ?? "Passage"}
    </button>
  );
}

export function InsightsView({ insights, documentName, onOpenPassage }: InsightsViewProps) {
  const entitiesByType = (type: EntityType): Entity[] =>
    insights.entities.filter((entity) => entity.type === type);

  const passage = (chunkId: string, page: number | null, excerpt: string): PassageRef => ({
    document_id: insights.document_id,
    chunk_id: chunkId,
    document: documentName,
    page,
    section: null,
    excerpt,
  });

  return (
    <div className="space-y-6">
      {insights.truncated ? (
        <Alert tone="info">
          This document is long: only representative parts of it were analysed.
        </Alert>
      ) : null}

      <Card>
        <CardHeader title="Overview" />
        <dl className="grid gap-5 px-5 py-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Title">{insights.title ?? "—"}</Field>
          <Field label="Document type">
            {insights.document_type ? (
              <Badge tone="blue">{humanize(insights.document_type)}</Badge>
            ) : (
              "—"
            )}
          </Field>
          <Field label="Author">
            {insights.authors.length > 0 ? insights.authors.join(", ") : "—"}
          </Field>
          <Field label="Language">{insights.language?.toUpperCase() ?? "—"}</Field>
        </dl>
      </Card>

      <Card>
        <CardHeader title="Summary" />
        <div className="space-y-4 px-5 py-4">
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-700">
            {insights.summary || "No summary was produced."}
          </p>
          {insights.keywords.length > 0 ? (
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                Keywords
              </p>
              <div className="flex flex-wrap gap-1.5">
                {insights.keywords.map((keyword) => (
                  <Badge key={keyword}>{keyword}</Badge>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </Card>

      {insights.key_points.length > 0 ? (
        <Card>
          <CardHeader title="Key information" />
          <ul className="list-disc space-y-1.5 px-5 py-4 pl-10 text-sm text-slate-700">
            {insights.key_points.map((point) => (
              <li key={point}>{point}</li>
            ))}
          </ul>
        </Card>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="People and organizations" />
          <div className="space-y-4 px-5 py-4">
            {insights.entities.length === 0 ? (
              <p className="text-sm text-slate-500">None found.</p>
            ) : (
              ENTITY_GROUPS.map(({ type, label }) => {
                const entities = entitiesByType(type);
                if (entities.length === 0) {
                  return null;
                }
                return (
                  <div key={type}>
                    <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                      {label}
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {entities.map((entity) => (
                        <Badge key={entity.name} tone={type === "person" ? "violet" : "neutral"}>
                          {entity.name}
                        </Badge>
                      ))}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </Card>

        <Card>
          <CardHeader title="Amounts" />
          {insights.monetary_amounts.length === 0 ? (
            <p className="px-5 py-4 text-sm text-slate-500">None found.</p>
          ) : (
            <ul className="divide-y divide-slate-100">
              {insights.monetary_amounts.map((amount) => (
                <li
                  key={amount.text}
                  className="flex items-start justify-between gap-4 px-5 py-3 text-sm"
                >
                  <div className="min-w-0">
                    <p className="font-semibold text-slate-900">
                      {formatAmount(amount.value, amount.currency) ?? amount.text}
                    </p>
                    <p className="text-slate-600">{amount.description || amount.text}</p>
                  </div>
                  <PageLink
                    page={amount.page}
                    chunkId={amount.chunk_id}
                    onClick={() =>
                      amount.chunk_id &&
                      onOpenPassage(passage(amount.chunk_id, amount.page, amount.text))
                    }
                  />
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <Card>
        <CardHeader title="Important dates" />
        {insights.important_dates.length === 0 ? (
          <p className="px-5 py-4 text-sm text-slate-500">None found.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200">
              <thead className="bg-slate-50">
                <tr>
                  <th scope="col" className="table-header">
                    Date
                  </th>
                  <th scope="col" className="table-header">
                    As written
                  </th>
                  <th scope="col" className="table-header">
                    Description
                  </th>
                  <th scope="col" className="table-header">
                    Source
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {insights.important_dates.map((item) => (
                  <tr key={`${item.date ?? ""}-${item.text}`}>
                    <td className="table-cell whitespace-nowrap font-medium text-slate-900">
                      {item.date ? formatDate(item.date) : "—"}
                    </td>
                    <td className="table-cell">{item.text}</td>
                    <td className="table-cell">{item.description || "—"}</td>
                    <td className="table-cell whitespace-nowrap">
                      <PageLink
                        page={item.page}
                        chunkId={item.chunk_id}
                        onClick={() =>
                          item.chunk_id &&
                          onOpenPassage(passage(item.chunk_id, item.page, item.text))
                        }
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <p className="text-xs text-slate-400">
        Extracted {insights.model ? `by ${insights.model} ` : ""}on{" "}
        {formatDateTime(insights.updated_at)}. Generated by AI: check important facts against the
        document.
      </p>
    </div>
  );
}
