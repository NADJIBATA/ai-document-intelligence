import { useEffect, useState, type FormEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import {
  compareDocuments,
  deleteComparison,
  getComparison,
  listComparisons,
} from "../api/comparisons";
import { listDocuments } from "../api/documents";
import { ASPECTS, type Aspect, type ComparisonSummary, type PassageRef } from "../api/types";
import { SourcePanel } from "../components/chat/SourcePanel";
import { ComparisonView } from "../components/compare/ComparisonView";
import { StatusBadge } from "../components/documents/StatusBadge";
import { PageContainer, PageHeader } from "../components/layout/PageHeader";
import { Alert } from "../components/ui/Alert";
import { Button } from "../components/ui/Button";
import { Card, CardHeader } from "../components/ui/Card";
import { EmptyState } from "../components/ui/EmptyState";
import { CompareIcon, TrashIcon } from "../components/ui/Icons";
import { Spinner } from "../components/ui/Spinner";
import { useToast } from "../components/ui/toast-context";
import { useApi } from "../hooks/useApi";
import { usePolling } from "../hooks/usePolling";
import { errorMessage, isNotFound } from "../lib/errors";
import { classNames, formatDateTime, humanize } from "../lib/format";

const ASPECT_LABELS: Record<Aspect, string> = {
  pricing: "Prices",
  dates: "Dates",
  conditions: "Conditions",
  clauses: "Clauses",
  deadlines: "Deadlines",
  obligations: "Obligations",
};

export function ComparePage() {
  const { comparisonId } = useParams();
  const navigate = useNavigate();
  const { notify } = useToast();

  const [documentA, setDocumentA] = useState("");
  const [documentB, setDocumentB] = useState("");
  const [aspects, setAspects] = useState<Aspect[]>([...ASPECTS]);
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [passage, setPassage] = useState<PassageRef | null>(null);

  const documents = useApi(
    (signal) => listDocuments({ status: "indexed", limit: 100 }, signal),
    [],
  );
  const history = useApi((signal) => listComparisons({ limit: 20 }, signal), []);
  const comparison = useApi(
    comparisonId ? (signal) => getComparison(comparisonId, signal) : null,
    [comparisonId],
  );

  const running =
    comparison.data?.status === "pending" || comparison.data?.status === "processing";
  usePolling(comparison.reload, 2000, running);

  // When the comparison on screen finishes, the history shows its final status too.
  const shownStatus = comparison.data?.status;
  const reloadHistory = history.reload;
  useEffect(() => {
    if (shownStatus === "completed" || shownStatus === "failed") {
      reloadHistory();
    }
  }, [shownStatus, reloadHistory]);

  // Opening a past comparison pre-fills the form with its documents.
  const shown = comparison.data;
  useEffect(() => {
    if (shown) {
      setDocumentA(shown.document_a.id ?? "");
      setDocumentB(shown.document_b.id ?? "");
      setPassage(null);
    }
  }, [shown?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const options = documents.data?.items ?? [];
  const sameDocument = documentA !== "" && documentA === documentB;
  const canSubmit = documentA !== "" && documentB !== "" && !sameDocument && aspects.length > 0;

  function toggleAspect(aspect: Aspect): void {
    setAspects((current) =>
      current.includes(aspect)
        ? current.filter((item) => item !== aspect)
        : ASPECTS.filter((item) => item === aspect || current.includes(item)),
    );
  }

  async function onSubmit(event: FormEvent): Promise<void> {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }
    setSubmitting(true);
    setFormError(null);
    try {
      const created = await compareDocuments(documentA, documentB, aspects);
      history.reload();
      navigate(`/compare/${created.id}`);
    } catch (error) {
      setFormError(errorMessage(error));
    } finally {
      setSubmitting(false);
    }
  }

  async function remove(item: ComparisonSummary): Promise<void> {
    try {
      await deleteComparison(item.id);
      if (item.id === comparisonId) {
        navigate("/compare", { replace: true });
      }
      history.reload();
    } catch (error) {
      notify(errorMessage(error), "error");
    }
  }

  function documentSelect(
    id: string,
    label: string,
    value: string,
    onChange: (value: string) => void,
  ) {
    return (
      <div>
        <label htmlFor={id} className="field-label">
          {label}
        </label>
        <select
          id={id}
          className="field-input"
          value={value}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">Select a document…</option>
          {options.map((item) => (
            <option key={item.id} value={item.id}>
              {item.filename}
            </option>
          ))}
        </select>
      </div>
    );
  }

  return (
    <PageContainer>
      <PageHeader
        title="Compare Documents"
        description="Find what changed between two documents. Every difference links to the passages that support it."
      />

      <Card className="mb-6">
        <form onSubmit={(event) => void onSubmit(event)} className="space-y-5 p-5">
          {documents.data && options.length < 2 ? (
            <Alert tone="info">
              You need at least two indexed documents to run a comparison.{" "}
              <Link to="/documents" className="font-medium underline">
                Upload documents
              </Link>
            </Alert>
          ) : null}
          <div className="grid gap-4 md:grid-cols-2">
            {documentSelect("compare-document-a", "Document A", documentA, setDocumentA)}
            {documentSelect("compare-document-b", "Document B", documentB, setDocumentB)}
          </div>
          {sameDocument ? (
            <p className="text-sm text-red-600">Select two different documents.</p>
          ) : null}

          <fieldset>
            <legend className="field-label">Aspects to compare</legend>
            <div className="flex flex-wrap gap-2">
              {ASPECTS.map((aspect) => {
                const checked = aspects.includes(aspect);
                return (
                  <label
                    key={aspect}
                    className={classNames(
                      "flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-1.5 text-sm",
                      checked
                        ? "border-brand-300 bg-brand-50 text-brand-900"
                        : "border-slate-200 bg-white text-slate-600",
                    )}
                  >
                    <input
                      type="checkbox"
                      className="h-4 w-4 accent-brand-600"
                      checked={checked}
                      onChange={() => toggleAspect(aspect)}
                    />
                    {ASPECT_LABELS[aspect]}
                  </label>
                );
              })}
            </div>
          </fieldset>

          {formError ? <Alert>{formError}</Alert> : null}

          <Button
            type="submit"
            loading={submitting}
            disabled={!canSubmit}
            icon={<CompareIcon width={18} height={18} />}
          >
            Compare
          </Button>
        </form>
      </Card>

      {comparisonId ? (
        <section className="mb-6" aria-live="polite">
          {comparison.loading ? (
            <div className="flex justify-center py-12 text-slate-400">
              <Spinner />
            </div>
          ) : comparison.error && !shown ? (
            <Alert>
              {isNotFound(comparison.error)
                ? "This comparison does not exist any more."
                : errorMessage(comparison.error)}
            </Alert>
          ) : shown ? (
            <>
              <div className="mb-4 flex flex-wrap items-center gap-3">
                <h2 className="text-lg font-semibold text-slate-900">
                  {shown.document_a.name} <span className="font-normal text-slate-400">vs</span>{" "}
                  {shown.document_b.name}
                </h2>
                <StatusBadge status={shown.status} />
              </div>
              {running ? (
                <Card>
                  <EmptyState
                    icon={<Spinner />}
                    title="Comparing the documents…"
                    description="The relevant passages of both documents are being read. This usually takes a few seconds."
                  />
                </Card>
              ) : shown.status === "failed" ? (
                <Alert>{shown.error_message ?? "The comparison failed."}</Alert>
              ) : (
                <ComparisonView comparison={shown} onOpenPassage={setPassage} />
              )}
            </>
          ) : null}
        </section>
      ) : null}

      <Card>
        <CardHeader title="History" description="Your previous comparisons" />
        {history.loading ? (
          <div className="flex justify-center py-10 text-slate-400">
            <Spinner />
          </div>
        ) : history.error && !history.data ? (
          <div className="p-5">
            <Alert>{errorMessage(history.error)}</Alert>
          </div>
        ) : history.data && history.data.items.length > 0 ? (
          <ul className="divide-y divide-slate-100">
            {history.data.items.map((item) => (
              <li
                key={item.id}
                className={classNames(
                  "flex items-center justify-between gap-3 px-5 py-3",
                  item.id === comparisonId && "bg-brand-50/50",
                )}
              >
                <Link to={`/compare/${item.id}`} className="min-w-0 flex-1 hover:underline">
                  <span className="block truncate text-sm font-medium text-slate-900">
                    {item.document_a.name} vs {item.document_b.name}
                  </span>
                  <span className="block truncate text-xs text-slate-500">
                    {formatDateTime(item.created_at)} · {item.aspects.map(humanize).join(", ")}
                  </span>
                </Link>
                <StatusBadge status={item.status} />
                <button
                  type="button"
                  onClick={() => void remove(item)}
                  className="rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-red-600"
                  aria-label={`Delete comparison ${item.document_a.name} vs ${item.document_b.name}`}
                >
                  <TrashIcon width={16} height={16} />
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <EmptyState
            icon={<CompareIcon />}
            title="No comparison yet"
            description="Select two documents above to see what differs between them."
          />
        )}
      </Card>

      <SourcePanel passage={passage} onClose={() => setPassage(null)} />
    </PageContainer>
  );
}
