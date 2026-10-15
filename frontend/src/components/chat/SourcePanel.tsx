import { useState } from "react";

import { getChunk } from "../../api/documents";
import type { PassageRef } from "../../api/types";
import { useApi } from "../../hooks/useApi";
import { errorMessage, isNotFound } from "../../lib/errors";
import { openDocumentFile } from "../../lib/files";
import { classNames, pageLabel } from "../../lib/format";
import { Alert } from "../ui/Alert";
import { Button } from "../ui/Button";
import { CloseIcon, DocumentIcon, ExternalLinkIcon } from "../ui/Icons";
import { Spinner } from "../ui/Spinner";
import { useToast } from "../ui/toast-context";

interface SourcePanelProps {
  passage: PassageRef | null;
  onClose: () => void;
  /** Docked next to the content on wide screens (chat); always an overlay otherwise. */
  docked?: boolean;
}

/** Side panel showing the full passage behind a citation, with a link to the file. */
export function SourcePanel({ passage, onClose, docked = false }: SourcePanelProps) {
  const { notify } = useToast();
  const [opening, setOpening] = useState(false);
  const chunk = useApi(
    passage ? (signal) => getChunk(passage.document_id, passage.chunk_id, signal) : null,
    [passage?.document_id, passage?.chunk_id],
  );

  if (passage === null) {
    return null;
  }
  const deleted = isNotFound(chunk.error);
  const page = chunk.data?.page ?? passage.page;
  const section = chunk.data?.section ?? passage.section;

  async function openFile(): Promise<void> {
    if (passage === null) {
      return;
    }
    setOpening(true);
    try {
      await openDocumentFile(passage.document_id, page);
    } catch (error) {
      notify(errorMessage(error), "error");
    } finally {
      setOpening(false);
    }
  }

  return (
    <>
      <div
        className={classNames(
          "fixed inset-0 z-30 bg-slate-900/30",
          docked && "min-[1400px]:hidden",
        )}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        className={classNames(
          "fixed inset-y-0 right-0 z-40 flex w-full max-w-md flex-col border-l border-slate-200 bg-white shadow-xl",
          docked &&
            "min-[1400px]:static min-[1400px]:z-auto min-[1400px]:w-96 min-[1400px]:shrink-0 min-[1400px]:shadow-none",
        )}
        aria-label="Source passage"
      >
        <div className="flex items-start justify-between gap-3 border-b border-slate-100 px-5 py-4">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Source{passage.ref ? ` ${passage.ref.replace(/^S/, "")}` : ""}
            </p>
            <p className="mt-1 flex items-center gap-1.5 text-sm font-semibold text-slate-900">
              <DocumentIcon width={16} height={16} className="shrink-0 text-slate-400" />
              <span className="truncate" title={passage.document}>
                {passage.document}
              </span>
            </p>
            <p className="mt-0.5 text-sm text-slate-500">
              {[pageLabel(page), section].filter(Boolean).join(" · ") || "Passage"}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
            aria-label="Close source"
          >
            <CloseIcon />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          {chunk.loading ? (
            <Spinner showLabel label="Loading passage" className="text-slate-500" />
          ) : deleted ? (
            <>
              <Alert tone="warning" className="mb-4">
                This document has been deleted. Only the excerpt saved with the answer is
                available.
              </Alert>
              {passage.excerpt ? (
                <blockquote className="border-l-4 border-slate-200 pl-4 text-sm leading-relaxed text-slate-700">
                  {passage.excerpt}
                </blockquote>
              ) : null}
            </>
          ) : chunk.error ? (
            <Alert>{errorMessage(chunk.error)}</Alert>
          ) : (
            <blockquote className="whitespace-pre-wrap border-l-4 border-brand-200 pl-4 text-sm leading-relaxed text-slate-700">
              {chunk.data?.text}
            </blockquote>
          )}
        </div>

        {!deleted ? (
          <div className="border-t border-slate-100 px-5 py-3">
            <Button
              variant="secondary"
              loading={opening}
              icon={<ExternalLinkIcon width={16} height={16} />}
              onClick={() => void openFile()}
            >
              Open document{page ? ` at page ${page}` : ""}
            </Button>
          </div>
        ) : null}
      </aside>
    </>
  );
}
