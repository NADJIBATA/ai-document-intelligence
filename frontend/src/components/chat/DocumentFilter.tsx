import { useEffect, useRef, useState } from "react";

import type { DocumentItem } from "../../api/types";
import { classNames } from "../../lib/format";
import { DocumentIcon } from "../ui/Icons";

interface DocumentFilterProps {
  documents: DocumentItem[];
  selectedIds: string[];
  onChange: (ids: string[]) => void;
}

/** Restrict the answers to some documents (all indexed documents by default). */
export function DocumentFilter({ documents, selectedIds, onChange }: DocumentFilterProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    function onPointerDown(event: MouseEvent): void {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function onKeyDown(event: KeyboardEvent): void {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const selected = new Set(selectedIds);
  const label =
    selected.size === 0
      ? "All documents"
      : selected.size === 1
        ? "1 document"
        : `${selected.size} documents`;

  function toggle(id: string): void {
    const next = new Set(selected);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    onChange(Array.from(next));
  }

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-haspopup="true"
        aria-expanded={open}
        className={classNames(
          "inline-flex items-center gap-1.5 whitespace-nowrap rounded-lg border px-3 py-1.5 text-xs font-medium",
          selected.size > 0
            ? "border-brand-300 bg-brand-50 text-brand-800"
            : "border-slate-300 bg-white text-slate-700 hover:bg-slate-50",
        )}
      >
        <DocumentIcon width={14} height={14} />
        {label}
      </button>
      {open ? (
        <div className="absolute right-0 z-20 mt-2 w-72 rounded-xl border border-slate-200 bg-white p-2 shadow-lg">
          <div className="flex items-center justify-between px-2 py-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Search in
            </p>
            {selected.size > 0 ? (
              <button
                type="button"
                className="text-xs font-medium text-brand-700 hover:underline"
                onClick={() => onChange([])}
              >
                All documents
              </button>
            ) : null}
          </div>
          {documents.length === 0 ? (
            <p className="px-2 py-3 text-sm text-slate-500">No indexed document yet.</p>
          ) : (
            <ul className="max-h-64 overflow-y-auto">
              {documents.map((document) => (
                <li key={document.id}>
                  <label className="flex cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-slate-700 hover:bg-slate-50">
                    <input
                      type="checkbox"
                      className="h-4 w-4 accent-brand-600"
                      checked={selected.has(document.id)}
                      onChange={() => toggle(document.id)}
                    />
                    <span className="truncate" title={document.filename}>
                      {document.filename}
                    </span>
                  </label>
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : null}
    </div>
  );
}
