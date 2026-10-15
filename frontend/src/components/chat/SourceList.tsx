import type { Source } from "../../api/types";
import { classNames, pageLabel } from "../../lib/format";
import { DocumentIcon } from "../ui/Icons";

interface SourceListProps {
  sources: Source[];
  onSelect: (source: Source) => void;
  /** chunk_id of the source currently displayed in the side panel. */
  selectedChunkId?: string | null;
}

/** "Sources" block displayed under an answer: document, page, section. */
export function SourceList({ sources, onSelect, selectedChunkId }: SourceListProps) {
  if (sources.length === 0) {
    return null;
  }
  return (
    <div className="mt-3 border-t border-slate-200 pt-3">
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Sources</p>
      <ul className="flex flex-wrap gap-2">
        {sources.map((source) => {
          const selected = source.chunk_id === selectedChunkId;
          return (
            <li key={`${source.ref}-${source.chunk_id}`}>
              <button
                type="button"
                onClick={() => onSelect(source)}
                className={classNames(
                  "flex max-w-[16rem] items-start gap-2 rounded-lg border px-2.5 py-1.5 text-left text-xs transition-colors",
                  selected
                    ? "border-brand-400 bg-brand-50"
                    : "border-slate-200 bg-white hover:border-brand-300 hover:bg-brand-50/50",
                )}
                title={source.excerpt}
              >
                <span className="mt-0.5 inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded bg-brand-100 px-1 text-[11px] font-semibold text-brand-700">
                  {source.ref.slice(1)}
                </span>
                <span className="min-w-0">
                  <span className="flex items-center gap-1 font-medium text-slate-800">
                    <DocumentIcon width={14} height={14} className="shrink-0 text-slate-400" />
                    <span className="truncate">{source.document}</span>
                  </span>
                  <span className="block text-slate-500">
                    {[pageLabel(source.page), source.section].filter(Boolean).join(" · ") ||
                      "Passage"}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
