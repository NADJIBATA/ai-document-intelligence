import { Fragment, useMemo, type ReactNode } from "react";

import { type Align, type Block, type Inline, parseMarkdown } from "../../lib/markdown";

interface AnswerTextProps {
  text: string;
  /** Markers that have a matching source (the others are shown as plain text). */
  knownRefs: ReadonlySet<string>;
  onCitationClick: (ref: string) => void;
}

interface RenderContext {
  knownRefs: ReadonlySet<string>;
  onCitationClick: (ref: string) => void;
}

/**
 * Answer of the assistant: Markdown (paragraphs, lists, tables, emphasis...) rendered
 * as React elements, with [S1] markers turned into clickable chips. No HTML from the
 * answer is ever injected (see lib/markdown.ts).
 */
export function AnswerText({ text, knownRefs, onCitationClick }: AnswerTextProps) {
  const blocks = useMemo(() => parseMarkdown(text), [text]);
  const context: RenderContext = { knownRefs, onCitationClick };
  return (
    <div className="space-y-3 break-words leading-relaxed" data-answer="">
      {renderBlocks(blocks, context)}
    </div>
  );
}

function renderBlocks(blocks: Block[], context: RenderContext): ReactNode {
  return blocks.map((block, index) => <Fragment key={index}>{renderBlock(block, context)}</Fragment>);
}

function renderBlock(block: Block, context: RenderContext): ReactNode {
  switch (block.type) {
    case "paragraph":
      return <p>{renderInline(block.content, context)}</p>;
    case "heading":
      return block.level <= 2 ? (
        <h3 className="pt-1 text-[15px] font-semibold text-slate-900">
          {renderInline(block.content, context)}
        </h3>
      ) : (
        <h4 className="pt-1 font-semibold text-slate-900">{renderInline(block.content, context)}</h4>
      );
    case "list": {
      const items = block.items.map((item, index) => (
        <li key={index} className="pl-1 marker:text-slate-400">
          {renderInline(item.content, context)}
          {item.children.length > 0 ? (
            <div className="mt-1.5 space-y-1.5">{renderBlocks(item.children, context)}</div>
          ) : null}
        </li>
      ));
      return block.ordered ? (
        <ol start={block.start} className="list-decimal space-y-1.5 pl-5 marker:font-medium">
          {items}
        </ol>
      ) : (
        <ul className="list-disc space-y-1.5 pl-5">{items}</ul>
      );
    }
    case "table":
      return (
        <div className="overflow-x-auto rounded-lg border border-slate-200">
          <table className="min-w-full border-collapse text-left text-[13px]">
            <thead className="bg-slate-50">
              <tr>
                {block.header.map((cell, column) => (
                  <th
                    key={column}
                    scope="col"
                    className={`border-b border-slate-200 px-3 py-2 font-semibold text-slate-700 ${alignClass(block.align[column] ?? null)}`}
                  >
                    {renderInline(cell, context)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, rowIndex) => (
                <tr key={rowIndex} className="odd:bg-white even:bg-slate-50/60">
                  {row.map((cell, column) => (
                    <td
                      key={column}
                      className={`border-t border-slate-100 px-3 py-2 align-top ${alignClass(block.align[column] ?? null)}`}
                    >
                      {renderInline(cell, context)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case "quote":
      return (
        <blockquote className="space-y-2 border-l-4 border-slate-200 pl-3 text-slate-600">
          {renderBlocks(block.children, context)}
        </blockquote>
      );
    case "code":
      return (
        <pre className="overflow-x-auto rounded-lg bg-slate-900 px-3 py-2.5 text-[12px] leading-relaxed text-slate-100">
          <code>{block.value}</code>
        </pre>
      );
    case "rule":
      return <hr className="border-slate-200" />;
  }
}

// Classes rather than inline styles: the Content-Security-Policy forbids inline styles.
function alignClass(align: Align): string {
  if (align === "center") return "text-center";
  if (align === "right") return "text-right";
  return "text-left";
}

function renderInline(nodes: Inline[], context: RenderContext): ReactNode {
  return nodes.map((node, index) => {
    switch (node.type) {
      case "text":
        return <Fragment key={index}>{node.value}</Fragment>;
      case "break":
        return <br key={index} />;
      case "code":
        return (
          <code key={index} className="rounded bg-slate-100 px-1 py-0.5 text-[12px] text-slate-800">
            {node.value}
          </code>
        );
      case "strong":
        return (
          <strong key={index} className="font-semibold text-slate-900">
            {renderInline(node.children, context)}
          </strong>
        );
      case "em":
        return <em key={index}>{renderInline(node.children, context)}</em>;
      case "del":
        return <del key={index}>{renderInline(node.children, context)}</del>;
      case "citation":
        return <CitationChip key={index} citeRef={node.ref} context={context} />;
    }
  });
}

function CitationChip({ citeRef, context }: { citeRef: string; context: RenderContext }) {
  if (!context.knownRefs.has(citeRef)) {
    return <span>[{citeRef}]</span>;
  }
  const number = citeRef.slice(1);
  return (
    <button
      type="button"
      onClick={() => context.onCitationClick(citeRef)}
      className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-brand-100 px-1 align-text-top text-[11px] font-semibold text-brand-700 hover:bg-brand-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
      title={`Show source ${number}`}
      aria-label={`Source ${number}`}
    >
      {number}
    </button>
  );
}
