import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import type { DocumentItem, Source } from "../api/types";
import { AnswerText } from "../components/chat/AnswerText";
import { SourceList } from "../components/chat/SourceList";
import { DocumentTable } from "../components/documents/DocumentTable";
import { StatusBadge } from "../components/documents/StatusBadge";

function source(overrides: Partial<Source> = {}): Source {
  return {
    ref: "S1",
    document_id: "doc-1",
    document: "contract.pdf",
    page: 14,
    section: "Termination",
    chunk_id: "chunk-1",
    excerpt: "Either party may terminate with sixty days notice.",
    score: 0.93,
    ...overrides,
  };
}

function documentItem(overrides: Partial<DocumentItem> = {}): DocumentItem {
  return {
    id: "doc-1",
    filename: "contract.pdf",
    content_type: "application/pdf",
    extension: "pdf",
    size_bytes: 1536,
    status: "indexed",
    error_message: null,
    page_count: 3,
    chunk_count: 7,
    created_at: "2025-03-31T10:00:00Z",
    updated_at: "2025-03-31T10:00:05Z",
    indexed_at: "2025-03-31T10:00:05Z",
    ...overrides,
  };
}

describe("AnswerText", () => {
  it("turns known markers into buttons and keeps unknown ones as text", async () => {
    const onCitationClick = vi.fn();
    render(
      <AnswerText
        text="The notice period is sixty days [S1]. See also [S7]."
        knownRefs={new Set(["S1"])}
        onCitationClick={onCitationClick}
      />,
    );

    expect(screen.getByText(/The notice period is sixty days/)).toBeInTheDocument();
    expect(screen.getByText("[S7]")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Source 1" }));
    expect(onCitationClick).toHaveBeenCalledWith("S1");
    expect(screen.queryByRole("button", { name: "Source 7" })).not.toBeInTheDocument();
  });

  it("renders Markdown lists, tables and emphasis with clickable citations", async () => {
    const onCitationClick = vi.fn();
    const { container } = render(
      <AnswerText
        text={[
          "Two obligations:",
          "",
          "- Pay within **45 days** [S1]",
          "- Report incidents [S2]",
          "",
          "| Aspect | Value |",
          "|---|---|",
          "| Notice | 90 days [S2] |",
        ].join("\n")}
        knownRefs={new Set(["S1", "S2"])}
        onCitationClick={onCitationClick}
      />,
    );

    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    expect(screen.getByText("45 days").tagName).toBe("STRONG");
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Aspect" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: /90 days/ })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Source 2" })).toHaveLength(2);
    await userEvent.click(screen.getAllByRole("button", { name: "Source 2" })[1] as HTMLElement);
    expect(onCitationClick).toHaveBeenCalledWith("S2");
    expect(container.querySelector("ul")).not.toBeNull();
  });

  it("shows HTML and links from the answer as text, never as markup", () => {
    const { container } = render(
      <AnswerText
        text={'<img src=x onerror="alert(1)"> [click](javascript:alert(1)) <script>alert(2)</script>'}
        knownRefs={new Set()}
        onCitationClick={vi.fn()}
      />,
    );
    expect(container.querySelector("img, script, a")).toBeNull();
    expect(container.textContent).toContain('<img src=x onerror="alert(1)">');
    expect(container.textContent).toContain("[click](javascript:alert(1))");
  });
});

describe("SourceList", () => {
  it("shows the document, the page and the section of each source", async () => {
    const onSelect = vi.fn();
    const first = source();
    const second = source({
      ref: "S2",
      document: "manual.pdf",
      page: 7,
      section: null,
      chunk_id: "chunk-2",
    });
    render(<SourceList sources={[first, second]} onSelect={onSelect} />);

    expect(screen.getByText("Sources")).toBeInTheDocument();
    expect(screen.getByText("contract.pdf")).toBeInTheDocument();
    expect(screen.getByText("Page 14 · Termination")).toBeInTheDocument();
    expect(screen.getByText("manual.pdf")).toBeInTheDocument();
    expect(screen.getByText("Page 7")).toBeInTheDocument();

    await userEvent.click(screen.getByText("manual.pdf"));
    expect(onSelect).toHaveBeenCalledWith(second);
  });

  it("renders nothing without sources", () => {
    const { container } = render(<SourceList sources={[]} onSelect={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("StatusBadge", () => {
  it("labels every status", () => {
    render(
      <>
        <StatusBadge status="uploaded" />
        <StatusBadge status="processing" />
        <StatusBadge status="indexed" />
        <StatusBadge status="failed" />
      </>,
    );
    for (const label of ["Uploaded", "Processing", "Indexed", "Failed"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });
});

describe("DocumentTable", () => {
  it("lists documents with the actions that match their status", async () => {
    const onDelete = vi.fn();
    const onReprocess = vi.fn();
    const onDownload = vi.fn();
    const indexed = documentItem();
    const failed = documentItem({
      id: "doc-2",
      filename: "scan.pdf",
      status: "failed",
      error_message: "The document contains no extractable text.",
      page_count: null,
      chunk_count: 0,
    });
    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <DocumentTable
          documents={[indexed, failed]}
          onDelete={onDelete}
          onReprocess={onReprocess}
          onDownload={onDownload}
        />
      </MemoryRouter>,
    );

    for (const header of ["Filename", "Status", "Upload date", "Size", "Actions"]) {
      expect(screen.getByRole("columnheader", { name: header })).toBeInTheDocument();
    }
    expect(screen.getByText("contract.pdf")).toBeInTheDocument();
    expect(screen.getAllByText("1.5 KB")).toHaveLength(2);
    expect(screen.getByText("The document contains no extractable text.")).toBeInTheDocument();

    // Insights only for the indexed document, Retry only for the failed one.
    expect(screen.getAllByRole("link", { name: /Insights/ })).toHaveLength(1);
    expect(screen.getByRole("link", { name: /Insights/ })).toHaveAttribute(
      "href",
      "/intelligence/doc-1",
    );
    await userEvent.click(screen.getByRole("button", { name: /Retry/ }));
    expect(onReprocess).toHaveBeenCalledWith(failed);

    await userEvent.click(screen.getByRole("button", { name: "Delete contract.pdf" }));
    expect(onDelete).toHaveBeenCalledWith(indexed);
    await userEvent.click(screen.getByRole("button", { name: "Download scan.pdf" }));
    expect(onDownload).toHaveBeenCalledWith(failed);
  });
});
