import { describe, expect, it } from "vitest";

import { type Block, type Inline, inlineText, parseInline, parseMarkdown } from "../lib/markdown";

const text = (value: string): Inline => ({ type: "text", value });

function only<T>(items: T[]): T {
  expect(items).toHaveLength(1);
  return items[0] as T;
}

describe("parseInline", () => {
  it("keeps plain text and citations", () => {
    expect(parseInline("Thirty days [S1].")).toEqual([
      text("Thirty days "),
      { type: "citation", ref: "S1" },
      text("."),
    ]);
  });

  it("parses bold, italic, strikethrough and code", () => {
    expect(parseInline("**90 days** and *notice* ~~old~~ `CHUNK_SIZE`")).toEqual([
      { type: "strong", children: [text("90 days")] },
      text(" and "),
      { type: "em", children: [text("notice")] },
      text(" "),
      { type: "del", children: [text("old")] },
      text(" "),
      { type: "code", value: "CHUNK_SIZE" },
    ]);
    expect(parseInline("***both***")).toEqual([
      { type: "strong", children: [{ type: "em", children: [text("both")] }] },
    ]);
  });

  it("keeps citations inside emphasis", () => {
    expect(parseInline("**12 000 € per year [S2]**")).toEqual([
      {
        type: "strong",
        children: [text("12 000 € per year "), { type: "citation", ref: "S2" }],
      },
    ]);
  });

  it("does not treat arithmetic, snake_case or lone markers as emphasis", () => {
    expect(inlineText(parseInline("2 * 3 * 4 = 24"))).toBe("2 * 3 * 4 = 24");
    expect(parseInline("max_upload_size_mb")).toEqual([text("max_upload_size_mb")]);
    expect(parseInline("**unclosed")).toEqual([text("**unclosed")]);
    expect(parseInline("price: 5 *")).toEqual([text("price: 5 *")]);
  });

  it("supports escapes and line breaks", () => {
    expect(parseInline("\\*not italic\\*")).toEqual([text("*not italic*")]);
    expect(parseInline("line one\nline two")).toEqual([
      text("line one"),
      { type: "break" },
      text("line two"),
    ]);
  });

  it("never produces markup from HTML or links", () => {
    const nodes = parseInline('<img src=x onerror="alert(1)"> [click](javascript:alert(1))');
    expect(nodes.every((node) => node.type === "text")).toBe(true);
    expect(inlineText(nodes)).toBe('<img src=x onerror="alert(1)"> [click](javascript:alert(1))');
  });
});

describe("parseMarkdown", () => {
  it("splits paragraphs on blank lines and keeps single line breaks", () => {
    const blocks = parseMarkdown("First paragraph\nstill first.\n\nSecond [S1].");
    expect(blocks).toEqual([
      {
        type: "paragraph",
        content: [text("First paragraph"), { type: "break" }, text("still first.")],
      },
      { type: "paragraph", content: [text("Second "), { type: "citation", ref: "S1" }, text(".")] },
    ]);
  });

  it("parses headings and rules", () => {
    expect(parseMarkdown("## Payment terms ##\n\n---")).toEqual([
      { type: "heading", level: 2, content: [text("Payment terms")] },
      { type: "rule" },
    ]);
  });

  it("parses bullet lists with citations in items", () => {
    const list = only(parseMarkdown("Obligations:\n- Pay within **45 days** [S1]\n- Notify breaches [S2]").slice(1));
    expect(list).toEqual({
      type: "list",
      ordered: false,
      start: 1,
      items: [
        {
          content: [
            text("Pay within "),
            { type: "strong", children: [text("45 days")] },
            text(" "),
            { type: "citation", ref: "S1" },
          ],
          children: [],
        },
        { content: [text("Notify breaches "), { type: "citation", ref: "S2" }], children: [] },
      ],
    });
  });

  it("parses numbered lists with their start number and nested lists", () => {
    const list = only(parseMarkdown("3. Third\n4. Fourth\n   - detail a\n   - detail b\n5. Fifth"));
    expect(list.type).toBe("list");
    if (list.type !== "list") return;
    expect(list.ordered).toBe(true);
    expect(list.start).toBe(3);
    expect(list.items.map((item) => inlineText(item.content))).toEqual(["Third", "Fourth", "Fifth"]);
    const nested = only(list.items[1]?.children ?? []);
    expect(nested.type === "list" && nested.items.map((item) => inlineText(item.content))).toEqual([
      "detail a",
      "detail b",
    ]);
  });

  it("keeps loose lists (blank lines between items) as one list", () => {
    const list = only(parseMarkdown("- one\n\n- two\n\n- three"));
    expect(list.type === "list" && list.items).toHaveLength(3);
  });

  it("does not start a list for a year at the start of a wrapped line", () => {
    const blocks = parseMarkdown("The contract was signed in\n2024. It runs for three years.");
    expect(only(blocks).type).toBe("paragraph");
  });

  it("parses tables with alignment, citations and uneven rows", () => {
    const table = only(
      parseMarkdown(
        "| Aspect | Contract A | Contract B |\n|:--|:-:|--:|\n| Notice | 30 days [A1] | 90 days [S2] |\n| Fees | 10 € |",
      ),
    );
    expect(table.type).toBe("table");
    if (table.type !== "table") return;
    expect(table.align).toEqual(["left", "center", "right"]);
    expect(table.header.map(inlineText)).toEqual(["Aspect", "Contract A", "Contract B"]);
    expect(table.rows.map((row) => row.map(inlineText))).toEqual([
      ["Notice", "30 days [A1]", "90 days [S2]"],
      ["Fees", "10 €", ""],
    ]);
    expect(table.rows[0]?.[2]).toContainEqual({ type: "citation", ref: "S2" });
  });

  it("does not take a line with a pipe for a table", () => {
    expect(only(parseMarkdown("Either A | B applies.")).type).toBe("paragraph");
  });

  it("parses quotes and fenced code blocks", () => {
    const blocks = parseMarkdown("> Quoted **text**\n> second line\n\n```json\n{\"a\": 1}\n```");
    expect(blocks[0]?.type).toBe("quote");
    expect(blocks[1]).toEqual({ type: "code", value: '{"a": 1}' });
  });

  it("handles Windows line endings and an unclosed code fence", () => {
    const blocks: Block[] = parseMarkdown("- a\r\n- b\r\n\r\n```\ncode without end");
    expect(blocks.map((block) => block.type)).toEqual(["list", "code"]);
  });

  it("never throws on unusual input", () => {
    const inputs = ["", "\n\n", "|", "| --- |", "-", "1.", "#", "> ", "```", "* * *", "**", "`", "_a_b_", "\\"];
    for (const input of inputs) {
      expect(() => parseMarkdown(input)).not.toThrow();
    }
    expect(() => parseMarkdown(`${"> ".repeat(50)}deep`)).not.toThrow();
    expect(() => parseMarkdown(`${"*".repeat(200)}x`)).not.toThrow();
  });
});
