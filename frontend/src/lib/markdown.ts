/**
 * Small Markdown parser for the assistant answers.
 *
 * Supported: paragraphs (single line breaks kept), headings, bullet and numbered lists
 * (nested by indentation), tables, block quotes, fenced code blocks, horizontal rules,
 * **bold**, *italic*, ~~strikethrough~~, `code` and citation markers [S1].
 *
 * The result is a tree rendered by React components: no HTML string is ever produced or
 * injected, so text coming from the model (or from a document it quotes) cannot add
 * markup, scripts or links. Raw HTML, links and images are shown as plain text.
 * The parser never throws: anything it does not recognise stays text.
 */

import { splitAnswer } from "./citations";

export type Inline =
  | { type: "text"; value: string }
  | { type: "citation"; ref: string }
  | { type: "code"; value: string }
  | { type: "strong"; children: Inline[] }
  | { type: "em"; children: Inline[] }
  | { type: "del"; children: Inline[] }
  | { type: "break" };

export type Align = "left" | "center" | "right" | null;

export interface ListItem {
  content: Inline[];
  children: Block[];
}

export type Block =
  | { type: "paragraph"; content: Inline[] }
  | { type: "heading"; level: number; content: Inline[] }
  | { type: "list"; ordered: boolean; start: number; items: ListItem[] }
  | { type: "table"; align: Align[]; header: Inline[][]; rows: Inline[][][] }
  | { type: "quote"; children: Block[] }
  | { type: "code"; value: string }
  | { type: "rule" };

const FENCE = /^ {0,3}(`{3,}|~{3,})/;
const HEADING = /^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$/;
const RULE = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const QUOTE = /^ {0,3}>[ \t]?(.*)$/;
const LIST_ITEM = /^([ \t]*)([-*+]|\d{1,9}[.)])[ \t]+(.*)$/;
const TABLE_SEPARATOR = /^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$/;
const MAX_DEPTH = 8;

// ----------------------------------------------------------------------- blocks
export function parseMarkdown(source: string): Block[] {
  return parseBlocks(source.replace(/\r\n?/g, "\n").split("\n"), 0);
}

function isBlank(line: string): boolean {
  return line.trim() === "";
}

function indentOf(line: string): number {
  let width = 0;
  for (const char of line) {
    if (char === " ") width += 1;
    else if (char === "\t") width += 4;
    else break;
  }
  return width;
}

function isTableStart(lines: string[], index: number): boolean {
  const header = lines[index] ?? "";
  const separator = lines[index + 1] ?? "";
  return (
    header.includes("|") &&
    TABLE_SEPARATOR.test(separator) &&
    (separator.includes("|") || header.trim().startsWith("|")) &&
    splitRow(header).length === splitRow(separator).length
  );
}

/** A line that interrupts a paragraph (start of another block). */
function startsBlock(lines: string[], index: number): boolean {
  const line = lines[index] ?? "";
  const item = LIST_ITEM.exec(line);
  // As in CommonMark, a numbered line only interrupts a paragraph when it starts at 1:
  // "...signed in\n2024. The contract" stays one paragraph.
  const interruptsWithList = item !== null && (!/\d/.test(item[2] ?? "") || /^1[.)]$/.test(item[2] ?? ""));
  return (
    FENCE.test(line) ||
    HEADING.test(line) ||
    RULE.test(line) ||
    QUOTE.test(line) ||
    interruptsWithList ||
    isTableStart(lines, index)
  );
}

function parseBlocks(lines: string[], depth: number): Block[] {
  const blocks: Block[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index] ?? "";
    if (isBlank(line)) {
      index += 1;
      continue;
    }

    const fence = FENCE.exec(line);
    if (fence) {
      const marker = fence[1] ?? "```";
      const body: string[] = [];
      index += 1;
      while (index < lines.length) {
        const current = lines[index] ?? "";
        if (current.trim().startsWith(marker) && current.trim().replace(/[`~]/g, "") === "") {
          index += 1;
          break;
        }
        body.push(current);
        index += 1;
      }
      blocks.push({ type: "code", value: body.join("\n") });
      continue;
    }

    const heading = HEADING.exec(line);
    if (heading) {
      blocks.push({
        type: "heading",
        level: (heading[1] ?? "#").length,
        content: parseInline(heading[2] ?? ""),
      });
      index += 1;
      continue;
    }

    if (RULE.test(line)) {
      blocks.push({ type: "rule" });
      index += 1;
      continue;
    }

    if (isTableStart(lines, index)) {
      const align = splitRow(lines[index + 1] ?? "").map(parseAlign);
      const header = splitRow(line).map(parseInline);
      const rows: Inline[][][] = [];
      index += 2;
      while (index < lines.length) {
        const row = lines[index] ?? "";
        if (isBlank(row) || !row.includes("|")) break;
        const cells = splitRow(row);
        // Rows with missing or extra cells are aligned on the header.
        rows.push(header.map((_, column) => parseInline(cells[column] ?? "")));
        index += 1;
      }
      blocks.push({ type: "table", align, header, rows });
      continue;
    }

    if (QUOTE.test(line)) {
      const inner: string[] = [];
      while (index < lines.length) {
        const match = QUOTE.exec(lines[index] ?? "");
        if (!match) break;
        inner.push(match[1] ?? "");
        index += 1;
      }
      blocks.push({
        type: "quote",
        children:
          depth < MAX_DEPTH
            ? parseBlocks(inner, depth + 1)
            : [{ type: "paragraph", content: parseInline(inner.join("\n")) }],
      });
      continue;
    }

    if (LIST_ITEM.test(line)) {
      const parsed = parseList(lines, index, depth);
      blocks.push(parsed.block);
      index = parsed.next;
      continue;
    }

    const paragraph: string[] = [line.trim()];
    index += 1;
    while (index < lines.length && !isBlank(lines[index] ?? "") && !startsBlock(lines, index)) {
      paragraph.push((lines[index] ?? "").trim());
      index += 1;
    }
    blocks.push({ type: "paragraph", content: parseInline(paragraph.join("\n")) });
  }
  return blocks;
}

function parseList(
  lines: string[],
  start: number,
  depth: number,
): { block: Extract<Block, { type: "list" }>; next: number } {
  const first = LIST_ITEM.exec(lines[start] ?? "");
  const baseIndent = indentOf(first?.[1] ?? "");
  const firstMarker = first?.[2] ?? "-";
  const ordered = /\d/.test(firstMarker);
  const startNumber = ordered ? Number.parseInt(firstMarker, 10) : 1;

  const items: ListItem[] = [];
  let index = start;
  while (index < lines.length) {
    const match = LIST_ITEM.exec(lines[index] ?? "");
    if (!match) break;
    const indent = indentOf(match[1] ?? "");
    const isOrdered = /\d/.test(match[2] ?? "");
    if (indent < baseIndent || isOrdered !== ordered) break;

    // Lines that belong to this item: continuation text and deeper (nested) lines.
    const text: string[] = [match[3] ?? ""];
    const nested: string[] = [];
    index += 1;
    while (index < lines.length) {
      const current = lines[index] ?? "";
      if (isBlank(current)) {
        // A blank line ends the item unless the list (or this item) goes on after it.
        const nextLine = lines.slice(index + 1).find((candidate) => !isBlank(candidate));
        if (nextLine !== undefined && indentOf(nextLine) > baseIndent) {
          nested.push("");
          index += 1;
          continue;
        }
        if (nextLine !== undefined && LIST_ITEM.test(nextLine) && indentOf(nextLine) === baseIndent) {
          index += 1;
        }
        break;
      }
      const currentIndent = indentOf(current);
      if (currentIndent > baseIndent && (nested.length > 0 || LIST_ITEM.test(current))) {
        nested.push(current);
      } else if (LIST_ITEM.test(current) || startsBlock(lines, index)) {
        break;
      } else if (nested.length > 0) {
        nested.push(current);
      } else {
        text.push(current.trim());
      }
      index += 1;
    }

    items.push({
      content: parseInline(text.join("\n")),
      children:
        nested.length > 0 && depth < MAX_DEPTH
          ? parseBlocks(dedent(nested), depth + 1)
          : nested.length > 0
            ? [{ type: "paragraph", content: parseInline(nested.join("\n").trim()) }]
            : [],
    });
  }
  return { block: { type: "list", ordered, start: startNumber, items }, next: index };
}

function dedent(lines: string[]): string[] {
  const widths = lines.filter((line) => !isBlank(line)).map(indentOf);
  const shift = widths.length > 0 ? Math.min(...widths) : 0;
  return lines.map((line) => {
    let removed = 0;
    let position = 0;
    while (position < line.length && removed < shift) {
      const char = line[position];
      if (char === " ") removed += 1;
      else if (char === "\t") removed += 4;
      else break;
      position += 1;
    }
    return line.slice(position);
  });
}

function splitRow(row: string): string[] {
  let trimmed = row.trim();
  if (trimmed.startsWith("|")) trimmed = trimmed.slice(1);
  if (trimmed.endsWith("|") && !trimmed.endsWith("\\|")) trimmed = trimmed.slice(0, -1);
  const cells: string[] = [];
  let current = "";
  for (let position = 0; position < trimmed.length; position += 1) {
    const char = trimmed[position];
    if (char === "\\" && trimmed[position + 1] === "|") {
      current += "|";
      position += 1;
    } else if (char === "|") {
      cells.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  cells.push(current.trim());
  return cells;
}

function parseAlign(cell: string): Align {
  const left = cell.startsWith(":");
  const right = cell.endsWith(":");
  if (left && right) return "center";
  if (right) return "right";
  if (left) return "left";
  return null;
}

// ----------------------------------------------------------------------- inline
const ESCAPABLE = new Set("\\`*_~[]()#+-.!|>");
const WORD_CHAR = /[\p{L}\p{N}]/u;

export function parseInline(text: string, depth = 0): Inline[] {
  const out: Inline[] = [];
  let buffer = "";

  const flush = () => {
    if (buffer) {
      for (const part of splitAnswer(buffer)) {
        if (part.type === "citation") {
          out.push({ type: "citation", ref: part.ref });
        } else {
          pushText(out, part.value);
        }
      }
      buffer = "";
    }
  };

  let position = 0;
  while (position < text.length) {
    const char = text[position] ?? "";
    const next = text[position + 1] ?? "";

    if (char === "\\" && ESCAPABLE.has(next)) {
      buffer += next;
      position += 2;
      continue;
    }

    if (char === "\n") {
      flush();
      out.push({ type: "break" });
      position += 1;
      continue;
    }

    if (char === "`") {
      const run = /^`+/.exec(text.slice(position))?.[0] ?? "`";
      const close = text.indexOf(run, position + run.length);
      if (close > position) {
        flush();
        out.push({ type: "code", value: text.slice(position + run.length, close).trim() });
        position = close + run.length;
        continue;
      }
      buffer += run;
      position += run.length;
      continue;
    }

    if (depth < MAX_DEPTH) {
      const triple = text.slice(position, position + 3);
      if ((triple === "***" || triple === "___") && !/\s/.test(text[position + 3] ?? " ")) {
        const close = findClosing(text, position + 3, triple);
        if (close !== -1) {
          flush();
          const children = parseInline(text.slice(position + 3, close), depth + 1);
          out.push({ type: "strong", children: [{ type: "em", children }] });
          position = close + 3;
          continue;
        }
      }
      const delimiter = emphasisDelimiter(text, position);
      if (delimiter) {
        const close = findClosing(text, position + delimiter.length, delimiter);
        if (close !== -1) {
          flush();
          const children = parseInline(text.slice(position + delimiter.length, close), depth + 1);
          const type = delimiter === "~~" ? "del" : delimiter.length === 2 ? "strong" : "em";
          out.push({ type, children });
          position = close + delimiter.length;
          continue;
        }
      }
    }

    buffer += char;
    position += 1;
  }
  flush();
  return out;
}

function pushText(out: Inline[], value: string): void {
  const last = out[out.length - 1];
  if (last?.type === "text") {
    last.value += value;
  } else {
    out.push({ type: "text", value });
  }
}

/** Opening emphasis delimiter at ``position``, if any. */
function emphasisDelimiter(text: string, position: number): string | null {
  const char = text[position];
  if (char !== "*" && char !== "_" && char !== "~") return null;
  const double = text[position + 1] === char;
  if (char === "~" && !double) return null;
  const delimiter = double ? char + char : char;
  const after = text[position + delimiter.length] ?? "";
  // Must be followed by text: "2 * 3" or "* item" are not emphasis.
  if (after === "" || /\s/.test(after)) return null;
  // Underscores inside words (snake_case) are not emphasis.
  if (char === "_" && WORD_CHAR.test(text[position - 1] ?? "")) return null;
  return delimiter;
}

function findClosing(text: string, from: number, delimiter: string): number {
  let position = from;
  while (position < text.length) {
    const found = text.indexOf(delimiter, position);
    if (found === -1) return -1;
    const before = text[found - 1] ?? "";
    const after = text[found + delimiter.length] ?? "";
    const doubled = delimiter.length === 1 && after === delimiter;
    const valid =
      found > from &&
      !/\s/.test(before) &&
      before !== "\\" &&
      !doubled &&
      !(delimiter[0] === "_" && WORD_CHAR.test(after));
    if (valid) return found;
    // "**" inside "*...*": skip both characters.
    position = found + (doubled ? 2 : 1);
  }
  return -1;
}

/** Plain text of an inline tree (accessible labels, tests). */
export function inlineText(nodes: Inline[]): string {
  return nodes
    .map((node) => {
      switch (node.type) {
        case "text":
        case "code":
          return node.value;
        case "citation":
          return `[${node.ref}]`;
        case "break":
          return "\n";
        default:
          return inlineText(node.children);
      }
    })
    .join("");
}
