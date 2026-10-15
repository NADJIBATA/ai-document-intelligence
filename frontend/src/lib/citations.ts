export type AnswerPart = { type: "text"; value: string } | { type: "citation"; ref: string };

const MARKER = /\[(S\d+)\]/g;

/**
 * Split an answer into plain text and citation markers:
 * "Thirty days [S1]." -> text "Thirty days ", citation "S1", text "."
 */
export function splitAnswer(answer: string): AnswerPart[] {
  const parts: AnswerPart[] = [];
  let cursor = 0;
  for (const match of answer.matchAll(MARKER)) {
    const index = match.index ?? 0;
    if (index > cursor) {
      parts.push({ type: "text", value: answer.slice(cursor, index) });
    }
    parts.push({ type: "citation", ref: match[1] ?? "" });
    cursor = index + match[0].length;
  }
  if (cursor < answer.length) {
    parts.push({ type: "text", value: answer.slice(cursor) });
  }
  return parts;
}
