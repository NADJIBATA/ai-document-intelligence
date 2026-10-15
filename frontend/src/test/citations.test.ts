import { describe, expect, it } from "vitest";

import { splitAnswer } from "../lib/citations";

describe("splitAnswer", () => {
  it("separates text and citation markers", () => {
    expect(splitAnswer("The notice period is thirty days [S1]. Fees apply [S2][S3].")).toEqual([
      { type: "text", value: "The notice period is thirty days " },
      { type: "citation", ref: "S1" },
      { type: "text", value: ". Fees apply " },
      { type: "citation", ref: "S2" },
      { type: "citation", ref: "S3" },
      { type: "text", value: "." },
    ]);
  });

  it("returns a single text part when there is no marker", () => {
    expect(splitAnswer("No source here.")).toEqual([{ type: "text", value: "No source here." }]);
    expect(splitAnswer("")).toEqual([]);
  });

  it("ignores brackets that are not citation markers", () => {
    expect(splitAnswer("See annex [A] and [12].")).toEqual([
      { type: "text", value: "See annex [A] and [12]." },
    ]);
  });

  it("handles a marker at the very beginning", () => {
    expect(splitAnswer("[S1] first")).toEqual([
      { type: "citation", ref: "S1" },
      { type: "text", value: " first" },
    ]);
  });
});
