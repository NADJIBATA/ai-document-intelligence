import { describe, expect, it } from "vitest";

import {
  classNames,
  formatAmount,
  formatBytes,
  formatDate,
  formatDateTime,
  formatDuration,
  humanize,
  pageLabel,
} from "../lib/format";

describe("formatBytes", () => {
  it("uses the largest suitable unit", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
    expect(formatBytes(250 * 1024 * 1024)).toBe("250 MB");
  });

  it("handles invalid values", () => {
    expect(formatBytes(-1)).toBe("0 B");
    expect(formatBytes(Number.NaN)).toBe("0 B");
  });
});

describe("dates", () => {
  it("returns a dash for missing or invalid values", () => {
    expect(formatDateTime(null)).toBe("—");
    expect(formatDateTime("not a date")).toBe("—");
    expect(formatDate(null)).toBe("—");
  });

  it("keeps the calendar day of an ISO date in every time zone", () => {
    expect(formatDate("2025-03-31")).toContain("31");
    expect(formatDate("2025-03-31")).toContain("2025");
  });

  it("returns text that is not an ISO date as is", () => {
    expect(formatDate("March 2025")).toBe("March 2025");
  });
});

describe("other helpers", () => {
  it("formats amounts with or without a currency", () => {
    expect(formatAmount(null, "EUR")).toBeNull();
    expect(formatAmount(1200, "EUR")).toContain("1");
    expect(formatAmount(1200, "euros")).toContain("euros");
    expect(formatAmount(12, null)).toBe("12");
  });

  it("formats durations", () => {
    expect(formatDuration(420)).toBe("420 ms");
    expect(formatDuration(2650)).toBe("2.6 s");
  });

  it("humanizes identifiers", () => {
    expect(humanize("scientific_paper")).toBe("Scientific paper");
    expect(humanize("pricing")).toBe("Pricing");
  });

  it("builds page labels and class names", () => {
    expect(pageLabel(14)).toBe("Page 14");
    expect(pageLabel(null)).toBeNull();
    expect(classNames("a", false, null, undefined, "b")).toBe("a b");
  });
});
