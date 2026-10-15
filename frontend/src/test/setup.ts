import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Unmount what each test rendered (automatic cleanup needs Vitest globals, which this
// project does not enable).
afterEach(() => {
  cleanup();
});
