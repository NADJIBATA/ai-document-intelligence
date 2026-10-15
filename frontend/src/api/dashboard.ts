import { api } from "./client";
import type { DashboardStats } from "./types";

export function getDashboardStats(signal?: AbortSignal): Promise<DashboardStats> {
  return api<DashboardStats>("/dashboard/stats", { signal });
}
