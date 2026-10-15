// Types mirroring the JSON returned by the backend (see /docs for the OpenAPI schema).

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

// ----------------------------------------------------------------------- auth
export interface User {
  id: string;
  email: string;
  full_name: string | null;
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  refresh_expires_in: number;
}

// ------------------------------------------------------------------ documents
export type DocumentStatus = "uploaded" | "processing" | "indexed" | "failed";

export interface DocumentItem {
  id: string;
  filename: string;
  content_type: string;
  extension: string;
  size_bytes: number;
  status: DocumentStatus;
  error_message: string | null;
  page_count: number | null;
  chunk_count: number;
  created_at: string;
  updated_at: string;
  indexed_at: string | null;
}

export interface DocumentStatusInfo {
  id: string;
  status: DocumentStatus;
  error_message: string | null;
  page_count: number | null;
  chunk_count: number;
  updated_at: string;
  indexed_at: string | null;
}

export interface Chunk {
  id: string;
  document_id: string;
  chunk_index: number;
  text: string;
  page: number | null;
  section: string | null;
  char_start: number;
  char_end: number;
}

// ----------------------------------------------------------------------- chat
export interface Source {
  ref: string;
  document_id: string;
  document: string;
  page: number | null;
  section: string | null;
  chunk_id: string;
  excerpt: string;
  score: number;
}

export interface ChatMetadata {
  model: string | null;
  retrieval_ms: number;
  llm_ms: number;
  total_ms: number;
  degraded: string[];
}

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  answer: string;
  answered: boolean;
  sources: Source[];
  metadata: ChatMetadata;
}

export interface ConversationSummary {
  id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  answered: boolean | null;
  created_at: string;
}

export interface ConversationDetail {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: Message[];
}

// --------------------------------------------------------------- intelligence
export type JobStatus = "pending" | "processing" | "completed" | "failed";

export type EntityType = "person" | "organization" | "location" | "other";

export interface Entity {
  name: string;
  type: EntityType;
}

export interface KeyDate {
  date: string | null;
  text: string;
  description: string;
  page: number | null;
  chunk_id: string | null;
}

export interface MonetaryAmount {
  value: number | null;
  currency: string | null;
  text: string;
  description: string;
  page: number | null;
  chunk_id: string | null;
}

export interface Insights {
  document_id: string;
  status: JobStatus;
  document_type: string | null;
  title: string | null;
  language: string | null;
  summary: string | null;
  authors: string[];
  keywords: string[];
  entities: Entity[];
  important_dates: KeyDate[];
  monetary_amounts: MonetaryAmount[];
  key_points: string[];
  truncated: boolean;
  model: string | null;
  error_message: string | null;
  updated_at: string;
}

// ----------------------------------------------------------------- comparison
export const ASPECTS = [
  "pricing",
  "dates",
  "conditions",
  "clauses",
  "deadlines",
  "obligations",
] as const;

export type Aspect = (typeof ASPECTS)[number];

export type Significance = "high" | "medium" | "low";

export interface ComparisonSource {
  ref: string;
  document_id: string;
  document: string;
  page: number | null;
  section: string | null;
  chunk_id: string;
  excerpt: string;
}

export interface ComparisonSide {
  value: string | null;
  sources: ComparisonSource[];
}

export interface Difference {
  aspect: string;
  title: string;
  significance: Significance;
  document_a: ComparisonSide;
  document_b: ComparisonSide;
}

export interface ComparisonResult {
  summary: string;
  differences: Difference[];
  similarities: string[];
  model: string | null;
}

export interface ComparedDocument {
  id: string | null;
  name: string;
}

export interface ComparisonSummary {
  id: string;
  status: JobStatus;
  document_a: ComparedDocument;
  document_b: ComparedDocument;
  aspects: string[];
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface Comparison extends ComparisonSummary {
  result: ComparisonResult | null;
}

// ------------------------------------------------------------------ dashboard
export interface DashboardStats {
  documents: {
    total: number;
    uploaded: number;
    processing: number;
    indexed: number;
    failed: number;
  };
  conversations: number;
  comparisons: number;
  passages: number;
  storage_bytes: number;
}

/** A passage that can be opened in the side panel (chat source, comparison source...). */
export interface PassageRef {
  document_id: string;
  chunk_id: string;
  document: string;
  page: number | null;
  section: string | null;
  excerpt?: string;
  ref?: string;
}
