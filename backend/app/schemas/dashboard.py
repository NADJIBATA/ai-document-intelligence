from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentStats(BaseModel):
    total: int
    uploaded: int = Field(description="Waiting for a worker.")
    processing: int
    indexed: int
    failed: int


class DashboardStats(BaseModel):
    documents: DocumentStats
    conversations: int
    comparisons: int
    passages: int = Field(description="Indexed passages (chunks) across my documents.")
    storage_bytes: int = Field(description="Total size of my original files.")
