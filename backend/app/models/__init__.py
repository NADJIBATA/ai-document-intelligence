"""Import every model so that ``Base.metadata`` is complete (Alembic autogenerate)."""

from app.models.chunk import DocumentChunk
from app.models.comparison import DocumentComparison
from app.models.conversation import Conversation, Message
from app.models.document import Document
from app.models.enums import DocumentStatus, JobStatus, MessageRole
from app.models.insight import DocumentInsight
from app.models.refresh_token import RefreshToken
from app.models.user import User

__all__ = [
    "Conversation",
    "Document",
    "DocumentChunk",
    "DocumentComparison",
    "DocumentInsight",
    "DocumentStatus",
    "JobStatus",
    "Message",
    "MessageRole",
    "RefreshToken",
    "User",
]
