"""FastAPI dependencies shared by the routes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.context import bind_user_id
from app.core.exceptions import AuthenticationError
from app.llm.base import LLMProvider
from app.models.user import User
from app.rag.factory import RetrievalComponents
from app.rag.generation import AnswerGenerator
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.comparison_service import ComparisonService
from app.services.dashboard_service import DashboardService
from app.services.document_service import DocumentService
from app.services.intelligence_service import InsightService
from app.services.search_service import SearchService
from app.services.storage_service import ObjectStorage
from app.services.task_dispatcher import TaskDispatcher

bearer_scheme = HTTPBearer(
    auto_error=False,
    description="Access token returned by POST /api/auth/login (valid 15 minutes by default).",
)


def get_settings_from_app(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_settings_from_app)]


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """One session per request. Uncommitted work is rolled back when the session closes."""
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with factory() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(get_db_session)]


def get_auth_service(session: DbSession, settings: SettingsDep) -> AuthService:
    return AuthService(session, settings)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


async def get_current_user(
    request: Request,
    auth_service: AuthServiceDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_scheme)],
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError()
    user = await auth_service.get_user_from_access_token(credentials.credentials)
    request.state.user_id = str(user.id)  # read by the access-log middleware
    bind_user_id(str(user.id))
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_storage(request: Request) -> ObjectStorage:
    storage: ObjectStorage = request.app.state.storage
    return storage


def get_task_dispatcher(request: Request) -> TaskDispatcher:
    dispatcher: TaskDispatcher = request.app.state.task_dispatcher
    return dispatcher


def get_retrieval(request: Request) -> RetrievalComponents:
    retrieval: RetrievalComponents = request.app.state.retrieval
    return retrieval


RetrievalDep = Annotated[RetrievalComponents, Depends(get_retrieval)]


def get_document_service(
    session: DbSession,
    settings: SettingsDep,
    storage: Annotated[ObjectStorage, Depends(get_storage)],
    dispatcher: Annotated[TaskDispatcher, Depends(get_task_dispatcher)],
    retrieval: RetrievalDep,
) -> DocumentService:
    return DocumentService(session, settings, storage, dispatcher, retrieval.vector_store)


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]


def get_search_service(retrieval: RetrievalDep, settings: SettingsDep) -> SearchService:
    return SearchService(retrieval.pipeline, settings)


SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]


def get_llm(request: Request) -> LLMProvider:
    llm: LLMProvider = request.app.state.llm
    return llm


def get_chat_service(
    session: DbSession,
    settings: SettingsDep,
    retrieval: RetrievalDep,
    llm: Annotated[LLMProvider, Depends(get_llm)],
) -> ChatService:
    generator = AnswerGenerator(llm, history_turns=settings.chat_history_turns)
    return ChatService(session, settings, retrieval.pipeline, generator)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


def get_insight_service(
    session: DbSession, dispatcher: Annotated[TaskDispatcher, Depends(get_task_dispatcher)]
) -> InsightService:
    return InsightService(session, dispatcher)


InsightServiceDep = Annotated[InsightService, Depends(get_insight_service)]


def get_comparison_service(
    session: DbSession, dispatcher: Annotated[TaskDispatcher, Depends(get_task_dispatcher)]
) -> ComparisonService:
    return ComparisonService(session, dispatcher)


ComparisonServiceDep = Annotated[ComparisonService, Depends(get_comparison_service)]


def get_dashboard_service(session: DbSession) -> DashboardService:
    return DashboardService(session)


DashboardServiceDep = Annotated[DashboardService, Depends(get_dashboard_service)]
