import logging
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any

from app.agents.critic import CriticAgent
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearcherAgent
from app.agents.retriever import RetrieverAgent
from app.agents.writer import WriterAgent
from app.api.dependencies import ApplicationServices
from app.config import Settings, get_settings
from app.observability import install_redaction_filter, log_event
from app.providers.factory import build_chat_provider, build_embedding_provider
from app.retrieval.index import LocalVectorIndex
from app.retrieval.retriever import EvidenceRetriever
from app.services.documents import DocumentService
from app.services.research import ResearchService
from app.storage.database import Database
from app.storage.documents import DocumentRepository
from app.storage.files import LocalDocumentStore
from app.storage.runs import RunRepository
from app.workflow.checkpoints import checkpointed_node
from app.workflow.citations import CitationValidatorNode
from app.workflow.graph import WorkflowDependencies, build_research_graph


logger = logging.getLogger(__name__)
install_redaction_filter(logger)


def configured_chat_identity(settings: Settings) -> tuple[str, str]:
    if settings.chat_provider == "dashscope":
        return "dashscope", settings.dashscope_chat_model
    if settings.chat_provider == "openai":
        return "openai", settings.openai_chat_model
    return "fake", "fake-chat"


def _close_target(provider: object) -> object | None:
    if callable(getattr(provider, "close", None)):
        return provider
    client = getattr(provider, "client", None)
    if callable(getattr(client, "close", None)):
        return client
    return None


def _close_providers(*providers: object | None) -> None:
    closed: set[int] = set()
    for provider in providers:
        if provider is None:
            continue
        target = _close_target(provider)
        if target is None or id(target) in closed:
            continue
        closed.add(id(target))
        try:
            target.close()
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "provider_close_failed",
                error_type=type(exc).__name__,
            )


@dataclass
class ApplicationContainer:
    settings: Settings
    database: Database
    chat_provider: object
    embedding_provider: object
    services: ApplicationServices
    _closed: bool = field(default=False, init=False, repr=False)

    def database_ready(self) -> bool:
        try:
            with closing(self.database.connect()) as connection:
                connection.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def provider_readiness(self) -> dict[str, bool]:
        return {
            "chat": self.chat_provider is not None,
            "embedding": self.embedding_provider is not None,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _close_providers(self.chat_provider, self.embedding_provider)


def build_container(settings: Settings | None = None) -> ApplicationContainer:
    active_settings = settings or get_settings()
    active_settings.data_dir.mkdir(parents=True, exist_ok=True)
    active_settings.upload_dir.mkdir(parents=True, exist_ok=True)

    database = Database(active_settings.database_path)
    database.initialize()
    documents = DocumentRepository(database)
    runs = RunRepository(database)
    chat: Any | None = None
    embeddings: Any | None = None

    try:
        chat = build_chat_provider(active_settings)
        embeddings = build_embedding_provider(active_settings)
        document_service = DocumentService(
            documents,
            LocalDocumentStore(active_settings.upload_dir),
            embeddings,
            active_settings,
        )
        retriever = EvidenceRetriever(
            embeddings,
            LocalVectorIndex(documents),
            candidate_k=active_settings.retrieval_candidate_k,
            rrf_k=active_settings.retrieval_rrf_k,
            min_similarity=active_settings.retrieval_min_similarity,
            max_query_expansions=active_settings.max_query_expansions,
        )
        dependencies = WorkflowDependencies(
            planner=PlannerAgent(chat),
            retriever=RetrieverAgent(
                retriever,
                top_k=active_settings.retrieval_top_k,
            ),
            researcher=ResearcherAgent(chat),
            critic=CriticAgent(
                chat,
                max_iterations=active_settings.max_revision_iterations,
            ),
            writer=WriterAgent(chat),
            citation_validator=CitationValidatorNode(),
            max_iterations=active_settings.max_revision_iterations,
            node_wrapper=lambda stage, handler: checkpointed_node(
                stage, handler, runs
            ),
        )
        graph = build_research_graph(dependencies)
        provider_name, model_name = configured_chat_identity(active_settings)
        research_service = ResearchService(
            documents,
            runs,
            graph,
            provider_name,
            model_name,
            recursion_limit=active_settings.max_workflow_steps,
        )
        services = ApplicationServices(
            documents=document_service,
            research=research_service,
        )
        return ApplicationContainer(
            settings=active_settings,
            database=database,
            chat_provider=chat,
            embedding_provider=embeddings,
            services=services,
        )
    except Exception:
        _close_providers(chat, embeddings)
        raise
