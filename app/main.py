from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.dependencies import ApplicationServices
from app.api.errors import register_error_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routes import api_router
from app.api.tasks import RunTaskManager
from app.bootstrap import ApplicationContainer, build_container
from app.observability import configure_json_logging


_STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"


def create_app(
    container: ApplicationContainer | None = None,
    services: ApplicationServices | None = None,
    task_manager: RunTaskManager | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_json_logging()
        active_container = container
        active_task_manager = task_manager
        try:
            if active_container is None and services is None:
                active_container = build_container()
            active_services = services or active_container.services
            active_task_manager = active_task_manager or RunTaskManager(
                active_services.research
            )
            app.state.container = active_container
            app.state.services = active_services
            app.state.run_tasks = active_task_manager
            app.state.settings = (
                active_container.settings if active_container is not None else None
            )
            for run_id in dict.fromkeys(
                active_services.research.list_incomplete_ids()
            ):
                active_task_manager.start(run_id)
            yield
        finally:
            if active_task_manager is not None:
                active_task_manager.close()
            if active_container is not None:
                active_container.close()

    application = FastAPI(
        title="Multi-Agent Research Assistant",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(RequestContextMiddleware)
    register_error_handlers(application)
    application.include_router(api_router)
    application.mount(
        "/static",
        StaticFiles(directory=_STATIC_DIRECTORY),
        name="static",
    )
    return application


app = create_app()
