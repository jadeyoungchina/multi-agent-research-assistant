from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dependencies import ApplicationServices
from app.api.errors import register_error_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routes import api_router
from app.bootstrap import ApplicationContainer, build_container


def create_app(
    container: ApplicationContainer | None = None,
    services: ApplicationServices | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_container = container
        if active_container is None and services is None:
            active_container = build_container()
        active_services = services or active_container.services
        app.state.container = active_container
        app.state.services = active_services
        app.state.settings = (
            active_container.settings if active_container is not None else None
        )
        try:
            yield
        finally:
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
    return application


app = create_app()
