from fastapi import APIRouter, Request

from app.api.schemas import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    container = getattr(request.app.state, "container", None)
    if container is None:
        database_ready = True
        providers = {"chat": True, "embedding": True}
    else:
        database_ready = container.database_ready()
        providers = container.provider_readiness()
    ready = database_ready and all(providers.values())
    return HealthResponse(
        status="ok" if ready else "degraded",
        version=request.app.version,
        database_ready=database_ready,
        providers=providers,
    )
