from fastapi import APIRouter, Depends, Response

from app.api.dependencies import get_services, get_task_manager
from app.api.schemas import (
    ResearchCreateRequest,
    ResearchCreateResponse,
    ResearchRunResponse,
)


router = APIRouter(prefix="/api/research", tags=["research"])


@router.post("", status_code=202, response_model=ResearchCreateResponse)
def create_research(
    payload: ResearchCreateRequest,
    response: Response,
    services=Depends(get_services),
    tasks=Depends(get_task_manager),
) -> ResearchCreateResponse:
    document_ids = list(dict.fromkeys(payload.document_ids))
    run = services.research.create_run(payload.question, document_ids)
    tasks.start(run.id)
    response.headers["Location"] = f"/api/research/{run.id}"
    return ResearchCreateResponse(run_id=run.id, status=run.status)


@router.get("/{run_id}", response_model=ResearchRunResponse)
def get_research(
    run_id: str,
    services=Depends(get_services),
) -> ResearchRunResponse:
    run = services.research.get_run(run_id)
    return ResearchRunResponse(
        run=run,
        report=services.research.get_report(run_id),
    )
