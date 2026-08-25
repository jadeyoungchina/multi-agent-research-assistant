from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_services, get_task_manager
from app.api.schemas import (
    ResearchCreateRequest,
    ResearchCreateResponse,
    ResearchRunResponse,
)
from app.api.sse import stream_run_events


router = APIRouter(prefix="/api/research", tags=["research"])

_POLL_INTERVAL_SECONDS = 0.25
_HEARTBEAT_INTERVAL_SECONDS = 15.0


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


@router.get("/{run_id}/events")
async def stream_research_events(
    run_id: str,
    request: Request,
    after: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    services=Depends(get_services),
) -> StreamingResponse:
    try:
        header_cursor = int(last_event_id) if last_event_id is not None else 0
    except ValueError:
        header_cursor = 0
    after_sequence = max(after, header_cursor)

    await run_in_threadpool(services.research.get_run, run_id)
    return StreamingResponse(
        stream_run_events(
            request,
            run_id,
            services.research,
            after_sequence,
            _POLL_INTERVAL_SECONDS,
            _HEARTBEAT_INTERVAL_SECONDS,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
