import asyncio
from time import monotonic
from typing import AsyncIterator

from fastapi import Request
from starlette.concurrency import run_in_threadpool

from app.domain.runs import RunEvent


def encode_sse(event: RunEvent) -> bytes:
    data = event.model_dump_json(exclude_none=True)
    return f"id: {event.sequence}\nevent: workflow\ndata: {data}\n\n".encode("utf-8")


async def stream_run_events(
    request: Request,
    run_id: str,
    service: object,
    after_sequence: int,
    poll_interval: float,
    heartbeat_interval: float,
) -> AsyncIterator[bytes]:
    cursor = after_sequence
    last_sent = monotonic()
    while not await request.is_disconnected():
        events = await run_in_threadpool(service.list_events, run_id, cursor)
        for event in events:
            cursor = event.sequence
            last_sent = monotonic()
            yield encode_sse(event)
        run = await run_in_threadpool(service.get_run, run_id)
        if run.status in {"completed", "failed"} and not events:
            break
        if monotonic() - last_sent >= heartbeat_interval:
            last_sent = monotonic()
            yield b": keep-alive\n\n"
        await asyncio.sleep(poll_interval)
