from dataclasses import dataclass

from fastapi import Request


@dataclass(frozen=True)
class ApplicationServices:
    documents: object
    research: object


def get_services(request: Request) -> ApplicationServices:
    return request.app.state.services
