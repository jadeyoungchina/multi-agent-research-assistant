from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import ApplicationServices
from app.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    services = ApplicationServices(
        documents=SimpleNamespace(),
        research=SimpleNamespace(list_incomplete_ids=lambda: []),
    )
    with TestClient(create_app(services=services), raise_server_exceptions=False) as client:
        yield client
