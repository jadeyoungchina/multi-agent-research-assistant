from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import ApplicationServices
from app.bootstrap import ApplicationContainer
from app.config import Settings
from app.main import create_app
from app.storage.database import Database


class CloseableProvider:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


@pytest.fixture
def application_container(tmp_path) -> ApplicationContainer:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        database_path=tmp_path / "research.db",
        chat_provider="fake",
        embedding_provider="fake",
    )
    database = Database(settings.database_path)
    database.initialize()
    return ApplicationContainer(
        settings=settings,
        database=database,
        chat_provider=CloseableProvider(),
        embedding_provider=CloseableProvider(),
        services=ApplicationServices(
            documents=SimpleNamespace(),
            research=SimpleNamespace(list_incomplete_ids=lambda: []),
        ),
    )


@pytest.fixture
def client(application_container: ApplicationContainer) -> Iterator[TestClient]:
    app = create_app(container=application_container)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
