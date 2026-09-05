from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


router = APIRouter(include_in_schema=False)
_STATIC_DIRECTORY = Path(__file__).resolve().parents[2] / "static"


@router.get("/")
def dashboard() -> FileResponse:
    return FileResponse(_STATIC_DIRECTORY / "index.html")
