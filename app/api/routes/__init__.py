from fastapi import APIRouter

from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.research import router as research_router
from app.api.routes.ui import router as ui_router


api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(documents_router)
api_router.include_router(research_router)
api_router.include_router(ui_router)
