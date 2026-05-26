from fastapi import APIRouter, Depends
from app.api.v1.auth_deps import require_admin
from .collection import router as collection_router
from .config import router as config_router
from .unanswered import router as unanswered_router

admin_router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])
admin_router.include_router(collection_router)
admin_router.include_router(config_router)
admin_router.include_router(unanswered_router)
