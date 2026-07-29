"""HTTP router facade for the canonical learning workflow."""

from fastapi import APIRouter

from al_medlit.workflow.routes.data import router as data_router
from al_medlit.workflow.routes.guidelines import router as guidelines_router
from al_medlit.workflow.routes.learning import router as learning_router
from al_medlit.workflow.routes.models import router as models_router
from al_medlit.workflow.routes.rounds import router as rounds_router
from al_medlit.workflow.routes.settings import router as settings_router
from al_medlit.workflow.routes.training import router as training_router
from al_medlit.workflow.routes.workspace import router as workspace_router

router = APIRouter(tags=["workflow"])
router.include_router(settings_router)
router.include_router(data_router)
router.include_router(workspace_router)
router.include_router(models_router)
router.include_router(learning_router)
router.include_router(rounds_router)
router.include_router(training_router)
router.include_router(guidelines_router)
