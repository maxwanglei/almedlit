from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from al_medlit.annotation.router import router as annotation_router
from al_medlit.annotation_workbench.router import router as annotation_workbench_router
from al_medlit.auth.dependencies import enforce_authentication
from al_medlit.auth.router import router as auth_router
from al_medlit.co_learning.error_guideline_learning.events import register_event_handlers
from al_medlit.co_learning.error_guideline_learning.router import (
    router as error_guideline_router,
)
from al_medlit.core.config import settings
from al_medlit.core.exceptions import APIError
from al_medlit.corpus.router import router as corpus_router
from al_medlit.evidence.router import router as evidence_router
from al_medlit.export.router import router as export_router
from al_medlit.guideline.router import router as guideline_router
from al_medlit.iaa.router import router as iaa_router
from al_medlit.importers.router import router as importers_router
from al_medlit.inference.router import router as inference_router
from al_medlit.lineage.router import router as lineage_router
from al_medlit.model_artifacts.quota_router import router as artifact_quota_router
from al_medlit.model_artifacts.router import router as model_artifact_router
from al_medlit.project.router import router as project_router
from al_medlit.submission.router import router as submission_router
from al_medlit.workflow.router import router as workflow_router
from al_medlit.workspace import router as workspace_routes

API_PREFIX = "/api"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings.validate_runtime_secrets()
    register_event_handlers()
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="AL-MedLit Backend",
        description=(
            "Versioned NLP and LLM data, annotation, active-learning, training, "
            "evaluation, and guideline platform."
        ),
        version="0.1.0",
        lifespan=lifespan,
        dependencies=[Depends(enforce_authentication)],
    )

    @application.exception_handler(APIError)
    async def _handle_api_error(_request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    @application.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "al-medlit-backend"}

    application.include_router(auth_router, prefix=API_PREFIX)
    application.include_router(workspace_routes.router, prefix=API_PREFIX)
    application.include_router(workspace_routes.invite_router, prefix=API_PREFIX)
    application.include_router(workspace_routes.join_request_router, prefix=API_PREFIX)
    application.include_router(project_router, prefix=API_PREFIX)
    application.include_router(corpus_router, prefix=API_PREFIX)
    application.include_router(importers_router, prefix=API_PREFIX)
    application.include_router(annotation_router, prefix=API_PREFIX)
    application.include_router(annotation_workbench_router, prefix=API_PREFIX)
    application.include_router(evidence_router, prefix=API_PREFIX)
    application.include_router(lineage_router, prefix=API_PREFIX)
    application.include_router(model_artifact_router, prefix=API_PREFIX)
    application.include_router(artifact_quota_router, prefix=API_PREFIX)
    application.include_router(export_router, prefix=API_PREFIX)
    application.include_router(inference_router, prefix=API_PREFIX)
    application.include_router(iaa_router, prefix=API_PREFIX)
    application.include_router(submission_router, prefix=API_PREFIX)
    application.include_router(guideline_router, prefix=API_PREFIX)
    application.include_router(error_guideline_router, prefix=API_PREFIX)
    application.include_router(workflow_router, prefix=API_PREFIX)

    return application


app = create_app()
