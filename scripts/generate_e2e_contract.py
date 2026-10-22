"""Export the backend contracts exercised by the shared browser fixtures."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import get_args
from urllib.parse import urlsplit

from fastapi.routing import APIRoute
from pydantic import TypeAdapter

from al_medlit.core import capabilities
from al_medlit.main import app
from al_medlit.workflow import access
from al_medlit.workflow.schemas import StoragePolicyCreate
from al_medlit.workspace.dependencies import ROLE_RANK

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "frontend/e2e/fixtures/backendContract.json"
PATHS = (
    "/api/projects",
    "/api/projects/my-work",
    "/api/projects/{project_id}",
    "/api/projects/{project_id}/datasets/annotation-progress",
    "/api/projects/{project_id}/prediction-runs",
    "/api/projects/{project_id}/prediction-runs/{run_id}",
    "/api/projects/{project_id}/prediction-runs/{run_id}/results",
    "/api/projects/{project_id}/prediction-runs/{run_id}/retry",
    "/api/projects/{project_id}/prediction-runs/{run_id}/review-round",
    "/api/rounds/{round_id}/work-context",
    "/api/storage/policies",
    "/api/workspaces/{workspace_id}/capabilities",
    "/api/workspaces/{workspace_id}/models",
    "/api/workspaces/{workspace_id}/training-contexts",
    "/api/workspaces/{workspace_id}/training-runs",
)


def build_contract() -> dict:
    document = app.openapi()
    operations = {}
    schemas = {}

    def collect(value):
        if isinstance(value, dict):
            reference = value.get("$ref")
            if reference:
                name = reference.removeprefix("#/components/schemas/")
                if name not in schemas:
                    schemas[name] = document["components"]["schemas"][name]
                    collect(schemas[name])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    for path in PATHS:
        operations[path] = {}
        for method, operation in document["paths"][path].items():
            if method not in {"get", "post", "patch", "delete"}:
                continue
            success = next(code for code in operation["responses"] if code.startswith("2"))
            response = operation["responses"][success].get("content", {}).get(
                "application/json", {}
            ).get("schema", {})
            operations[path][method] = {"status": int(success), "response": response}
            collect(response)
    return {
        "enums": {
            "capability_preset": sorted([*capabilities.PRESETS, "custom"]),
            "storage_retention_class": sorted(
                get_args(StoragePolicyCreate.model_fields["retention_class"].annotation)
            ),
        },
        "policy": {
            "capabilities": {key: asdict(value) for key, value in capabilities.CAPABILITIES.items()},
            "presets": {key: sorted(value) for key, value in capabilities.PRESETS.items()},
            "role_rank": ROLE_RANK,
            "module_order": list(access.PROJECT_MODULES),
            "module_requirements": access._WORKSPACE_REQUIREMENTS,
            "module_dependencies": access._MODULE_DEPENDENCIES,
        },
        "operations": operations,
        "schemas": schemas,
    }


def validate_responses(responses: list[dict]) -> None:
    for response in responses:
        path = urlsplit(response["path"]).path
        method = response["method"].upper()
        route = next(
            route for route in app.routes
            if isinstance(route, APIRoute)
            and route.path_regex.fullmatch(path)
            and method in route.methods
        )
        expected_status = route.status_code or 200
        if response["status"] != expected_status:
            raise ValueError(f"{method} {path}: expected {expected_status}, got {response['status']}")
        TypeAdapter(route.response_model).validate_python(response["body"])
        if path.endswith("/capabilities"):
            body = response["body"]
            if body["preset"] not in {*capabilities.PRESETS, "custom"}:
                raise ValueError(f"Unknown fixture capability preset {body['preset']!r}")
            if set(body["effective"]) - set(capabilities.CAPABILITIES):
                raise ValueError("Fixture returned unknown capabilities")
        if path in {"/api/projects", "/api/projects/my-work", "/api/projects/1"}:
            projects = response["body"] if isinstance(response["body"], list) else [response["body"]]
            for project in projects:
                for count in ("workflow_task_count", "workflow_round_count"):
                    if count not in project:
                        raise ValueError(f"Project fixture omitted {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--responses", type=Path)
    args = parser.parse_args()
    if args.responses:
        responses = json.loads(args.responses.read_text())
        validate_responses(responses)
        print(f"Validated {len(responses)} fixture responses against backend models and statuses")
        return 0
    expected = json.dumps(build_contract(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not SNAPSHOT.exists() or SNAPSHOT.read_text() != expected:
            print("Browser contract snapshot differs from the backend. Run scripts/generate_e2e_contract.py.")
            return 1
        print("Browser contract snapshot matches the backend")
    else:
        SNAPSHOT.write_text(expected)
        print(f"Wrote {SNAPSHOT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
