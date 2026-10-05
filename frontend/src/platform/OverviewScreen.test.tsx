// @vitest-environment jsdom

import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import OverviewScreen from "./OverviewScreen";
import ProjectPlatform from "./ProjectPlatform";
import { createAccessSnapshot } from "@/navigation/AccessContext";
import type { Project } from "@/types/api";
import {
  EMPTY_PLATFORM_PROJECT_DATA,
  type PlatformProjectData,
} from "./types";

afterEach(cleanup);

function readinessData(): PlatformProjectData {
  return {
    ...EMPTY_PLATFORM_PROJECT_DATA,
    projectModules: { project_id: 1, selected: ["data", "annotate", "train", "models"], effective: ["data", "annotate", "train", "models"], workspace_capabilities: ["annotation", "training", "inference"] },
    datasetVersions: [
      { id: 22, dataset_id: 21, version_number: 1, item_count: 3, provenance: {} },
      { id: 32, dataset_id: 31, version_number: 1, item_count: 3, provenance: { ingestion: "training_preparation_v1" } },
    ] as PlatformProjectData["datasetVersions"],
    rounds: [
      { id: 71, name: "My review", status: "open", sequence: 1, annotator_user_ids: [9], open_to_all_annotators: false },
      { id: 72, name: "Another review", status: "open", sequence: 2, annotator_user_ids: [8], open_to_all_annotators: false },
      { id: 73, name: "Closed review", status: "closed", sequence: 3, annotator_user_ids: [9], open_to_all_annotators: false },
    ] as PlatformProjectData["rounds"],
    trainingDatasets: [{ id: 151, name: "Prepared papers", version_number: 2 }] as PlatformProjectData["trainingDatasets"],
    taskVersions: [{ id: 12, task_kind: "classification" }] as PlatformProjectData["taskVersions"],
    modelVersions: [
      { id: 81, recipe_key: "bert", framework: "pytorch", checkpoint_package_id: 100, task_version_id: 12 },
      { id: 82, recipe_key: "tfidf_logistic_regression", framework: "scikit-learn", checkpoint_package_id: null, task_version_id: 12 },
      { id: 83, recipe_key: "tfidf_logistic_regression", framework: "scikit-learn", checkpoint_package_id: 101, task_version_id: 12 },
    ] as PlatformProjectData["modelVersions"],
  };
}

describe("OverviewScreen", () => {
  it("shows the staged research loop and deep-links model development", async () => {
    const onOpenTraining = vi.fn();
    const onOpenModels = vi.fn();
    const data: PlatformProjectData = {
      ...EMPTY_PLATFORM_PROJECT_DATA,
      projectModules: {
        project_id: 1,
        selected: [
          "data",
          "annotate",
          "learning",
          "train",
          "models",
          "guidelines",
          "activity",
        ],
        effective: [
          "data",
          "annotate",
          "learning",
          "train",
          "models",
          "guidelines",
          "activity",
        ],
        workspace_capabilities: [
          "annotation",
          "training",
          "inference",
          "active_learning",
          "co_learning",
          "lineage",
        ],
      },
    };

    render(
      <OverviewScreen
        data={data}
        documents={[]}
        assignments={[]}
        progress={null}
        onOpenData={vi.fn()}
        onOpenTraining={onOpenTraining}
        onOpenModels={onOpenModels}
      />,
    );

    expect(
      screen.getByRole("heading", { name: "Research loop roadmap" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("article", { name: "Inference roadmap" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("article", { name: "Active Learning roadmap" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("article", { name: "Co-learning roadmap" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("article", { name: "Lineage & Export roadmap" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("article", { name: "Guideline Learning roadmap" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("article", { name: "HPC & LLM Serving roadmap" }),
    ).toBeTruthy();
    expect(screen.getByText("Inference capability available")).toBeTruthy();
    expect(
      screen.getByText("Active learning capability available"),
    ).toBeTruthy();
    expect(screen.getAllByText("Not available in this workspace")).toHaveLength(
      1,
    );
    const roadmap = screen.getByLabelText(
      "Research loop capability roadmap",
    );
    expect(within(roadmap).queryByRole("link")).toBeNull();
    expect(within(roadmap).queryByRole("button")).toBeNull();
    screen.getByRole("link", { name: /Training/ }).click();
    screen.getByRole("link", { name: /Models/ }).click();
    expect(onOpenTraining).toHaveBeenCalledTimes(1);
    expect(onOpenModels).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("checkbox")).toBeNull();
  });

  it("continues only the current user's open round and launches the exact prepared training version", () => {
    const onContinueAnnotation = vi.fn(); const onTrainModel = vi.fn();
    render(<OverviewScreen data={readinessData()} documents={[]} assignments={[]} progress={null} currentUserId={9}
      onOpenData={vi.fn()} onContinueAnnotation={onContinueAnnotation} onSetupAnnotation={vi.fn()} onImport={vi.fn()}
      onTrainModel={onTrainModel} onCreateTraining={vi.fn()} onOpenInference={vi.fn()} />);
    const annotation = within(screen.getByRole("article", { name: "Annotate readiness" }));
    const resume = annotation.getByRole("link", { name: /Continue annotation/ });
    expect(resume.getAttribute("href")).toBe("/my-work/rounds/71");
    resume.click(); expect(onContinueAnnotation).toHaveBeenCalledWith(71);
    const train = within(screen.getByRole("article", { name: "Train readiness" })).getByRole("link", { name: /Train model/ });
    expect(train.getAttribute("href")).toBe("/training/new?projectId=1&trainingDatasetVersionId=151");
    train.click(); expect(onTrainModel).toHaveBeenCalledWith(151);
  });

  it("offers annotation setup for an existing source while predictions are ready without training labels", () => {
    const data = { ...readinessData(), rounds: [], trainingDatasets: [], labelSets: [] };
    const onSetupAnnotation = vi.fn();
    render(<OverviewScreen data={data} documents={[]} assignments={[]} progress={null} currentUserId={9}
      onOpenData={vi.fn()} onSetupAnnotation={onSetupAnnotation} onImport={vi.fn()} onOpenInference={vi.fn()} onCreateTraining={vi.fn()} />);
    const setup = within(screen.getByRole("article", { name: "Annotate readiness" })).getByRole("link", { name: /Set up annotation/ });
    expect(setup.getAttribute("href")).toBe("/projects/1/data?tab=source");
    setup.click(); expect(onSetupAnnotation).toHaveBeenCalledTimes(1);
    const inference = screen.getByRole("article", { name: "Run predictions readiness" });
    expect(inference.textContent).toContain("1 source dataset · 1 compatible TF-IDF model");
    expect(within(inference).getByRole("link", { name: /Run predictions/ }).getAttribute("href")).toBe("/projects/1/inference");
    expect(within(screen.getByRole("article", { name: "Train readiness" })).getByRole("link", { name: /Create training dataset/ })).toBeTruthy();
  });

  it("shows classification and NER separately and sets up the exact task that has no round", () => {
    const data = readinessData();
    data.taskDefinitions = [
      { id: 1, project_id: 1, key: "relevance", name: "Paper relevance", description: null },
      { id: 2, project_id: 1, key: "drugs", name: "Drug names", description: null },
    ];
    data.taskVersions = [
      { id: 12, task_definition_id: 1, version_number: 1, task_kind: "classification" },
      { id: 13, task_definition_id: 1, version_number: 2, task_kind: "classification" },
      { id: 14, task_definition_id: 2, version_number: 1, task_kind: "token_labeling" },
    ] as PlatformProjectData["taskVersions"];
    data.rounds = [{ ...data.rounds[0], task_version_id: 12 }];
    const onContinueAnnotation = vi.fn();
    const onSetupTask = vi.fn();
    render(<OverviewScreen data={data} documents={[]} assignments={[]} progress={null} currentUserId={9}
      onOpenData={vi.fn()} onContinueAnnotation={onContinueAnnotation} onSetupTask={onSetupTask} />);

    const choices = within(screen.getByRole("region", { name: "Project annotation tasks" }));
    expect(choices.getByText("Classification · v2")).toBeTruthy();
    expect(choices.getByText("Token labeling (tokenized data) · v1")).toBeTruthy();
    expect(choices.getByText("No annotation round yet")).toBeTruthy();
    expect(screen.getByText("3 saved task versions").parentElement?.textContent).toContain("Annotation tasks2");
    const annotation = within(screen.getByRole("article", { name: "Annotate readiness" }));
    const choose = annotation.getByRole("link", { name: /Choose annotation task/ });
    expect(choose.getAttribute("href")).toBe("#project-annotation-tasks");
    choose.click();
    expect(window.document.activeElement?.id).toBe("project-annotation-tasks");
    expect(onContinueAnnotation).not.toHaveBeenCalled();

    const continueClassification = choices.getByRole("link", { name: "Continue Paper relevance — My review" });
    expect(continueClassification.getAttribute("href")).toBe("/my-work/rounds/71");
    continueClassification.click();
    expect(onContinueAnnotation).toHaveBeenCalledWith(71);
    const setupNer = choices.getByRole("link", { name: "Set up Drug names" });
    expect(setupNer.getAttribute("href")).toBe("/projects/1/data?tab=source&taskVersionId=14");
    setupNer.click();
    expect(onSetupTask).toHaveBeenCalledWith(14);
  });

  it("keeps all assigned rounds visible instead of choosing the newest task silently", () => {
    const data = readinessData();
    data.taskDefinitions = [{ id: 1, project_id: 1, key: "relevance", name: "Paper relevance", description: null }];
    data.taskVersions = [{ id: 12, task_definition_id: 1, version_number: 1, task_kind: "classification" }] as PlatformProjectData["taskVersions"];
    data.rounds = [
      { ...data.rounds[0], task_version_id: 12 },
      { ...data.rounds[0], id: 74, name: "Second source", sequence: 4, task_version_id: 12 },
    ];
    const onContinueAnnotation = vi.fn();
    render(<OverviewScreen data={data} documents={[]} assignments={[]} progress={null} currentUserId={9}
      onOpenData={vi.fn()} onContinueAnnotation={onContinueAnnotation} />);
    expect(within(screen.getByRole("article", { name: "Annotate readiness" })).getByRole("link", { name: /Choose annotation task/ })).toBeTruthy();
    const choices = within(screen.getByRole("region", { name: "Project annotation tasks" }));
    choices.getByRole("link", { name: "Continue Paper relevance — My review" }).click();
    choices.getByRole("link", { name: "Continue Paper relevance — Second source" }).click();
    expect(onContinueAnnotation.mock.calls).toEqual([[71], [74]]);
  });

  it("does not offer annotation or inference actions to a trainer without those capabilities", () => {
    const project: Project = { id: 1, name: "Papers", description: null, workspace_id: 1, tasks: [], settings: {}, annotation_schema: { labels: {} }, annotation_validation_mode: "strict" };
    const access = createAccessSnapshot({ workspaceId: 1, workspaceKind: "team", membershipRole: "trainer", effectiveCapabilities: ["training"] });
    render(<ProjectPlatform pathname="/projects/1/overview" project={project} projects={[project]} data={readinessData()} documents={[]} assignments={[]} progress={null}
      currentUserId={9} access={access} loading={false} busy={false} error={null} dialog={null} onDialogChange={vi.fn()} onProjectSelect={vi.fn()} onNavigate={vi.fn()} onOpenRound={vi.fn()}
      onUpdateProject={async () => undefined} onUpdateModules={async () => undefined} onRefresh={async () => undefined} />);
    expect(within(screen.getByRole("article", { name: "Annotate readiness" })).queryByRole("link")).toBeNull();
    expect(within(screen.getByRole("article", { name: "Run predictions readiness" })).queryByRole("link")).toBeNull();
    expect(within(screen.getByRole("article", { name: "Train readiness" })).getByRole("link", { name: /Train model/ })).toBeTruthy();
    expect(screen.queryByRole("link", { name: /Import PMIDs/ })).toBeNull();
  });
});
