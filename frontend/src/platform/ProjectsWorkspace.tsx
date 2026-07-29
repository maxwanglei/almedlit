import { useId, useMemo, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { X } from "lucide-react";

import { createProject } from "@/api/client";
import DialogFrame from "@/components/DialogFrame";
import type { Project } from "@/types/api";

import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformRouteLink,
  PlatformStatus,
} from "./components";

type ProjectTemplate = "annotation" | "training" | "combined";

const TEMPLATES: Array<{
  id: ProjectTemplate;
  label: string;
  description: string;
  modules: string[];
  capability: "annotation" | "training";
}> = [
  {
    id: "annotation",
    label: "Annotation",
    description: "Data, tasks, rounds, review, and activity.",
    modules: ["data", "annotate", "activity"],
    capability: "annotation",
  },
  {
    id: "training",
    label: "Training only",
    description: "Uploaded or public data, named models, and training runs.",
    modules: ["data", "models", "train", "activity"],
    capability: "training",
  },
  {
    id: "combined",
    label: "Annotation + training",
    description: "One project context for the complete labeled-data workflow.",
    modules: ["data", "annotate", "models", "train", "activity"],
    capability: "annotation",
  },
];

function projectPurpose(project: Project): "training" | "annotation" | "combined" {
  const configured = project.settings.modules;
  const modules = new Set(Array.isArray(configured) ? configured : []);
  if (modules.has("train") && !modules.has("annotate")) {
    return "training";
  }
  if (modules.has("train") && modules.has("annotate")) {
    return "combined";
  }
  return "annotation";
}

export default function ProjectsWorkspace({
  workspaceId,
  projects,
  canCreate,
  capabilities,
  loading,
  onOpenProject,
  onProjectCreated,
}: {
  workspaceId: number | null;
  projects: Project[];
  canCreate: boolean;
  capabilities: readonly string[];
  loading: boolean;
  onOpenProject: (projectId: number) => void;
  onProjectCreated: (project: Project) => void;
}): React.ReactElement {
  const dialogTitleId = useId();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [template, setTemplate] = useState<ProjectTemplate>("annotation");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const effectiveCapabilities = useMemo(
    () => new Set(capabilities),
    [capabilities],
  );

  function templateIsEnabled(item: (typeof TEMPLATES)[number]): boolean {
    if (item.id === "combined") {
      return (
        effectiveCapabilities.has("annotation") &&
        effectiveCapabilities.has("training")
      );
    }
    return effectiveCapabilities.has(item.capability);
  }

  function openDialog(): void {
    const selectedTemplate = TEMPLATES.find((item) => item.id === template);
    if (!selectedTemplate || !templateIsEnabled(selectedTemplate)) {
      setTemplate(
        TEMPLATES.find((item) => templateIsEnabled(item))?.id ?? "annotation",
      );
    }
    setError(null);
    setDialogOpen(true);
  }

  function closeDialog(): void {
    if (busy) {
      return;
    }
    setDialogOpen(false);
    setError(null);
  }

  async function submit(): Promise<void> {
    if (workspaceId === null || !name.trim()) {
      return;
    }
    const selectedTemplate = TEMPLATES.find((item) => item.id === template);
    if (!selectedTemplate || !templateIsEnabled(selectedTemplate)) {
      setError("This project template is not enabled for the workspace.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const created = await createProject({
        workspace_id: workspaceId,
        name: name.trim(),
        description: description.trim() || null,
        settings: {
          modules: selectedTemplate.modules,
        },
      });
      onProjectCreated(created);
      setDialogOpen(false);
      setName("");
      setDescription("");
      setTemplate("annotation");
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "Could not create project",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main id="main-content" className="module-workspace-main" tabIndex={-1}>
      <div className="platform-page projects-directory">
        <PlatformPageHeader
          title="Projects"
          description="Scientific contexts for data, annotation, training, evaluation, and reproducible lineage."
          actionLabel={canCreate ? "New project" : undefined}
          onAction={
            canCreate ? openDialog : undefined
          }
        />

        {loading ? (
          <div className="platform-loading" role="status" aria-live="polite">
            <span aria-hidden="true" />
            Loading projects…
          </div>
        ) : projects.length ? (
          <div
            className="platform-table-scroll platform-table-scroll--summary projects-directory-table"
            role="region"
            aria-label="Projects table"
            tabIndex={0}
          >
            <table className="platform-table platform-table--summary">
              <thead>
                <tr>
                  <th scope="col">Project</th>
                  <th scope="col">Purpose</th>
                  <th scope="col">Tasks</th>
                  <th scope="col">Workspace</th>
                  <th scope="col"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {projects.map((project) => {
                  const purpose = projectPurpose(project);
                  return (
                    <tr key={project.id}>
                      <td data-label="Project" data-priority="identity">
                        <strong>{project.name}</strong>
                        <span>{project.description ?? "No description"}</span>
                      </td>
                      <td data-label="Purpose" data-priority="status">
                        <PlatformStatus
                          value={
                            purpose === "training"
                              ? "training only"
                              : purpose
                          }
                        />
                      </td>
                      <td data-label="Tasks">{project.tasks.length}</td>
                      <td data-label="Workspace">#{project.workspace_id}</td>
                      <td data-label="Action" data-priority="action">
                        <PlatformRouteLink
                          href={`/projects/${project.id}/overview`}
                          className="platform-text-action"
                          onNavigate={() => onOpenProject(project.id)}
                        >
                          Open
                        </PlatformRouteLink>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <PlatformEmpty
            title="No projects yet"
            detail={
              canCreate
                ? "Create an annotation, training-only, or combined project."
                : "A workspace manager must create a project before work can begin."
            }
            actionLabel={canCreate ? "Create project" : undefined}
            onAction={canCreate ? openDialog : undefined}
          />
        )}
      </div>

      {dialogOpen ? (
        <DialogFrame
          busy={busy}
          error={error}
          labelledBy={dialogTitleId}
          backdropClassName="platform-dialog-backdrop"
          dialogClassName="platform-dialog"
          dialogElement="section"
          initialFocusSelector="form input:not([disabled])"
          onDismiss={closeDialog}
        >
          <header>
            <div>
              <h2 id={dialogTitleId}>Create project</h2>
            </div>
            <Button
              label="Close"
              icon={<X size={17} />}
              isIconOnly
              variant="ghost"
              isDisabled={busy}
              onClick={closeDialog}
            />
          </header>
          {error ? (
            <p className="platform-dialog-error" role="alert">
              {error}
            </p>
          ) : null}
          <form
            className="platform-dialog-form"
            onSubmit={(event) => {
              event.preventDefault();
              void submit();
            }}
          >
            <label>
              <span>Project name</span>
              <input
                autoFocus
                required
                maxLength={255}
                value={name}
                onChange={(event) => setName(event.target.value)}
              />
            </label>
            <label>
              <span>Description</span>
              <textarea
                rows={3}
                value={description}
                onChange={(event) => setDescription(event.target.value)}
              />
            </label>
            <fieldset className="platform-form-fieldset">
              <legend>Project template</legend>
              <div className="platform-option-list">
                {TEMPLATES.map((item) => {
                  const enabled = templateIsEnabled(item);
                  return (
                    <label key={item.id} data-disabled={!enabled}>
                      <input
                        type="radio"
                        name="project-template"
                        value={item.id}
                        checked={template === item.id}
                        disabled={!enabled}
                        onChange={() => setTemplate(item.id)}
                      />
                      <span>
                        <strong>{item.label}</strong>
                        <small>{item.description}</small>
                      </span>
                    </label>
                  );
                })}
              </div>
            </fieldset>
            <div className="platform-dialog-actions">
              <Button label="Cancel" onClick={closeDialog} isDisabled={busy} />
              <Button
                label={busy ? "Creating…" : "Create project"}
                variant="primary"
                type="submit"
                isDisabled={
                  busy ||
                  !name.trim() ||
                  !TEMPLATES.some(
                    (item) => item.id === template && templateIsEnabled(item),
                  )
                }
              />
            </div>
          </form>
        </DialogFrame>
      ) : null}
    </main>
  );
}
