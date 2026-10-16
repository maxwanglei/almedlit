// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Document, EvidenceTarget, GuidelineVersion, ProjectTask, TaskAssignment, WorkspaceMember } from "@/types/api";

import DocumentAssignmentManagement, { DocumentTaskManagement } from "./DocumentAssignmentManagement";

const mocks = vi.hoisted(() => ({ createTask: vi.fn(), assign: vi.fn(), update: vi.fn(), updateTask: vi.fn() }));
vi.mock("@/api/client", () => ({
  createProjectTask: mocks.createTask,
  createTaskAssignment: mocks.assign,
  updateTaskAssignment: mocks.update,
  updateProjectTask: mocks.updateTask,
}));

function task(id = 51, annotation_type: ProjectTask["annotation_type"] = "evidence_block", enabled = true): ProjectTask {
  return { id, project_id: 7, annotation_type, display_name: `Task ${id}`, description: null, enabled, sort_order: 0, labels: [], settings: {} };
}
function document(id = 41, structure: number | null = 501): Document {
  return { id, project_id: 7, external_id: null, title: `Paper ${id}`, text: "Example paper.", source: null, metadata_: {}, sentences: [], active_structure_version_id: structure };
}
function member(id = 9, active = true): WorkspaceMember {
  return { id, workspace_id: 2, user_id: id, username: `user${id}`, display_name: `Annotator ${id}`, email: null, is_active: active, role: "annotator" };
}
function target(id = 61, taskId = 51, active = true): EvidenceTarget {
  return { id, project_id: 7, task_id: taskId, key: `target${id}`, name: `Target ${id}`, description: null, is_active: active,
    active_version_id: id * 10 + 1, versions: [{ id: id * 10 + 1, target_id: id, version_number: 1, text: "Identify treatment evidence.", guidance: null, inclusion_guidance: null, exclusion_guidance: null, metadata_: {}, created_by_user_id: 1, created_at: "2026-01-01", updated_at: "2026-01-01" }],
    created_by_user_id: 1, created_at: "2026-01-01", updated_at: "2026-01-01" };
}
function guideline(id = 8, status = "active"): GuidelineVersion {
  return { id, project_id: 7, version_label: `v${id}`, markdown: "Review the full paper.", author_id: null, status };
}
function assignment(status: TaskAssignment["status"] = "assigned"): TaskAssignment {
  return { id: 71, project_id: 7, task_id: 51, document_id: 41, assignee_user_id: 9, annotator_id: "user9", status, assigned_by_user_id: 1, assigned_by: "manager", notes: null, metadata_: {}, target_version_id: 611, structure_version_id: 501, guideline_version_id: 8, assignment_scope_key: "target:611:structure:501:guideline:8" };
}
function props() {
  return { projectId: 7, tasks: [task()], documents: [document()], members: [member()], targets: [target()], guidelines: [guideline()], assignments: [] as TaskAssignment[], onChanged: vi.fn().mockResolvedValue(undefined), onOpenAssignment: vi.fn() };
}
async function fillAssignment(user: ReturnType<typeof userEvent.setup>) {
  await user.selectOptions(screen.getByLabelText("Document task"), "51");
  await user.selectOptions(screen.getByLabelText("Document"), "41");
  await user.selectOptions(screen.getByLabelText("Annotator"), "9");
  await user.selectOptions(screen.getByLabelText("Evidence target version"), "611");
}

beforeEach(() => {
  mocks.createTask.mockReset().mockResolvedValue(task());
  mocks.assign.mockReset().mockResolvedValue(assignment());
  mocks.update.mockReset().mockResolvedValue(assignment("blocked"));
  mocks.updateTask.mockReset().mockResolvedValue(task());
});
afterEach(cleanup);

describe("document task and assignment management", () => {
  it("creates an evidence task through the project task API and refreshes the controlled data", async () => {
    const user = userEvent.setup();
    const input = { ...props(), tasks: [] };
    render(<DocumentTaskManagement {...input} />);
    await user.type(screen.getByLabelText("Task name"), "Treatment evidence");
    await user.click(screen.getByRole("button", { name: "Create document task" }));

    expect(mocks.createTask).toHaveBeenCalledExactlyOnceWith(7, expect.objectContaining({ annotation_type: "evidence_block", display_name: "Treatment evidence", enabled: true }));
    await waitFor(() => expect(input.onChanged).toHaveBeenCalledOnce());
  });

  it("validates entity labels before posting and retains a failed task draft for correction", async () => {
    const user = userEvent.setup();
    render(<DocumentTaskManagement {...props()} />);
    await user.selectOptions(screen.getByLabelText("Annotation type"), "entity");
    await user.type(screen.getByLabelText("Task name"), "Clinical entities");
    const labels = screen.getByLabelText<HTMLInputElement>("Task labels");
    await user.type(labels, ", ,");
    await act(async () => { fireEvent.submit(labels.form!); });
    expect(mocks.createTask).not.toHaveBeenCalled();
    expect((await screen.findByRole("alert")).textContent).toMatch(/label/i);
    mocks.createTask.mockRejectedValueOnce(new Error("Task save failed"));
    await user.clear(labels);
    await user.type(labels, "Drug, Disease");
    await user.click(screen.getByRole("button", { name: "Create document task" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Task save failed");
    expect(screen.getByLabelText<HTMLInputElement>("Task name").value).toBe("Clinical entities");
    expect(labels.value).toBe("Drug, Disease");
    await user.click(screen.getByRole("button", { name: "Create document task" }));
    await waitFor(() => expect(mocks.createTask).toHaveBeenCalledTimes(2));
    expect(mocks.createTask.mock.calls[1]?.[1].labels.map((label: { name: string }) => label.name)).toEqual(["Drug", "Disease"]);
  });

  it("pins evidence assignments to the active target, document structure, and newest active guideline", async () => {
    const user = userEvent.setup();
    const input = { ...props(), guidelines: [guideline(7), guideline(8), guideline(9, "draft")] };
    render(<DocumentAssignmentManagement {...input} />);
    await fillAssignment(user);
    await user.type(screen.getByLabelText("Assignment notes"), "  Review methods and results.  ");
    await user.click(screen.getByRole("button", { name: "Assign document" }));

    expect(mocks.assign).toHaveBeenCalledExactlyOnceWith(7, {
      task_id: 51, document_id: 41, assignee_user_id: 9, target_version_id: 611,
      structure_version_id: 501, guideline_version_id: 8, notes: "Review methods and results.",
    });
    await waitFor(() => expect(input.onChanged).toHaveBeenCalledOnce());
  });

  it("excludes disabled tasks, foreign documents, inactive members, and inactive or unrelated targets", async () => {
    const user = userEvent.setup();
    const input = { ...props(), tasks: [task(), task(52, "entity", false)], documents: [document(), { ...document(42), project_id: 999 }], members: [member(), member(10, false)], targets: [target(), target(62, 51, false), target(63, 99), { ...target(64), active_version_id: null }] };
    render(<DocumentAssignmentManagement {...input} />);
    expect(within(screen.getByLabelText("Document task")).queryByRole("option", { name: "Task 52" })).toBeNull();
    expect(screen.queryByRole("option", { name: "Paper 42" })).toBeNull();
    expect(screen.queryByRole("option", { name: /Annotator 10/ })).toBeNull();
    await user.selectOptions(screen.getByLabelText("Document task"), "51");
    expect(within(screen.getByLabelText("Evidence target version")).getAllByRole("option").map((option) => option.getAttribute("value"))).toEqual(["0", "611"]);
  });

  it.each(["guideline", "structure"])("rejects evidence assignment without an active %s before posting", async (missing) => {
    const user = userEvent.setup();
    render(<DocumentAssignmentManagement {...props()} guidelines={missing === "guideline" ? [] : [guideline()]} documents={missing === "structure" ? [document(41, null)] : [document()]} />);
    await fillAssignment(user);
    const notes = screen.getByLabelText<HTMLTextAreaElement>("Assignment notes");
    await act(async () => { fireEvent.submit(notes.form!); });

    expect(mocks.assign).not.toHaveBeenCalled();
    expect((await screen.findByRole("alert")).textContent).toMatch(missing === "guideline" ? /active.*guideline/i : /structure|process.*document/i);
  });

  it("revalidates a target deactivated while an assignment draft is open", async () => {
    const user = userEvent.setup();
    const input = props();
    const view = render(<DocumentAssignmentManagement {...input} />);
    await fillAssignment(user);
    view.rerender(<DocumentAssignmentManagement {...input} targets={[target(61, 51, false)]} />);
    await act(async () => { fireEvent.submit(screen.getByLabelText<HTMLTextAreaElement>("Assignment notes").form!); });

    expect(mocks.assign).not.toHaveBeenCalled();
    expect((await screen.findByRole("alert")).textContent).toMatch(/active.*target/i);
  });

  it("retains an assignment draft after failure and allows a retry", async () => {
    const user = userEvent.setup();
    mocks.assign.mockRejectedValueOnce(new Error("Assignment save failed"));
    render(<DocumentAssignmentManagement {...props()} />);
    await fillAssignment(user);
    await user.type(screen.getByLabelText("Assignment notes"), "Review all sentences");
    await user.click(screen.getByRole("button", { name: "Assign document" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Assignment save failed");
    expect(screen.getByLabelText<HTMLTextAreaElement>("Assignment notes").value).toBe("Review all sentences");
    expect(screen.getByLabelText<HTMLSelectElement>("Evidence target version").value).toBe("611");
    await user.click(screen.getByRole("button", { name: "Assign document" }));
    await waitFor(() => expect(mocks.assign).toHaveBeenCalledTimes(2));
  });

  it("shows immutable pins, opens an assignment, and updates its status without changing its scope", async () => {
    const user = userEvent.setup();
    const input = { ...props(), assignments: [assignment()] };
    render(<DocumentAssignmentManagement {...input} />);
    expect(screen.getByText(/Target version 611/).textContent).toContain("structure 501; guideline 8");
    await user.click(screen.getByRole("button", { name: "Open assignment 71" }));
    expect(input.onOpenAssignment).toHaveBeenCalledExactlyOnceWith(input.assignments[0]);
    await user.click(screen.getByRole("button", { name: "Block assignment 71" }));
    expect(mocks.update).toHaveBeenCalledExactlyOnceWith(7, 71, { status: "blocked" });
    await waitFor(() => expect(input.onChanged).toHaveBeenCalledOnce());
  });

  it("disables writes while parent data is loading", () => {
    render(<><DocumentTaskManagement {...props()} busy /><DocumentAssignmentManagement {...props()} busy /></>);
    expect(screen.getByRole<HTMLButtonElement>("button", { name: "Assign document" }).disabled).toBe(true);
    expect(screen.getByRole<HTMLButtonElement>("button", { name: "Create document task" }).disabled).toBe(true);
  });

  it("enables an existing disabled task instead of offering a duplicate type", async () => {
    const user = userEvent.setup();
    const input = { ...props(), tasks: [task(51, "evidence_block", false)] };
    render(<DocumentTaskManagement {...input} />);
    expect(within(screen.getByLabelText("Annotation type")).queryByRole("option", { name: "Evidence blocks" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Enable task 51" }));

    expect(mocks.updateTask).toHaveBeenCalledExactlyOnceWith(7, 51, { enabled: true });
    expect(mocks.createTask).not.toHaveBeenCalled();
    await waitFor(() => expect(input.onChanged).toHaveBeenCalledOnce());
  });

  it("uses an existing assignment with the same pinned scope instead of posting a duplicate", async () => {
    const user = userEvent.setup();
    render(<DocumentAssignmentManagement {...props()} assignments={[assignment()]} />);
    await fillAssignment(user);
    await user.click(screen.getByRole("button", { name: "Assign document" }));

    expect(mocks.assign).not.toHaveBeenCalled();
    expect((await screen.findByRole("alert")).textContent).toContain("Assignment 71 already exists");
  });

  it("assigns entity tasks without requiring an evidence target or guideline", async () => {
    const user = userEvent.setup();
    render(<DocumentAssignmentManagement {...props()} tasks={[task(51, "entity")]} targets={[]} guidelines={[]} />);
    await user.selectOptions(screen.getByLabelText("Document task"), "51");
    await user.selectOptions(screen.getByLabelText("Document"), "41");
    await user.selectOptions(screen.getByLabelText("Annotator"), "9");
    expect(screen.queryByLabelText("Evidence target version")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Assign document" }));

    expect(mocks.assign).toHaveBeenCalledExactlyOnceWith(7, { task_id: 51, document_id: 41, assignee_user_id: 9, notes: null });
  });

  it("remembers a successful assignment when refreshing fails and prevents a second POST", async () => {
    const user = userEvent.setup();
    const input = props();
    input.onChanged.mockRejectedValue(new Error("Refresh failed"));
    render(<DocumentAssignmentManagement {...input} />);
    await fillAssignment(user);
    await user.click(screen.getByRole("button", { name: "Assign document" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Saved, but the project could not refresh");
    expect(screen.getByText(/Target version 611/)).toBeTruthy();
    expect(screen.getByRole<HTMLButtonElement>("button", { name: "Assign document" }).disabled).toBe(true);
    input.onChanged.mockResolvedValue(undefined);
    await user.click(screen.getByRole("button", { name: "Retry refresh" }));
    await user.click(screen.getByRole("button", { name: "Assign document" }));

    expect(mocks.assign).toHaveBeenCalledOnce();
    expect((await screen.findByRole("alert")).textContent).toContain("Assignment 71 already exists");
  });

  it.each(["   ", "x".repeat(121)])("rejects an invalid task name before posting", async (name) => {
    render(<DocumentTaskManagement {...props()} tasks={[]} />);
    const field = screen.getByLabelText<HTMLInputElement>("Task name");
    fireEvent.change(field, { target: { value: name } });
    await act(async () => { fireEvent.submit(field.form!); });

    expect(mocks.createTask).not.toHaveBeenCalled();
    expect((await screen.findByRole("alert")).textContent).toMatch(/task name/i);
  });

  it("retries refreshing a confirmed task toggle without repeating its PATCH", async () => {
    const user = userEvent.setup();
    const input = { ...props(), tasks: [task(51, "evidence_block", false)] };
    input.onChanged.mockRejectedValueOnce(new Error("Refresh failed"));
    const view = render(<DocumentTaskManagement {...input} />);
    await user.click(screen.getByRole("button", { name: "Enable task 51" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Saved, but the project could not refresh");
    expect(screen.getByRole<HTMLButtonElement>("button", { name: "Enable task 51" }).disabled).toBe(true);
    input.onChanged.mockImplementationOnce(async () => { view.rerender(<DocumentTaskManagement {...input} tasks={[task()]} />); });

    await user.click(screen.getByRole("button", { name: "Retry refresh" }));

    expect(await screen.findByRole("button", { name: "Disable task 51" })).toBeTruthy();
    expect(mocks.updateTask).toHaveBeenCalledOnce();
    expect(input.onChanged).toHaveBeenCalledTimes(2);
  });

  it("retries refreshing a confirmed status update without repeating its PATCH", async () => {
    const user = userEvent.setup();
    const input = { ...props(), assignments: [assignment()] };
    input.onChanged.mockRejectedValueOnce(new Error("Refresh failed"));
    const view = render(<DocumentAssignmentManagement {...input} />);
    await user.click(screen.getByRole("button", { name: "Block assignment 71" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Saved, but the project could not refresh");
    expect(screen.getByRole<HTMLButtonElement>("button", { name: "Block assignment 71" }).disabled).toBe(true);
    input.onChanged.mockImplementationOnce(async () => { view.rerender(<DocumentAssignmentManagement {...input} assignments={[assignment("blocked")]} />); });

    await user.click(screen.getByRole("button", { name: "Retry refresh" }));

    expect(await screen.findByRole("button", { name: "Resume assignment 71" })).toBeTruthy();
    expect(mocks.update).toHaveBeenCalledOnce();
    expect(input.onChanged).toHaveBeenCalledTimes(2);
  });

  it("explicitly closes an incomplete open evidence assignment while preserving its pinned history", async () => {
    const user = userEvent.setup();
    const input = { ...props(), assignments: [{ ...assignment(), guideline_version_id: null }] };
    mocks.update.mockResolvedValue({ ...input.assignments[0], status: "completed" });
    render(<DocumentAssignmentManagement {...input} />);
    expect(screen.getByText(/This assignment is missing pinned versions/).textContent).toContain("recorded work is retained");

    await user.click(screen.getByRole("button", { name: "Close incomplete assignment 71" }));

    expect(mocks.update).toHaveBeenCalledExactlyOnceWith(7, 71, { status: "completed" });
    await waitFor(() => expect(input.onChanged).toHaveBeenCalledOnce());
    expect(input.assignments[0].guideline_version_id).toBeNull();
  });

  it.each([
    { label: "finalized evidence", task: task(), assignment: { ...assignment("submitted"), guideline_version_id: null } },
    { label: "fully pinned evidence", task: task(), assignment: assignment() },
    { label: "entity annotation", task: task(51, "entity"), assignment: { ...assignment(), target_version_id: null, guideline_version_id: null } },
  ])("does not offer incomplete-assignment closure for $label", (input) => {
    render(<DocumentAssignmentManagement {...props()} tasks={[input.task]} assignments={[input.assignment]} />);

    expect(screen.queryByRole("button", { name: /Close incomplete assignment/ })).toBeNull();
  });
});
