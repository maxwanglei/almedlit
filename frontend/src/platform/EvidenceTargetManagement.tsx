import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Button } from "@astryxdesign/core/Button";

import { activateEvidenceTarget, createEvidenceTarget, createEvidenceTargetVersion, deactivateEvidenceTarget } from "@/api/client";
import type { EvidenceTarget, EvidenceTargetVersion, EvidenceTargetVersionCreate, ProjectTask } from "@/types/api";

import { PlatformSection } from "./components";

export interface EvidenceTargetManagementProps {
  projectId: number;
  tasks: ProjectTask[];
  targets: EvidenceTarget[];
  busy?: boolean;
  activationBlockedReason?: string;
  onChanged: () => Promise<void>;
}

function hasSavedState(incoming: EvidenceTarget, saved: EvidenceTarget): boolean {
  return incoming.is_active === saved.is_active && incoming.active_version_id === saved.active_version_id
    && saved.versions.every((version) => incoming.versions.some((item) => item.id === version.id));
}

export default function EvidenceTargetManagement({ projectId, tasks, targets, busy = false,
  activationBlockedReason, onChanged }: EvidenceTargetManagementProps): React.ReactElement {
  const evidenceTasks = tasks.filter((task) => task.project_id === projectId && task.annotation_type === "evidence_block");
  const [taskId, setTaskId] = useState<number | null>(null);
  const selectedTask = evidenceTasks.find((task) => task.id === taskId) ?? evidenceTasks.find((task) => task.enabled);
  const [key, setKey] = useState("");
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [guidance, setGuidance] = useState("");
  const [savedTargets, setSavedTargets] = useState(new Map<number, EvidenceTarget>());
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const inFlight = useRef(false);
  const mounted = useRef(true);
  const activeProject = useRef(projectId);
  activeProject.current = projectId;
  const keyHelpId = useId();
  const locked = busy || saving;

  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    // Keep a successful write visible until a refresh includes it. A failed
    // refresh must not make a created target or version look unsaved.
    setSavedTargets((current) => {
      const next = new Map(current);
      for (const incoming of targets) {
        const saved = current.get(incoming.id);
        if (saved && (hasSavedState(incoming, saved) || Date.parse(incoming.updated_at) > Date.parse(saved.updated_at))) {
          next.delete(incoming.id);
        }
      }
      return next.size === current.size ? current : next;
    });
  }, [targets]);

  const visibleTargets = useMemo(() => {
    const merged = new Map(targets.map((target) => [target.id, target]));
    savedTargets.forEach((target, id) => merged.set(id, target));
    return [...merged.values()].filter((target) => target.project_id === projectId
      && evidenceTasks.some((task) => task.id === target.task_id));
  }, [evidenceTasks, projectId, savedTargets, targets]);

  function remember(target: EvidenceTarget): void {
    setSavedTargets((current) => new Map(current).set(target.id, target));
  }

  async function mutate<T>(operation: () => Promise<T>, commit: (result: T) => void, message: string): Promise<T | null> {
    if (busy || inFlight.current) return null;
    const initiatingProject = projectId;
    inFlight.current = true;
    setSaving(true); setError(null); setRefreshError(false); setStatus(null);
    try {
      const result = await operation();
      if (!mounted.current || activeProject.current !== initiatingProject) return null;
      commit(result);
      setStatus(message);
      try { await onChanged(); }
      catch {
        if (mounted.current && activeProject.current === initiatingProject) setRefreshError(true);
      }
      return result;
    } catch (caught) {
      if (mounted.current && activeProject.current === initiatingProject) {
        setError(caught instanceof Error ? caught.message : "The evidence target could not be saved.");
      }
      return null;
    } finally {
      inFlight.current = false;
      if (mounted.current && activeProject.current === initiatingProject) setSaving(false);
    }
  }

  async function createTarget(): Promise<void> {
    if (!selectedTask?.enabled) { setError("Choose an enabled evidence task."); return; }
    if (!/^[a-z0-9][a-z0-9_-]{0,119}$/.test(key.trim()) || !name.trim() || !text.trim()) {
      setError("Enter a valid target key, name, and target text."); return;
    }
    await mutate(() => createEvidenceTarget(projectId, {
      task_id: selectedTask.id, key: key.trim(), name: name.trim(),
      initial_version: { text: text.trim(), guidance: guidance.trim() || null },
    }), (created) => {
      remember(created); setKey(""); setName(""); setText(""); setGuidance("");
    }, "Target created. Choose a version and activate it when ready.");
  }

  async function createVersion(target: EvidenceTarget, payload: EvidenceTargetVersionCreate): Promise<EvidenceTargetVersion | null> {
    if (!payload.text.trim()) { setError("Enter new target text."); return null; }
    return mutate(() => createEvidenceTargetVersion(projectId, target.id, payload), (created) => {
      remember({ ...target, updated_at: created.updated_at, versions: [...target.versions, created] });
    }, "New target version created. Activate it when ready; earlier versions remain saved.");
  }

  async function retryRefresh(): Promise<void> {
    if (busy || inFlight.current) return;
    inFlight.current = true; setSaving(true);
    try { await onChanged(); if (mounted.current) setRefreshError(false); }
    catch { if (mounted.current) setRefreshError(true); }
    finally { inFlight.current = false; if (mounted.current) setSaving(false); }
  }

  return <PlatformSection title="Evidence targets" description="Define the evidence to find, then activate the version annotators should use. Earlier versions remain available for review.">
    {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
    {status ? <p role="status" className="platform-form-success">{status}</p> : null}
    {refreshError ? <div role="alert" className="platform-form-warning">
      <p>Your changes were saved, but the evidence setup could not refresh.</p>
      <Button label="Retry refresh" isDisabled={locked} onClick={() => void retryRefresh()} />
    </div> : null}
    {activationBlockedReason ? <p className="platform-form-note">{activationBlockedReason}</p> : null}
    {visibleTargets.map((target) => <TargetCard key={target.id} target={target}
      task={evidenceTasks.find((task) => task.id === target.task_id)!} busy={locked}
      activationBlockedReason={activationBlockedReason}
      onActivate={(versionId) => void mutate(() => activateEvidenceTarget(projectId, target.id, versionId), remember, "Target version activated.")}
      onDeactivate={() => void mutate(() => deactivateEvidenceTarget(projectId, target.id), remember, "Target deactivated.")}
      onCreateVersion={(payload) => createVersion(target, payload)} />)}
    {!visibleTargets.length ? <p className="platform-inline-empty">No evidence targets yet.</p> : null}
    <form className="platform-dialog-form" aria-label="Create evidence target" onSubmit={(event) => { event.preventDefault(); void createTarget(); }}>
      <h3>Create evidence target</h3>
      <div className="platform-form-grid">
        <label><span>Evidence task</span><select value={selectedTask?.id ?? ""} onChange={(event) => setTaskId(Number(event.target.value))} disabled={locked} required>
          {!selectedTask ? <option value="">Choose an enabled evidence task</option> : null}
          {evidenceTasks.map((task) => <option key={task.id} value={task.id} disabled={!task.enabled}>{task.display_name}{!task.enabled ? " (disabled)" : ""}</option>)}
        </select></label>
        <label><span>Target key</span><input value={key} onChange={(event) => setKey(event.target.value)} required maxLength={120}
          pattern="[a-z0-9][a-z0-9_-]*" aria-describedby={keyHelpId} disabled={locked} /></label>
      </div>
      <small id={keyHelpId}>Use a stable key with lowercase letters, numbers, hyphens, or underscores.</small>
      <label><span>Target name</span><input value={name} onChange={(event) => setName(event.target.value)} required maxLength={255} disabled={locked} /></label>
      <label><span>Target text</span><textarea value={text} onChange={(event) => setText(event.target.value)} required rows={3} disabled={locked} /></label>
      <label><span>Target guidance</span><textarea value={guidance} onChange={(event) => setGuidance(event.target.value)} rows={2} disabled={locked} /></label>
      <div className="platform-page-actions"><Button label="Create target" type="submit" variant="primary" isDisabled={locked || !selectedTask?.enabled} /></div>
    </form>
  </PlatformSection>;
}

function TargetCard({ target, task, busy, activationBlockedReason, onActivate, onDeactivate, onCreateVersion }: {
  target: EvidenceTarget;
  task: ProjectTask;
  busy: boolean;
  activationBlockedReason?: string;
  onActivate: (versionId: number) => void;
  onDeactivate: () => void;
  onCreateVersion: (payload: EvidenceTargetVersionCreate) => Promise<EvidenceTargetVersion | null>;
}): React.ReactElement {
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const versions = [...target.versions].sort((left, right) => right.version_number - left.version_number);
  const selected = versions.find((version) => version.id === selectedId)
    ?? versions.find((version) => version.id === target.active_version_id) ?? versions[0];
  const active = versions.find((version) => target.is_active && version.id === target.active_version_id);
  const [text, setText] = useState("");
  const [guidance, setGuidance] = useState("");

  return <article aria-label={`Evidence target ${target.name}`} className="platform-form-fieldset">
    <div className="platform-dialog-form">
      <h3>{target.name}</h3>
      <p>{task.display_name} · <code>{target.key}</code> · <strong>{active ? `Active · v${active.version_number}` : "Inactive"}</strong></p>
      {!task.enabled ? <p className="platform-form-note">Enable this evidence task before activating a target.</p> : null}
      <label><span>Target version</span><select value={selected?.id ?? ""} disabled={busy || versions.length === 0} onChange={(event) => setSelectedId(Number(event.target.value))}>
        {versions.map((version) => <option key={version.id} value={version.id}>Version {version.version_number}{active?.id === version.id ? " (active)" : ""}</option>)}
      </select></label>
      {selected ? <div><p>{selected.text}</p>{selected.guidance ? <p>{selected.guidance}</p> : null}</div> : null}
      <div className="platform-page-actions">
        <Button label="Activate version" isDisabled={busy || !selected || !task.enabled || Boolean(activationBlockedReason) || active?.id === selected?.id}
          onClick={() => { if (selected) onActivate(selected.id); }} />
        {target.is_active ? <Button label="Deactivate target" isDisabled={busy} onClick={onDeactivate} /> : null}
      </div>
      <details><summary>Create a new version</summary>
        <form className="platform-dialog-form" onSubmit={async (event) => {
          event.preventDefault();
          const created = await onCreateVersion({ text: text.trim(), guidance: guidance.trim() || null });
          if (created) { setSelectedId(created.id); setText(""); setGuidance(""); }
        }}>
          <p>Creates a new version without changing earlier versions or the active selection.</p>
          <label><span>New target text</span><textarea value={text} onChange={(event) => setText(event.target.value)} required rows={3} disabled={busy} /></label>
          <label><span>New target guidance</span><textarea value={guidance} onChange={(event) => setGuidance(event.target.value)} rows={2} disabled={busy} /></label>
          <div className="platform-page-actions"><Button label="Create version" type="submit" isDisabled={busy} /></div>
        </form>
      </details>
    </div>
  </article>;
}
