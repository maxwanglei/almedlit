import { NativeInput, NativeTextarea } from "@/components/NativeField";
import ValidatedForm from "@/components/ValidatedForm";
import { useState } from "react";
import type { FormEvent } from "react";

import { createGuidelineVersion } from "@/api/client";
import type { GuidelineVersion } from "@/types/api";

import { PlatformSection, PlatformStatus } from "./components";

interface EvidenceGuidelineManagementProps {
  projectId: number;
  guidelines: GuidelineVersion[];
  busy?: boolean;
  onChanged: () => Promise<void>;
}

export default function EvidenceGuidelineManagement(props: EvidenceGuidelineManagementProps): React.ReactElement {
  return <GuidelineManagementForm key={props.projectId} {...props} />;
}

function GuidelineManagementForm({
  projectId,
  guidelines,
  busy = false,
  onChanged,
}: EvidenceGuidelineManagementProps): React.ReactElement {
  const [versionLabel, setVersionLabel] = useState("");
  const [markdown, setMarkdown] = useState("");
  const [saving, setSaving] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState(false);
  const [savedGuideline, setSavedGuideline] = useState<GuidelineVersion | null>(null);
  const localGuideline = savedGuideline && !guidelines.some((guideline) => guideline.id === savedGuideline.id)
    ? savedGuideline
    : null;
  const versions = localGuideline
    ? [localGuideline, ...guidelines.map((guideline) => guideline.status === "active"
      ? { ...guideline, status: "superseded" }
      : guideline)]
    : guidelines;
  const activeGuideline = versions.find((guideline) => guideline.status === "active");
  const pending = busy || saving || refreshing;

  async function refreshData(): Promise<void> {
    setRefreshing(true);
    setRefreshError(false);
    try {
      await onChanged();
    } catch {
      setRefreshError(true);
    } finally {
      setRefreshing(false);
    }
  }

  async function saveGuideline(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (pending || refreshError) return;
    const label = versionLabel.trim();
    if (!label) {
      setError("Enter a guideline version label.");
      return;
    }
    if ([...label].length > 50) {
      setError("Guideline version labels must be 50 characters or fewer.");
      return;
    }
    if (!markdown.trim()) {
      setError("Enter guideline text before activating a version.");
      return;
    }

    setSaving(true);
    setError(null);
    try {
      const saved = await createGuidelineVersion({
        project_id: projectId,
        version_label: label,
        markdown,
        status: "active",
      });
      setSavedGuideline(saved);
      setVersionLabel("");
      setMarkdown("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The guideline could not be saved.");
      setSaving(false);
      return;
    }
    setSaving(false);
    await refreshData();
  }

  return (
    <PlatformSection
      title="Evidence annotation guidelines"
      description="Create an active guideline before activating evidence targets or assigning review work."
    >
      <p className="platform-form-note">
        New assignments use the active version. Existing assignments keep their pinned guideline.
      </p>
      {activeGuideline ? (
        <div>
          <h3>Active guideline: {activeGuideline.version_label}</h3>
          <pre>{activeGuideline.markdown}</pre>
        </div>
      ) : <p className="platform-inline-empty">No active evidence guideline.</p>}

      <ValidatedForm className="platform-dialog-form" aria-label="Create evidence guideline" onSubmit={(event) => void saveGuideline(event)}>
        <label>
          <span>Guideline version</span>
          <NativeInput required maxLength={50} value={versionLabel} disabled={pending || refreshError}
            placeholder="v1" onChange={(event) => setVersionLabel(event.target.value)} />
        </label>
        <label>
          <span>Guideline text</span>
          <NativeTextarea required rows={6} value={markdown} disabled={pending || refreshError}
            onChange={(event) => setMarkdown(event.target.value)} />
        </label>
        {error ? <p className="platform-form-warning" role="alert">{error}</p> : null}
        <div className="recovery-actions">
          <button type="submit" disabled={pending || refreshError}>
            {saving ? "Creating guideline…" : "Create active guideline"}
          </button>
        </div>
      </ValidatedForm>
      {savedGuideline ? (
        <p role="status">
          {activeGuideline?.id === savedGuideline.id
            ? `Guideline ${savedGuideline.version_label} is active.`
            : `Guideline ${savedGuideline.version_label} was saved.`}
        </p>
      ) : null}
      {refreshError ? (
        <div className="platform-form-warning" role="alert">
          <p>Guideline was saved, but project data could not be refreshed.</p>
          <button type="button" disabled={pending} onClick={() => void refreshData()}>Retry refresh</button>
        </div>
      ) : null}
      {versions.length ? (
        <div className="platform-table-scroll">
          <table className="platform-table" aria-label="Evidence guideline versions">
            <thead><tr><th scope="col">Version</th><th scope="col">Status</th><th scope="col">Guidance</th></tr></thead>
            <tbody>{versions.map((guideline) => (
              <tr key={guideline.id}>
                <td>{guideline.version_label}</td>
                <td><PlatformStatus value={guideline.status} /></td>
                <td><details><summary>Read {guideline.version_label}</summary><pre>{guideline.markdown}</pre></details></td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      ) : null}
    </PlatformSection>
  );
}
