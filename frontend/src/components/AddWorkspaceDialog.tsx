import { useEffect, useId, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";

import { applyToWorkspace, createWorkspace } from "@/api/client";

import DialogFrame from "./DialogFrame";

export default function AddWorkspaceDialog({
  open,
  onDismiss,
  onWorkspaceCreated,
  onJoinRequested,
  requestGeneration,
}: {
  open: boolean;
  onDismiss: () => void;
  onWorkspaceCreated: (workspaceId: number) => void | Promise<void>;
  onJoinRequested: () => void | Promise<void>;
  requestGeneration: number;
}): React.ReactElement | null {
  const titleId = useId();
  const descriptionId = useId();
  const firstFieldRef = useRef<HTMLInputElement>(null);
  const requestGenerationRef = useRef(requestGeneration);
  requestGenerationRef.current = requestGeneration;
  const [mode, setMode] = useState<"create" | "join">("create");
  const [teamName, setTeamName] = useState("");
  const [joinCode, setJoinCode] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setMode("create");
      setTeamName("");
      setJoinCode("");
      setMessage("");
      setBusy(false);
      setError(null);
      setStatus(null);
    }
  }, [open]);

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    setStatus(null);
    const requiredValue = mode === "create" ? teamName.trim() : joinCode.trim();
    if (!requiredValue) {
      setError(mode === "create" ? "Enter a team name." : "Enter a join code.");
      firstFieldRef.current?.focus();
      return;
    }

    setBusy(true);
    const startedGeneration = requestGenerationRef.current;
    try {
      if (mode === "create") {
        const workspace = await createWorkspace(requiredValue);
        if (startedGeneration !== requestGenerationRef.current) return;
        await onWorkspaceCreated(workspace.id);
        if (startedGeneration !== requestGenerationRef.current) return;
        onDismiss();
      } else {
        await applyToWorkspace(requiredValue, message);
        if (startedGeneration !== requestGenerationRef.current) return;
        setStatus("Your request was sent to the team administrators.");
      }
    } catch (caught) {
      if (startedGeneration !== requestGenerationRef.current) return;
      setError(
        caught instanceof Error ? caught.message : "The workspace could not be updated.",
      );
    } finally {
      if (startedGeneration === requestGenerationRef.current) {
        setBusy(false);
      }
    }
  }

  async function refreshMemberships(): Promise<void> {
    const startedGeneration = requestGenerationRef.current;
    setBusy(true);
    setError(null);
    try {
      await onJoinRequested();
      if (startedGeneration !== requestGenerationRef.current) return;
      setStatus(
        "Workspace memberships refreshed. Pending requests appear after a team administrator approves them.",
      );
    } catch (caught) {
      if (startedGeneration !== requestGenerationRef.current) return;
      setError(
        caught instanceof Error
          ? caught.message
          : "Workspace memberships could not be refreshed.",
      );
    } finally {
      if (startedGeneration === requestGenerationRef.current) {
        setBusy(false);
      }
    }
  }

  return (
    <DialogFrame
      open={open}
      labelledBy={titleId}
      describedBy={descriptionId}
      busy={busy}
      error={error}
      initialFocusRef={firstFieldRef}
      backdropClassName="platform-dialog-backdrop"
      dialogClassName="platform-dialog workspace-access-dialog"
      onDismiss={onDismiss}
    >
      <header>
        <div>
          <h2 id={titleId}>Add a workspace</h2>
          <p id={descriptionId}>
            Create a team or request access to an existing team. Your individual
            workspace is never converted or removed.
          </p>
        </div>
      </header>

      <div className="workspace-access-modes" role="group" aria-label="Workspace action">
        <Button
          label="Create a team"
          variant={mode === "create" ? "primary" : "ghost"}
          isDisabled={busy}
          aria-pressed={mode === "create"}
          onClick={() => {
            setMode("create");
            setError(null);
            setStatus(null);
            window.requestAnimationFrame(() => firstFieldRef.current?.focus());
          }}
        />
        <Button
          label="Join a team"
          variant={mode === "join" ? "primary" : "ghost"}
          isDisabled={busy}
          aria-pressed={mode === "join"}
          onClick={() => {
            setMode("join");
            setError(null);
            setStatus(null);
            window.requestAnimationFrame(() => firstFieldRef.current?.focus());
          }}
        />
      </div>

      <form className="platform-dialog-form" onSubmit={submit}>
        {mode === "create" ? (
          <label>
            <span>Team name</span>
            <input
              ref={firstFieldRef}
              name="teamName"
              value={teamName}
              maxLength={255}
              autoComplete="organization"
              disabled={busy}
              aria-invalid={Boolean(error) || undefined}
              onChange={(event) => setTeamName(event.target.value)}
            />
            <small>You will be the team workspace administrator.</small>
          </label>
        ) : (
          <>
            <label>
              <span>Join code</span>
              <input
                ref={firstFieldRef}
                name="joinCode"
                value={joinCode}
                maxLength={40}
                autoCapitalize="none"
                autoComplete="off"
                spellCheck={false}
                disabled={busy || status !== null}
                aria-invalid={Boolean(error) || undefined}
                onChange={(event) => setJoinCode(event.target.value)}
              />
            </label>
            <label>
              <span>Message <small>(optional)</small></span>
              <textarea
                name="message"
                rows={3}
                value={message}
                disabled={busy || status !== null}
                onChange={(event) => setMessage(event.target.value)}
              />
            </label>
          </>
        )}

        {error ? (
          <Banner
            className="platform-dialog-error"
            status="error"
            title="Workspace action failed"
            description={error}
          />
        ) : null}
        {status ? <p className="platform-form-success" role="status">{status}</p> : null}

        <div className="platform-dialog-actions">
          <Button
            label={status ? "Close" : "Cancel"}
            variant="ghost"
            isDisabled={busy}
            onClick={onDismiss}
          />
          {status ? (
            <Button
              label="Refresh workspaces"
              variant="primary"
              isDisabled={busy}
              isLoading={busy}
              onClick={() => void refreshMemberships()}
            />
          ) : (
            <Button
              label={mode === "create" ? "Create team" : "Send request"}
              variant="primary"
              type="submit"
              isDisabled={busy}
              isLoading={busy}
            />
          )}
        </div>
      </form>
    </DialogFrame>
  );
}
