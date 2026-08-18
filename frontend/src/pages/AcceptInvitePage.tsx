import { useEffect, useRef, useState } from "react";
import type {
  ComponentType,
  FormEvent,
  InputHTMLAttributes,
  ReactElement,
} from "react";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";
import {
  TextInput,
  type TextInputProps,
} from "@astryxdesign/core/TextInput";

import { acceptInvite, authenticate, getInvitePreview } from "@/api/client";
import BrandLogo from "@/components/BrandLogo";
import type { InvitePreview } from "@/types/api";

interface AcceptInvitePageProps {
  token: string;
  signedIn: boolean;
  /** Called once membership exists so the shell can reload the session. */
  onAccepted: () => void;
}

type AuthTextInputProps = TextInputProps & {
  autoComplete?: InputHTMLAttributes<HTMLInputElement>["autoComplete"];
  autoCapitalize?: InputHTMLAttributes<HTMLInputElement>["autoCapitalize"];
  spellCheck?: InputHTMLAttributes<HTMLInputElement>["spellCheck"];
};

// Mirrors the shim in LoginPage: TextInput forwards native attributes but its
// public type omits autoComplete.
const AuthTextInput = TextInput as ComponentType<AuthTextInputProps>;

const PASSWORD_MIN_BYTES = 12;
const PASSWORD_MAX_BYTES = 72;

const ROLE_LABELS: Record<string, string> = {
  annotator: "an annotator",
  trainer: "a trainer",
  manager: "a manager",
  admin: "an admin",
};

function describeRole(role: string): string {
  return ROLE_LABELS[role] ?? `a ${role}`;
}

export default function AcceptInvitePage({
  token,
  signedIn,
  onAccepted,
}: AcceptInvitePageProps): ReactElement {
  const [preview, setPreview] = useState<InvitePreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [mode, setMode] = useState<"register" | "login">("register");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [usernameError, setUsernameError] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const usernameRef = useRef<HTMLInputElement>(null);
  const passwordRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setPreviewError(null);

    async function loadPreview(): Promise<void> {
      try {
        const nextPreview = await getInvitePreview(token);
        if (!cancelled) {
          setPreview(nextPreview);
        }
      } catch (caught) {
        if (!cancelled) {
          setPreviewError(
            caught instanceof Error
              ? caught.message
              : "This invite link is no longer valid.",
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadPreview();
    return () => {
      cancelled = true;
    };
  }, [token]);

  async function handleConfirm(): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      // The stored bearer token is attached by the shared request helper.
      await acceptInvite(token);
      onAccepted();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not accept the invite");
    } finally {
      setBusy(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const trimmedUsername = username.trim();
    const nextUsernameError = trimmedUsername ? null : "Enter your username.";
    const nextPasswordError = password ? null : "Enter your password.";
    setUsernameError(nextUsernameError);
    setPasswordError(nextPasswordError);
    setError(null);
    if (nextUsernameError || nextPasswordError) {
      (nextUsernameError ? usernameRef : passwordRef).current?.focus();
      return;
    }
    if (mode === "register") {
      const passwordBytes = new TextEncoder().encode(password).length;
      if (passwordBytes < PASSWORD_MIN_BYTES || passwordBytes > PASSWORD_MAX_BYTES) {
        setPasswordError(
          `Use between ${PASSWORD_MIN_BYTES} and ${PASSWORD_MAX_BYTES} UTF-8 bytes.`,
        );
        passwordRef.current?.focus();
        return;
      }
    }

    setBusy(true);
    try {
      if (mode === "register") {
        await acceptInvite(token, {
          username: trimmedUsername,
          password,
          display_name: displayName.trim() || undefined,
        });
      } else {
        // Stage the login token until redemption succeeds. Adopting it earlier
        // would navigate away on a same-tab token event and hide a stale or
        // concurrently revoked invitation error.
        const stagedToken = await authenticate(trimmedUsername, password);
        await acceptInvite(token, {}, stagedToken);
      }
      onAccepted();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not accept the invite");
    } finally {
      setBusy(false);
    }
  }

  function renderBody(): ReactElement {
    if (loading) {
      return (
        <div className="status" role="status" aria-live="polite">
          Loading invite…
        </div>
      );
    }

    if (previewError !== null || preview === null) {
      return (
        <Banner
          className="form-banner"
          status="error"
          title="This invite cannot be used"
          description={
            previewError ??
            "The link may have already been used, expired, or been revoked. Ask a workspace admin for a new one."
          }
        />
      );
    }

    if (signedIn) {
      return (
        <>
          <p className="invite-summary">
            You have been invited to join <strong>{preview.workspace_name}</strong> as{" "}
            {describeRole(preview.role)}.
          </p>
          {error ? (
            <Banner
              className="form-banner"
              status="error"
              title="Could not accept the invite"
              description={error}
            />
          ) : null}
          <Button
            label="Accept invite"
            variant="primary"
            isDisabled={busy}
            isLoading={busy}
            className="auth-submit"
            onClick={() => {
              void handleConfirm();
            }}
          />
        </>
      );
    }

    return (
      <>
        <p className="invite-summary">
          You have been invited to join <strong>{preview.workspace_name}</strong> as{" "}
          {describeRole(preview.role)}.
        </p>
        <p className="invite-mode-hint">
          {mode === "register"
            ? "Create an account to join."
            : "Sign in to join with your existing account."}
        </p>
        <form key={mode} className="auth-form" onSubmit={handleSubmit}>
          <AuthTextInput
            ref={usernameRef}
            className="auth-field-control"
            label="Username"
            value={username}
            onChange={(value) => {
              setUsername(value);
              setUsernameError(null);
            }}
            isRequired
            htmlName="username"
            autoCapitalize="none"
            spellCheck={false}
            autoComplete={
              mode === "register" ? "section-invite username" : "section-login username"
            }
            status={usernameError ? { type: "error", message: usernameError } : undefined}
            width="100%"
          />
          {mode === "register" ? (
            <AuthTextInput
              className="auth-field-control"
              label="Display name"
              value={displayName}
              onChange={setDisplayName}
              htmlName="displayName"
              autoComplete="section-invite name"
              width="100%"
            />
          ) : null}
          <AuthTextInput
            ref={passwordRef}
            className="auth-field-control"
            label="Password"
            type="password"
            value={password}
            onChange={(value) => {
              setPassword(value);
              setPasswordError(null);
            }}
            isRequired
            htmlName="password"
            autoComplete={
              mode === "register"
                ? "section-invite new-password"
                : "section-login current-password"
            }
            status={passwordError ? { type: "error", message: passwordError } : undefined}
            width="100%"
          />
          {error ? (
            <Banner
              className="form-banner"
              status="error"
              title="Could not accept the invite"
              description={error}
            />
          ) : null}
          <Button
            label="Accept invite"
            variant="primary"
            type="submit"
            isDisabled={busy}
            isLoading={busy}
            className="auth-submit"
          />
        </form>
        <Button
          className="auth-mode"
          label={
            mode === "register" ? "Use an existing account" : "Create a new account"
          }
          variant="ghost"
          onClick={() => {
            setMode((current) => (current === "register" ? "login" : "register"));
            setError(null);
            setUsernameError(null);
            setPasswordError(null);
            setUsername("");
            setPassword("");
            setDisplayName("");
          }}
        />
      </>
    );
  }

  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <main id="main-content" className="app-shell auth-shell" tabIndex={-1}>
        <section className="auth-panel" aria-labelledby="invite-title">
          <div className="auth-heading">
            <BrandLogo size="hero" />
            <div>
              <span className="auth-eyebrow">An active learning platform</span>
              <h1 id="invite-title">Accept your invite</h1>
            </div>
          </div>
          {renderBody()}
        </section>
      </main>
    </>
  );
}
