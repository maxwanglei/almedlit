import {
  useEffect,
  useRef,
  useState,
  type ComponentType,
  type FormEvent,
  type InputHTMLAttributes,
  type ReactElement,
} from "react";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";
import {
  TextInput,
  type TextInputProps,
} from "@astryxdesign/core/TextInput";

import {
  completeAccountAction,
  getAccountActionPreview,
} from "@/api/client";
import BrandLogo from "@/components/BrandLogo";
import type { AccountActionPreview } from "@/types/api";

interface AccountActionPageProps {
  token: string;
  /** Called after the one-time action completes and its new session is stored. */
  onCompleted: () => void;
}

type AuthTextInputProps = TextInputProps & {
  autoComplete?: InputHTMLAttributes<HTMLInputElement>["autoComplete"];
};

const AuthTextInput = TextInput as ComponentType<AuthTextInputProps>;
const PASSWORD_MIN_BYTES = 12;
const PASSWORD_MAX_BYTES = 72;

function messageFrom(caught: unknown, fallback: string): string {
  return caught instanceof Error ? caught.message : fallback;
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function actionCopy(preview: AccountActionPreview): {
  title: string;
  eyebrow: string;
  instruction: string;
  submit: string;
} {
  if (preview.purpose === "activation") {
    return {
      title: "Activate your account",
      eyebrow: "Account activation",
      instruction: "Choose a password to activate your account and sign in.",
      submit: "Activate account",
    };
  }
  return {
    title: "Reset your password",
    eyebrow: "Account recovery",
    instruction: "Choose a new password. Completing this reset signs out your other sessions.",
    submit: "Reset password",
  };
}

export default function AccountActionPage({
  token,
  onCompleted,
}: AccountActionPageProps): ReactElement {
  const passwordRef = useRef<HTMLInputElement>(null);
  const confirmationRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<AccountActionPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [confirmationError, setConfirmationError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setPreview(null);
    setPreviewError(null);
    setPassword("");
    setConfirmation("");
    setPasswordError(null);
    setConfirmationError(null);
    setError(null);

    void getAccountActionPreview(token)
      .then((nextPreview) => {
        if (!cancelled) setPreview(nextPreview);
      })
      .catch((caught) => {
        if (!cancelled) {
          setPreviewError(
            messageFrom(caught, "This one-time account link is no longer valid."),
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [token]);

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    setPasswordError(null);
    setConfirmationError(null);

    const passwordBytes = new TextEncoder().encode(password).length;
    if (passwordBytes < PASSWORD_MIN_BYTES || passwordBytes > PASSWORD_MAX_BYTES) {
      setPasswordError(
        `Use between ${PASSWORD_MIN_BYTES} and ${PASSWORD_MAX_BYTES} UTF-8 bytes.`,
      );
      passwordRef.current?.focus();
      return;
    }
    if (password !== confirmation) {
      setConfirmationError("Passwords do not match.");
      confirmationRef.current?.focus();
      return;
    }

    setBusy(true);
    try {
      await completeAccountAction(token, password);
      onCompleted();
    } catch (caught) {
      setError(messageFrom(caught, "The account action could not be completed."));
    } finally {
      setBusy(false);
    }
  }

  const copy = preview ? actionCopy(preview) : null;
  const accountName = preview
    ? preview.display_name.trim() || preview.username
    : null;

  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <main
        id="main-content"
        className="app-shell auth-shell account-action-page"
        tabIndex={-1}
      >
        <section className="auth-panel" aria-labelledby="account-action-title">
          <div className="auth-heading">
            <BrandLogo size="hero" />
            <div>
              <span className="auth-eyebrow">
                {copy?.eyebrow ?? "One-time account link"}
              </span>
              <h1 id="account-action-title">
                {copy?.title ?? "Checking your link"}
              </h1>
            </div>
          </div>

          {loading ? (
            <div className="status" role="status" aria-live="polite">
              Checking account link…
            </div>
          ) : null}

          {!loading && (previewError !== null || !preview || !copy) ? (
            <Banner
              className="form-banner"
              status="error"
              title="This account link cannot be used"
              description={
                previewError ??
                "The link may have expired, already been used, or been replaced. Ask a system administrator for a new one."
              }
            />
          ) : null}

          {preview && copy ? (
            <>
              <p className="account-action-summary">
                <strong>{accountName}</strong> · {preview.username}
              </p>
              <p className="account-action-instruction">{copy.instruction}</p>
              <p className="account-action-expiry">
                This link expires <time dateTime={preview.expires_at}>{formatDate(preview.expires_at)}</time>.
              </p>
              <form className="auth-form" onSubmit={(event) => void submit(event)}>
                <AuthTextInput
                  ref={passwordRef}
                  className="auth-field-control"
                  label="New password"
                  type="password"
                  value={password}
                  onChange={(value) => {
                    setPassword(value);
                    setPasswordError(null);
                  }}
                  isRequired
                  htmlName="newPassword"
                  autoComplete="new-password"
                  status={
                    passwordError
                      ? { type: "error", message: passwordError }
                      : undefined
                  }
                  width="100%"
                />
                <AuthTextInput
                  ref={confirmationRef}
                  className="auth-field-control"
                  label="Confirm new password"
                  type="password"
                  value={confirmation}
                  onChange={(value) => {
                    setConfirmation(value);
                    setConfirmationError(null);
                  }}
                  isRequired
                  htmlName="confirmPassword"
                  autoComplete="new-password"
                  status={
                    confirmationError
                      ? { type: "error", message: confirmationError }
                      : undefined
                  }
                  width="100%"
                />
                {error ? (
                  <Banner
                    className="form-banner"
                    status="error"
                    title="Could not update the account"
                    description={error}
                  />
                ) : null}
                <Button
                  label={copy.submit}
                  type="submit"
                  variant="primary"
                  className="auth-submit"
                  isDisabled={busy}
                  isLoading={busy}
                />
              </form>
            </>
          ) : null}
        </section>
      </main>
    </>
  );
}
