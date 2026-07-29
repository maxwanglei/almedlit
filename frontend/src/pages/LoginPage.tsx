import { useRef, useState } from "react";
import type {
  ComponentType,
  FormEvent,
  InputHTMLAttributes,
  ReactElement,
} from "react";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";
import { Selector } from "@astryxdesign/core/Selector";
import {
  TextInput,
  type TextInputProps,
} from "@astryxdesign/core/TextInput";

import { login, register } from "@/api/client";
import BrandLogo from "@/components/BrandLogo";

interface LoginPageProps {
  onAuthed: (justRegistered: boolean) => void;
}

type AuthTextInputProps = TextInputProps & {
  autoComplete?: InputHTMLAttributes<HTMLInputElement>["autoComplete"];
  autoCapitalize?: InputHTMLAttributes<HTMLInputElement>["autoCapitalize"];
  spellCheck?: InputHTMLAttributes<HTMLInputElement>["spellCheck"];
};

// TextInput forwards native attributes to its input, but its current public
// type omits autoComplete. Keep the compatibility shim local to the auth form.
const AuthTextInput = TextInput as ComponentType<AuthTextInputProps>;

export default function LoginPage({ onAuthed }: LoginPageProps): ReactElement {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [workspaceKind, setWorkspaceKind] = useState<"individual" | "team">("individual");
  const [workspaceName, setWorkspaceName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [usernameError, setUsernameError] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const usernameRef = useRef<HTMLInputElement>(null);
  const passwordRef = useRef<HTMLInputElement>(null);

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
      if (passwordBytes < 12 || passwordBytes > 72) {
        setPasswordError("Use between 12 and 72 UTF-8 bytes.");
        passwordRef.current?.focus();
        return;
      }
    }

    setBusy(true);
    setError(null);
    try {
      if (mode === "register") {
        await register(trimmedUsername, password, {
          displayName: displayName.trim() || undefined,
          workspaceKind,
          workspaceName: workspaceKind === "team" ? workspaceName.trim() || undefined : undefined,
        });
        onAuthed(true);
      } else {
        await login(trimmedUsername, password);
        onAuthed(false);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Authentication failed");
    } finally {
      setBusy(false);
    }
  }

  function switchMode(): void {
    setError(null);
    setUsernameError(null);
    setPasswordError(null);
    setUsername("");
    setPassword("");
    setDisplayName("");
    setWorkspaceKind("individual");
    setWorkspaceName("");
    setMode((current) => (current === "register" ? "login" : "register"));
  }

  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <main
        id="main-content"
        className="app-shell auth-shell"
        tabIndex={-1}
      >
      <section className="auth-panel" aria-labelledby="auth-title">
        <div className="auth-heading">
          <BrandLogo size="hero" />
          <div>
            <span className="auth-eyebrow">An active learning platform</span>
            <h1 id="auth-title">AL-MedLit</h1>
            <p>{mode === "register" ? "Create your workspace" : "Sign in to your workspace"}</p>
          </div>
        </div>
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
              mode === "register" ? "section-register username" : "section-login username"
            }
            status={
              usernameError
                ? { type: "error", message: usernameError }
                : undefined
            }
            width="100%"
          />
          {mode === "register" ? (
            <>
              <AuthTextInput
                className="auth-field-control"
                label="Display name"
                value={displayName}
                onChange={setDisplayName}
                htmlName="displayName"
                autoComplete="section-register name"
                width="100%"
              />
              <Selector
                label="Workspace"
                value={workspaceKind}
                onChange={(value) => setWorkspaceKind(value as "individual" | "team")}
                placement="below"
                options={[
                  { value: "individual", label: "Individual" },
                  { value: "team", label: "Team" },
                ]}
                width="100%"
              />
              {workspaceKind === "team" ? (
                <AuthTextInput
                  className="auth-field-control"
                  label="Team name"
                  value={workspaceName}
                  onChange={setWorkspaceName}
                  htmlName="workspaceName"
                  autoComplete="section-register organization"
                  width="100%"
                />
              ) : null}
            </>
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
                ? "section-register new-password"
                : "section-login current-password"
            }
            status={
              passwordError
                ? { type: "error", message: passwordError }
                : undefined
            }
            width="100%"
          />
          {error ? (
            <Banner
              className="form-banner"
              status="error"
              title="Authentication failed"
              description={error}
            />
          ) : null}
          <Button
            label={mode === "register" ? "Register" : "Sign in"}
            variant="primary"
            type="submit"
            isDisabled={busy}
            isLoading={busy}
            className="auth-submit"
          />
        </form>
        <Button
          className="auth-mode"
          label={mode === "register" ? "Use an existing account" : "Create an account"}
          variant="ghost"
          onClick={switchMode}
        />
      </section>
      </main>
    </>
  );
}
