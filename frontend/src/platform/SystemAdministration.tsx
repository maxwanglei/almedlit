import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type FormEvent,
  type ReactElement,
} from "react";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";
import {
  Copy,
  KeyRound,
  Plus,
  RefreshCw,
  ServerCog,
  UserRound,
  X,
} from "lucide-react";

import {
  createAdminActivationLink,
  createAdminPasswordResetLink,
  createAdminUser,
  getAdminSettings,
  getAdminUser,
  listAdminUsers,
  setAdminUserStatus,
  updateAdminSettings,
} from "@/api/client";
import ConfirmDialog from "@/components/ConfirmDialog";
import DialogFrame from "@/components/DialogFrame";
import { shouldHandleSpaClick } from "@/components/ModuleSwitcher";
import type {
  AccountActionLink,
  AdminSettings,
  AdminUserCreateResult,
  AdminUserDetail,
  AdminUserListParams,
  AdminUserPage,
  AdminUserSummary,
} from "@/types/api";

import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformSection,
  PlatformStatus,
} from "./components";

const PAGE_SIZE = 20;
const INVITE_EXPIRY_MIN = 60;
const INVITE_EXPIRY_MAX = 43_200;
const ACTION_EXPIRY_MIN = 15;
const ACTION_EXPIRY_MAX = 1_440;

const SECTIONS = [
  { id: "users", label: "Users" },
  { id: "settings", label: "Instance settings" },
  { id: "health", label: "System health" },
  { id: "plugins", label: "Plugins" },
  { id: "audit", label: "Audit" },
] as const;

type AdminSectionId = (typeof SECTIONS)[number]["id"];

function messageFrom(caught: unknown, fallback: string): string {
  return caught instanceof Error ? caught.message : fallback;
}

function formatDate(value: string | null | undefined): string {
  if (!value) return "Never";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function absoluteActionUrl(value: string): string {
  try {
    return new URL(value, window.location.origin).toString();
  } catch {
    return value;
  }
}

function displayName(user: Pick<AdminUserSummary, "display_name" | "username">): string {
  return user.display_name.trim() || user.username;
}

async function copyText(value: string): Promise<void> {
  if (!navigator.clipboard?.writeText) {
    throw new Error("Clipboard access is unavailable. Select and copy the link manually.");
  }
  await navigator.clipboard.writeText(value);
}

function ActionLinkCard({ link }: { link: AccountActionLink }): ReactElement {
  const [copyStatus, setCopyStatus] = useState<string | null>(null);
  const label = link.purpose === "activation" ? "Activation link" : "Password-reset link";
  const actionUrl = absoluteActionUrl(link.url);

  return (
    <div className="system-admin-action-link" aria-live="polite">
      <div>
        <strong>{label}</strong>
        <span>Expires {formatDate(link.expires_at)}</span>
      </div>
      <div className="system-admin-copy-row">
        <input aria-label={label} readOnly value={actionUrl} onFocus={(event) => event.currentTarget.select()} />
        <Button
          label={`Copy ${label.toLowerCase()}`}
          icon={<Copy size={16} />}
          variant="ghost"
          onClick={() => {
            setCopyStatus(null);
            void copyText(actionUrl)
              .then(() => setCopyStatus("Link copied."))
              .catch((caught) => setCopyStatus(messageFrom(caught, "Could not copy the link.")));
          }}
        />
      </div>
      {copyStatus ? <small>{copyStatus}</small> : null}
    </div>
  );
}

function CreateUserDialog({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: (result: AdminUserCreateResult) => void;
}): ReactElement | null {
  const titleId = useId();
  const [username, setUsername] = useState("");
  const [displayNameValue, setDisplayNameValue] = useState("");
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AdminUserCreateResult | null>(null);

  useEffect(() => {
    if (!open) {
      setUsername("");
      setDisplayNameValue("");
      setEmail("");
      setBusy(false);
      setError(null);
      setResult(null);
    }
  }, [open]);

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const normalizedUsername = username.trim();
    if (!normalizedUsername) {
      setError("Enter a username.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const created = await createAdminUser({
        username: normalizedUsername,
        display_name: displayNameValue.trim() || undefined,
        email: email.trim() || undefined,
      });
      setResult(created);
      onCreated(created);
    } catch (caught) {
      setError(messageFrom(caught, "The account could not be created."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <DialogFrame
      open={open}
      labelledBy={titleId}
      busy={busy}
      error={error}
      backdropClassName="platform-dialog-backdrop"
      dialogClassName="platform-dialog system-admin-dialog"
      dialogElement="section"
      initialFocusSelector="form input:not([disabled])"
      onDismiss={onClose}
    >
      <header>
        <div>
          <UserRound size={18} aria-hidden="true" />
          <h2 id={titleId}>Create inactive account</h2>
        </div>
        <Button
          label="Close"
          icon={<X size={17} />}
          isIconOnly
          variant="ghost"
          isDisabled={busy}
          onClick={onClose}
        />
      </header>
      {result ? (
        <div className="platform-dialog-form system-admin-created-account">
          <Banner
            status="success"
            title="Account created"
            description={`${displayName(result.user)} is inactive until the one-time link is completed.`}
          />
          <ActionLinkCard link={result.action} />
          <div className="platform-dialog-actions">
            <Button label="Done" variant="primary" onClick={onClose} />
          </div>
        </div>
      ) : (
        <form className="platform-dialog-form" onSubmit={(event) => void submit(event)}>
          <p className="platform-form-note">
            The account starts inactive. Share the generated link out of band so the user can set a password.
          </p>
          <label>
            Username
            <input
              name="username"
              value={username}
              autoComplete="off"
              autoCapitalize="none"
              spellCheck={false}
              required
              disabled={busy}
              onChange={(event) => setUsername(event.target.value)}
            />
          </label>
          <label>
            Display name
            <input
              name="displayName"
              value={displayNameValue}
              autoComplete="off"
              disabled={busy}
              onChange={(event) => setDisplayNameValue(event.target.value)}
            />
          </label>
          <label>
            Email
            <input
              name="email"
              type="email"
              value={email}
              autoComplete="off"
              disabled={busy}
              onChange={(event) => setEmail(event.target.value)}
            />
          </label>
          {error ? <p className="platform-dialog-error" role="alert">{error}</p> : null}
          <div className="platform-dialog-actions">
            <Button
              label={busy ? "Creating…" : "Create account"}
              type="submit"
              variant="primary"
              isDisabled={busy}
              isLoading={busy}
            />
          </div>
        </form>
      )}
    </DialogFrame>
  );
}

function UserDetailDialog({
  userId,
  onClose,
  onChanged,
}: {
  userId: number | null;
  onClose: () => void;
  onChanged: (user: AdminUserDetail) => void;
}): ReactElement | null {
  const titleId = useId();
  const requestIdRef = useRef(0);
  const [user, setUser] = useState<AdminUserDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [link, setLink] = useState<AccountActionLink | null>(null);
  const [nextActive, setNextActive] = useState<boolean | null>(null);

  useEffect(() => {
    const requestId = ++requestIdRef.current;
    setUser(null);
    setLink(null);
    setError(null);
    setNextActive(null);
    if (userId === null) return;
    setLoading(true);
    void getAdminUser(userId)
      .then((nextUser) => {
        if (requestId === requestIdRef.current) setUser(nextUser);
      })
      .catch((caught) => {
        if (requestId === requestIdRef.current) {
          setError(messageFrom(caught, "The account could not be loaded."));
        }
      })
      .finally(() => {
        if (requestId === requestIdRef.current) setLoading(false);
      });
    return () => {
      requestIdRef.current += 1;
    };
  }, [userId]);

  async function generateLink(purpose: "activation" | "password_reset"): Promise<void> {
    if (!user) return;
    setBusy(true);
    setError(null);
    setLink(null);
    try {
      const nextLink =
        purpose === "activation"
          ? await createAdminActivationLink(user.id)
          : await createAdminPasswordResetLink(user.id);
      setLink(nextLink);
    } catch (caught) {
      setError(messageFrom(caught, "A one-time link could not be generated."));
    } finally {
      setBusy(false);
    }
  }

  async function changeStatus(): Promise<void> {
    if (!user || nextActive === null) return;
    setBusy(true);
    setError(null);
    try {
      const updated = await setAdminUserStatus(user.id, nextActive);
      setUser(updated);
      setLink(null);
      setNextActive(null);
      onChanged(updated);
    } catch (caught) {
      setError(messageFrom(caught, "The account status could not be changed."));
      setNextActive(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <DialogFrame
        open={userId !== null}
        labelledBy={titleId}
        busy={busy}
        error={error}
        backdropClassName="platform-dialog-backdrop"
        dialogClassName="platform-dialog system-admin-dialog system-admin-user-dialog"
        dialogElement="section"
        onDismiss={onClose}
      >
        <header>
          <div>
            <UserRound size={18} aria-hidden="true" />
            <h2 id={titleId}>{user ? displayName(user) : "User details"}</h2>
          </div>
          <Button
            label="Close"
            icon={<X size={17} />}
            isIconOnly
            variant="ghost"
            isDisabled={busy}
            onClick={onClose}
          />
        </header>
        <div className="system-admin-user-detail">
          {loading ? <p role="status">Loading account…</p> : null}
          {error ? (
            <Banner
              status="error"
              title="Account action failed"
              description={error}
            />
          ) : null}
          {user ? (
            <>
              <div className="system-admin-user-summary">
                <div>
                  <strong>{user.username}</strong>
                  <span>{user.email ?? "No email address"}</span>
                </div>
                <PlatformStatus value={user.is_active ? "active" : "inactive"} />
                {user.is_superuser ? <PlatformStatus value="superuser" /> : null}
              </div>
              <dl className="system-admin-detail-list">
                <div><dt>Last login</dt><dd>{formatDate(user.last_login_at)}</dd></div>
                <div><dt>Memberships</dt><dd>{user.membership_count}</dd></div>
              </dl>
              <section className="system-admin-memberships" aria-labelledby={`${titleId}-memberships`}>
                <h3 id={`${titleId}-memberships`}>Workspace memberships</h3>
                {user.memberships.length ? (
                  <div className="platform-table-scroll" tabIndex={0} aria-label="Workspace memberships">
                    <table className="platform-table platform-table--summary">
                      <thead><tr><th scope="col">Workspace</th><th scope="col">Type</th><th scope="col">Role</th></tr></thead>
                      <tbody>
                        {user.memberships.map((membership) => (
                          <tr key={membership.workspace_id}>
                            <td data-label="Workspace" data-priority="identity">
                              <strong>{membership.workspace_name}</strong>
                              <span>ID {membership.workspace_id}</span>
                            </td>
                            <td data-label="Type">{membership.workspace_kind === "individual" ? "Individual" : "Team"}</td>
                            <td data-label="Role"><PlatformStatus value={membership.role} /></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="platform-form-note">This user does not belong to a workspace.</p>
                )}
              </section>
              <div className="system-admin-account-actions">
                {user.is_active || !user.is_initialized ? (
                  <Button
                    label={user.is_active ? "Create password-reset link" : "Replace activation link"}
                    icon={<KeyRound size={16} />}
                    variant="ghost"
                    isDisabled={busy}
                    onClick={() => void generateLink(user.is_active ? "password_reset" : "activation")}
                  />
                ) : null}
                {user.is_active || user.is_initialized ? (
                  <Button
                    label={user.is_active ? "Deactivate account" : "Activate account"}
                    variant={user.is_active ? "destructive" : "primary"}
                    isDisabled={busy}
                    onClick={() => setNextActive(!user.is_active)}
                  />
                ) : null}
              </div>
              {link ? <ActionLinkCard link={link} /> : null}
            </>
          ) : null}
        </div>
      </DialogFrame>
      <ConfirmDialog
        open={user !== null && nextActive !== null}
        title={nextActive ? "Activate this account?" : "Deactivate this account?"}
        description={
          nextActive
            ? `${user ? displayName(user) : "This user"} will be able to sign in again. Withdrawn assignments are not restored.`
            : `${user ? displayName(user) : "This user"} will be signed out, unused account links will be revoked, and mutable assignments will be withdrawn.`
        }
        confirmLabel={nextActive ? "Activate account" : "Deactivate account"}
        destructive={!nextActive}
        busy={busy}
        onCancel={() => setNextActive(null)}
        onConfirm={() => changeStatus()}
      />
    </>
  );
}

function UsersAdministration(): ReactElement {
  const loadIdRef = useRef(0);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "active" | "inactive">("all");
  const [workspaceFilter, setWorkspaceFilter] = useState("");
  const [query, setQuery] = useState<AdminUserListParams>({
    status: "all",
    page: 1,
    pageSize: PAGE_SIZE,
  });
  const [page, setPage] = useState<AdminUserPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [createOpen, setCreateOpen] = useState(false);
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null);

  const load = useCallback(async (): Promise<void> => {
    void refreshVersion;
    const requestId = ++loadIdRef.current;
    setLoading(true);
    setError(null);
    try {
      const nextPage = await listAdminUsers(query);
      if (requestId === loadIdRef.current) setPage(nextPage);
    } catch (caught) {
      if (requestId === loadIdRef.current) {
        setError(messageFrom(caught, "Users could not be loaded."));
      }
    } finally {
      if (requestId === loadIdRef.current) setLoading(false);
    }
  }, [query, refreshVersion]);

  useEffect(() => {
    void load();
    return () => {
      loadIdRef.current += 1;
    };
  }, [load]);

  function applyFilters(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const workspaceId = workspaceFilter.trim() ? Number(workspaceFilter) : undefined;
    if (workspaceId !== undefined && (!Number.isInteger(workspaceId) || workspaceId < 1)) {
      setError("Workspace ID must be a positive whole number.");
      return;
    }
    setStatus(null);
    setQuery({
      search: search.trim() || undefined,
      status: statusFilter,
      workspaceId,
      page: 1,
      pageSize: PAGE_SIZE,
    });
  }

  function updateListedUser(updated: AdminUserDetail): void {
    setStatus(`${displayName(updated)} is now ${updated.is_active ? "active" : "inactive"}.`);
    setQuery((current) => ({ ...current, page: 1 }));
  }

  const currentPage = page?.page ?? query.page ?? 1;
  const totalPages = Math.max(1, Math.ceil((page?.total ?? 0) / (page?.page_size ?? PAGE_SIZE)));

  return (
    <>
      <PlatformSection
        title="Deployment users"
        description="Manage account access across the deployment. Workspace roles remain managed within each team."
        action={
          <Button
            label="Create account"
            icon={<Plus size={16} />}
            variant="primary"
            onClick={() => setCreateOpen(true)}
          />
        }
      >
        <form className="system-admin-user-filters" aria-label="Filter users" onSubmit={applyFilters}>
          <label>
            Search
            <input
              type="search"
              value={search}
              placeholder="Username, name, or email"
              onChange={(event) => setSearch(event.target.value)}
            />
          </label>
          <label>
            Account status
            <select
              value={statusFilter}
              onChange={(event) => setStatusFilter(event.target.value as typeof statusFilter)}
            >
              <option value="all">All accounts</option>
              <option value="active">Active</option>
              <option value="inactive">Inactive</option>
            </select>
          </label>
          <label>
            Workspace ID
            <input
              type="number"
              min={1}
              step={1}
              inputMode="numeric"
              value={workspaceFilter}
              placeholder="Any workspace"
              onChange={(event) => setWorkspaceFilter(event.target.value)}
            />
          </label>
          <div className="system-admin-filter-actions">
            <Button label="Apply filters" type="submit" variant="primary" isDisabled={loading} />
            <Button
              label="Refresh users"
              icon={<RefreshCw size={16} />}
              isIconOnly
              variant="ghost"
              isDisabled={loading}
              onClick={() => setRefreshVersion((value) => value + 1)}
            />
          </div>
        </form>
        {error ? <Banner status="error" title="Users unavailable" description={error} /> : null}
        {status ? <p className="platform-form-success" role="status">{status}</p> : null}
        {loading && !page ? <p role="status">Loading users…</p> : null}
        {page && page.items.length ? (
          <>
            <div className="platform-table-scroll" tabIndex={0} aria-label="Deployment users">
              <table className="platform-table platform-table--summary system-admin-users-table">
                <thead>
                  <tr>
                    <th scope="col">User</th>
                    <th scope="col">Status</th>
                    <th scope="col">Access</th>
                    <th scope="col">Workspaces</th>
                    <th scope="col">Last login</th>
                    <th scope="col"><span className="sr-only">Actions</span></th>
                  </tr>
                </thead>
                <tbody>
                  {page.items.map((user) => (
                    <tr key={user.id}>
                      <td data-label="User" data-priority="identity">
                        <strong>{displayName(user)}</strong>
                        <span>{user.username}{user.email ? ` · ${user.email}` : ""}</span>
                      </td>
                      <td data-label="Status" data-priority="status">
                        <PlatformStatus value={user.is_active ? "active" : "inactive"} />
                      </td>
                      <td data-label="Access">{user.is_superuser ? "Superuser" : "Standard user"}</td>
                      <td data-label="Workspaces">{user.membership_count}</td>
                      <td data-label="Last login">{formatDate(user.last_login_at)}</td>
                      <td data-label="Actions" data-priority="action">
                        <Button
                          label={`View ${displayName(user)}`}
                          variant="ghost"
                          onClick={() => setSelectedUserId(user.id)}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="system-admin-pagination" aria-label="User pagination">
              <span>{page.total} {page.total === 1 ? "user" : "users"} · Page {currentPage} of {totalPages}</span>
              <div>
                <Button
                  label="Previous page"
                  variant="ghost"
                  isDisabled={loading || currentPage <= 1}
                  onClick={() => setQuery((current) => ({ ...current, page: Math.max(1, currentPage - 1) }))}
                />
                <Button
                  label="Next page"
                  variant="ghost"
                  isDisabled={loading || currentPage >= totalPages}
                  onClick={() => setQuery((current) => ({ ...current, page: currentPage + 1 }))}
                />
              </div>
            </div>
          </>
        ) : page && !loading ? (
          <PlatformEmpty
            title="No users match these filters"
            detail="Change the search, account status, or workspace filter and try again."
          />
        ) : null}
      </PlatformSection>
      <CreateUserDialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={(result) => {
          setStatus(`${displayName(result.user)} was created.`);
          setRefreshVersion((value) => value + 1);
        }}
      />
      <UserDetailDialog
        userId={selectedUserId}
        onClose={() => setSelectedUserId(null)}
        onChanged={updateListedUser}
      />
    </>
  );
}

function InstanceSettingsAdministration(): ReactElement {
  const requestIdRef = useRef(0);
  const [settings, setSettings] = useState<AdminSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    const requestId = ++requestIdRef.current;
    setLoading(true);
    setError(null);
    try {
      const nextSettings = await getAdminSettings();
      if (requestId === requestIdRef.current) setSettings(nextSettings);
    } catch (caught) {
      if (requestId === requestIdRef.current) {
        setError(messageFrom(caught, "Instance settings could not be loaded."));
      }
    } finally {
      if (requestId === requestIdRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    return () => {
      requestIdRef.current += 1;
    };
  }, [load]);

  async function save(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!settings) return;
    if (
      !Number.isInteger(settings.default_invite_expiry_minutes) ||
      settings.default_invite_expiry_minutes < INVITE_EXPIRY_MIN ||
      settings.default_invite_expiry_minutes > INVITE_EXPIRY_MAX
    ) {
      setError(`Default invite expiry must be between ${INVITE_EXPIRY_MIN} and ${INVITE_EXPIRY_MAX} minutes.`);
      return;
    }
    if (
      !Number.isInteger(settings.account_action_expiry_minutes) ||
      settings.account_action_expiry_minutes < ACTION_EXPIRY_MIN ||
      settings.account_action_expiry_minutes > ACTION_EXPIRY_MAX
    ) {
      setError(`Account-link expiry must be between ${ACTION_EXPIRY_MIN} and ${ACTION_EXPIRY_MAX} minutes.`);
      return;
    }
    setSaving(true);
    setError(null);
    setStatus(null);
    try {
      const updated = await updateAdminSettings({
        allow_self_registration: settings.allow_self_registration,
        default_invite_expiry_minutes: settings.default_invite_expiry_minutes,
        account_action_expiry_minutes: settings.account_action_expiry_minutes,
      });
      setSettings(updated);
      setStatus("Instance policy saved and is now effective.");
    } catch (caught) {
      setError(messageFrom(caught, "Instance settings could not be saved."));
    } finally {
      setSaving(false);
    }
  }

  return (
    <PlatformSection
      title="Instance settings"
      description="Set deployment-wide account policy. Infrastructure details are shown for reference and remain environment-managed."
      action={
        <Button
          label="Refresh settings"
          icon={<RefreshCw size={16} />}
          isIconOnly
          variant="ghost"
          isDisabled={loading || saving}
          onClick={() => void load()}
        />
      }
    >
      {error ? <Banner status="error" title="Instance settings unavailable" description={error} /> : null}
      {loading && !settings ? <p role="status">Loading instance settings…</p> : null}
      {settings ? (
        <div className="system-admin-settings-layout">
          <form
            className="platform-settings-form system-admin-policy-form"
            noValidate
            onSubmit={(event) => void save(event)}
          >
            <label className="system-admin-checkbox-row">
              <input
                type="checkbox"
                checked={settings.allow_self_registration}
                disabled={saving}
                onChange={(event) => {
                  setStatus(null);
                  setSettings({ ...settings, allow_self_registration: event.target.checked });
                }}
              />
              <span>
                Allow self-registration
                <small>When disabled, new accounts require an invitation or a superuser-generated activation link.</small>
              </span>
            </label>
            <label>
              Default invitation expiry (minutes)
              <input
                type="number"
                min={INVITE_EXPIRY_MIN}
                max={INVITE_EXPIRY_MAX}
                step={1}
                value={settings.default_invite_expiry_minutes}
                disabled={saving}
                aria-describedby="default-invite-expiry-help"
                onChange={(event) => {
                  setStatus(null);
                  setSettings({ ...settings, default_invite_expiry_minutes: Number(event.target.value) });
                }}
              />
              <small id="default-invite-expiry-help">60–43,200 minutes. The default is 10,080 minutes (7 days).</small>
            </label>
            <label>
              Account-link expiry (minutes)
              <input
                type="number"
                min={ACTION_EXPIRY_MIN}
                max={ACTION_EXPIRY_MAX}
                step={1}
                value={settings.account_action_expiry_minutes}
                disabled={saving}
                aria-describedby="account-action-expiry-help"
                onChange={(event) => {
                  setStatus(null);
                  setSettings({ ...settings, account_action_expiry_minutes: Number(event.target.value) });
                }}
              />
              <small id="account-action-expiry-help">15–1,440 minutes. This applies to activation and password-reset links.</small>
            </label>
            <div className="platform-settings-actions">
              <Button
                label={saving ? "Saving…" : "Save policy"}
                type="submit"
                variant="primary"
                isDisabled={saving}
                isLoading={saving}
              />
              {status ? <span role="status">{status}</span> : null}
            </div>
          </form>
          <section className="system-admin-runtime" aria-labelledby="system-admin-runtime-heading">
            <h3 id="system-admin-runtime-heading">Runtime summary</h3>
            <p>Read-only deployment information. Secrets, credentials, URLs, and storage endpoints are never exposed here.</p>
            <dl className="system-admin-detail-list">
              <div><dt>Deployment profile</dt><dd>{settings.deployment_profile}</dd></div>
              <div><dt>Storage backend</dt><dd>{settings.storage_backend}</dd></div>
              <div><dt>Storage encryption</dt><dd>{settings.storage_encryption}</dd></div>
              <div><dt>Task execution</dt><dd>{settings.task_execution}</dd></div>
              <div><dt>JWT lifetime</dt><dd>{settings.jwt_lifetime_minutes} minutes</dd></div>
            </dl>
          </section>
        </div>
      ) : null}
    </PlatformSection>
  );
}

function FutureAdministrationSection({ id, label }: { id: AdminSectionId; label: string }): ReactElement {
  return (
    <PlatformSection
      title={label}
      description="This surface is reserved for deployment superusers and never grants workspace permissions."
    >
      <div className="system-administration-placeholder" data-section={id}>
        <ServerCog size={22} aria-hidden="true" />
        <PlatformEmpty
          title={`${label} integration is not configured`}
          detail="Connect the corresponding deployment service before exposing operational commands."
        />
      </div>
    </PlatformSection>
  );
}

export default function SystemAdministration({
  pathname,
  onNavigate,
}: {
  pathname: string;
  onNavigate: (path: string) => void;
}): ReactElement {
  const requested = pathname.match(/^\/admin\/([^/]+)/)?.[1];
  const active = SECTIONS.find((section) => section.id === requested) ?? SECTIONS[0];

  return (
    <main id="main-content" className="module-workspace-main" tabIndex={-1}>
      <div className="platform-page system-administration-page">
        <PlatformPageHeader
          title="System administration"
          description="Deployment-wide account access, instance policy, and operations."
        />
        <nav className="platform-subnav" aria-label="System administration">
          {SECTIONS.map((section) => (
            <a
              key={section.id}
              href={`/admin/${section.id}`}
              aria-current={active.id === section.id ? "page" : undefined}
              onClick={(event) => {
                if (!shouldHandleSpaClick(event)) return;
                event.preventDefault();
                onNavigate(`/admin/${section.id}`);
              }}
            >
              {section.label}
            </a>
          ))}
        </nav>
        {active.id === "users" ? <UsersAdministration /> : null}
        {active.id === "settings" ? <InstanceSettingsAdministration /> : null}
        {active.id !== "users" && active.id !== "settings" ? (
          <FutureAdministrationSection id={active.id} label={active.label} />
        ) : null}
      </div>
    </main>
  );
}
