// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  AdminSettings,
  AdminUserDetail,
  AdminUserPage,
} from "@/types/api";

import SystemAdministration from "./SystemAdministration";

const mocks = vi.hoisted(() => ({
  createAdminActivationLink: vi.fn(),
  createAdminPasswordResetLink: vi.fn(),
  createAdminUser: vi.fn(),
  getAdminSettings: vi.fn(),
  getAdminUser: vi.fn(),
  listAdminUsers: vi.fn(),
  setAdminUserStatus: vi.fn(),
  updateAdminSettings: vi.fn(),
}));

vi.mock("@/api/client", () => mocks);

const userDetail: AdminUserDetail = {
  id: 12,
  username: "ada",
  display_name: "Ada Lovelace",
  email: "ada@example.test",
  is_active: true,
  is_initialized: true,
  is_superuser: false,
  last_login_at: "2026-07-30T14:00:00Z",
  membership_count: 1,
  memberships: [
    {
      workspace_id: 4,
      workspace_name: "Evidence team",
      workspace_kind: "team",
      role: "manager",
    },
  ],
};

const userPage: AdminUserPage = {
  items: [userDetail],
  total: 21,
  page: 1,
  page_size: 20,
};

const settings: AdminSettings = {
  allow_self_registration: true,
  default_invite_expiry_minutes: 10_080,
  account_action_expiry_minutes: 60,
  deployment_profile: "production",
  storage_backend: "azure-blob",
  storage_encryption: "provider-managed",
  task_execution: "worker",
  jwt_lifetime_minutes: 30,
};

beforeEach(() => {
  vi.clearAllMocks();
  mocks.listAdminUsers.mockResolvedValue(userPage);
  mocks.getAdminUser.mockResolvedValue(userDetail);
  mocks.getAdminSettings.mockResolvedValue(settings);
  mocks.updateAdminSettings.mockResolvedValue(settings);
  mocks.setAdminUserStatus.mockResolvedValue({ ...userDetail, is_active: false });
  mocks.createAdminUser.mockResolvedValue({
    user: { ...userDetail, id: 13, username: "grace", display_name: "Grace Hopper", is_active: false },
    action: {
      purpose: "activation",
      url: "/account-actions/activate-token",
      expires_at: "2026-08-01T15:00:00Z",
    },
  });
  Object.defineProperty(window, "requestAnimationFrame", {
    configurable: true,
    value: (callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    },
  });
  Object.defineProperty(window, "cancelAnimationFrame", {
    configurable: true,
    value: vi.fn(),
  });
});

afterEach(cleanup);

describe("SystemAdministration", () => {
  it("defaults to the deployment user directory and applies server-side filters", async () => {
    const user = userEvent.setup();
    render(<SystemAdministration pathname="/admin" onNavigate={vi.fn()} />);

    expect(await screen.findByText("Ada Lovelace")).toBeTruthy();
    expect(mocks.listAdminUsers).toHaveBeenCalledWith({
      status: "all",
      page: 1,
      pageSize: 20,
    });

    await user.type(screen.getByLabelText("Search"), "ada");
    await user.selectOptions(screen.getByLabelText("Account status"), "active");
    await user.type(screen.getByLabelText("Workspace ID"), "4");
    await user.click(screen.getByRole("button", { name: "Apply filters" }));

    await waitFor(() =>
      expect(mocks.listAdminUsers).toHaveBeenLastCalledWith({
        search: "ada",
        status: "active",
        workspaceId: 4,
        page: 1,
        pageSize: 20,
      }),
    );
  });

  it("creates an inactive account and presents its one-time activation link", async () => {
    const user = userEvent.setup();
    render(<SystemAdministration pathname="/admin/users" onNavigate={vi.fn()} />);
    await screen.findByText("Ada Lovelace");

    const createButtons = screen.getAllByRole("button", { name: "Create account" });
    await user.click(createButtons[createButtons.length - 1]);
    await user.type(screen.getByLabelText("Username"), "grace");
    await user.type(screen.getByLabelText("Display name"), "Grace Hopper");
    await user.type(screen.getByLabelText("Email"), "grace@example.test");
    const submitButtons = screen.getAllByRole("button", { name: "Create account" });
    await user.click(submitButtons[submitButtons.length - 1]);

    await waitFor(() =>
      expect(mocks.createAdminUser).toHaveBeenCalledWith({
        username: "grace",
        display_name: "Grace Hopper",
        email: "grace@example.test",
      }),
    );
    expect(await screen.findByText("Account created")).toBeTruthy();
    expect((screen.getByLabelText("Activation link") as HTMLInputElement).value).toBe(
      `${window.location.origin}/account-actions/activate-token`,
    );
  });

  it("offers activation only for an uninitialized inactive account", async () => {
    const user = userEvent.setup();
    mocks.getAdminUser.mockResolvedValue({
      ...userDetail,
      is_active: false,
      is_initialized: false,
    });
    mocks.createAdminActivationLink.mockResolvedValue({
      purpose: "activation",
      url: "/account-actions/replacement-token",
      expires_at: "2026-08-01T15:00:00Z",
    });
    render(<SystemAdministration pathname="/admin/users" onNavigate={vi.fn()} />);
    await screen.findByText("Ada Lovelace");

    await user.click(screen.getByRole("button", { name: "View Ada Lovelace" }));
    await screen.findByRole("button", { name: "Replace activation link" });
    expect(screen.queryByRole("button", { name: "Activate account" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Replace activation link" }));

    await waitFor(() => expect(mocks.createAdminActivationLink).toHaveBeenCalledWith(12));
    expect((screen.getByLabelText("Activation link") as HTMLInputElement).value).toBe(
      `${window.location.origin}/account-actions/replacement-token`,
    );
  });

  it("offers direct reactivation only for an initialized inactive account", async () => {
    const user = userEvent.setup();
    mocks.getAdminUser.mockResolvedValue({
      ...userDetail,
      is_active: false,
      is_initialized: true,
    });
    render(<SystemAdministration pathname="/admin/users" onNavigate={vi.fn()} />);
    await screen.findByText("Ada Lovelace");

    await user.click(screen.getByRole("button", { name: "View Ada Lovelace" }));
    expect(await screen.findByRole("button", { name: "Activate account" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Replace activation link" })).toBeNull();
  });

  it("shows all memberships and confirms account deactivation", async () => {
    const user = userEvent.setup();
    render(<SystemAdministration pathname="/admin/users" onNavigate={vi.fn()} />);
    await screen.findByText("Ada Lovelace");

    await user.click(screen.getByRole("button", { name: "View Ada Lovelace" }));
    expect(await screen.findByRole("heading", { name: "Workspace memberships" })).toBeTruthy();
    expect(screen.getByText("Evidence team")).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Deactivate account" }));
    expect(screen.getByRole("heading", { name: "Deactivate this account?" })).toBeTruthy();
    const confirmButtons = screen.getAllByRole("button", { name: "Deactivate account" });
    await user.click(confirmButtons[confirmButtons.length - 1]);

    await waitFor(() => expect(mocks.setAdminUserStatus).toHaveBeenCalledWith(12, false));
    expect(await screen.findByText("Ada Lovelace is now inactive.")).toBeTruthy();
    await waitFor(() => expect(mocks.listAdminUsers).toHaveBeenCalledTimes(2));
  });

  it("edits only mutable instance policy while displaying safe runtime fields", async () => {
    const user = userEvent.setup();
    mocks.updateAdminSettings.mockResolvedValue({
      ...settings,
      allow_self_registration: false,
      default_invite_expiry_minutes: 1_440,
      account_action_expiry_minutes: 120,
    });
    render(<SystemAdministration pathname="/admin/settings" onNavigate={vi.fn()} />);

    expect(await screen.findByText("Runtime summary")).toBeTruthy();
    expect(screen.getByText("azure-blob")).toBeTruthy();
    expect(screen.getByText("provider-managed")).toBeTruthy();

    await user.click(screen.getByLabelText(/Allow self-registration/));
    fireEvent.change(screen.getByLabelText(/Default invitation expiry/), {
      target: { value: "1440" },
    });
    fireEvent.change(screen.getByLabelText(/Account-link expiry/), {
      target: { value: "120" },
    });
    await user.click(screen.getByRole("button", { name: "Save policy" }));

    await waitFor(() =>
      expect(mocks.updateAdminSettings).toHaveBeenCalledWith({
        allow_self_registration: false,
        default_invite_expiry_minutes: 1_440,
        account_action_expiry_minutes: 120,
      }),
    );
    expect(await screen.findByText(/now effective/)).toBeTruthy();

    await user.click(screen.getByLabelText(/Allow self-registration/));
    expect(screen.queryByText(/now effective/)).toBeNull();
  });

  it("rejects out-of-range policy values before sending them", async () => {
    const user = userEvent.setup();
    render(<SystemAdministration pathname="/admin/settings" onNavigate={vi.fn()} />);
    await screen.findByText("Runtime summary");

    fireEvent.change(screen.getByLabelText(/Default invitation expiry/), {
      target: { value: "59" },
    });
    await user.click(screen.getByRole("button", { name: "Save policy" }));

    expect(await screen.findByText(/between 60 and 43200 minutes/)).toBeTruthy();
    expect(mocks.updateAdminSettings).not.toHaveBeenCalled();
  });

  it("preserves future system administration placeholders", () => {
    render(<SystemAdministration pathname="/admin/health" onNavigate={vi.fn()} />);
    expect(screen.getByText("System health integration is not configured")).toBeTruthy();
    expect(mocks.listAdminUsers).not.toHaveBeenCalled();
    expect(mocks.getAdminSettings).not.toHaveBeenCalled();
  });
});
