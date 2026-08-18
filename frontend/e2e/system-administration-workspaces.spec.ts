import { expect, test, type Page, type Route } from "@playwright/test";

const timestamp = "2026-07-31T14:00:00Z";

async function json(route: Route, body: unknown, status = 200): Promise<void> {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

async function installToken(page: Page): Promise<void> {
  await page.addInitScript(() => {
    window.localStorage.setItem("al_medlit_access_token", "e2e-session");
  });
}

test("a membership-less superuser manages deployment users and safe policy", async ({
  page,
}) => {
  await installToken(page);
  let policy = {
    allow_self_registration: true,
    default_invite_expiry_minutes: 10_080,
    account_action_expiry_minutes: 60,
    deployment_profile: "production",
    storage_backend: "azure-blob",
    storage_encryption: "provider-managed",
    task_execution: "worker",
    jwt_lifetime_minutes: 30,
  };

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (!url.pathname.startsWith("/api/")) {
      await route.continue();
      return;
    }
    if (url.pathname === "/api/auth/me") {
      await json(route, {
        user: {
          id: 1,
          username: "root",
          display_name: "System Administrator",
          is_active: true,
          is_superuser: true,
        },
        memberships: [],
      });
      return;
    }
    if (url.pathname === "/api/admin/users" && request.method() === "GET") {
      await json(route, {
        items: [
          {
            id: 12,
            username: "ada",
            display_name: "Ada Lovelace",
            email: "ada@example.test",
            is_active: true,
            is_initialized: true,
            is_superuser: false,
            last_login_at: timestamp,
            membership_count: 2,
            created_at: timestamp,
          },
        ],
        total: 1,
        page: 1,
        page_size: 20,
      });
      return;
    }
    if (url.pathname === "/api/admin/users" && request.method() === "POST") {
      await json(
        route,
        {
          user: {
            id: 13,
            username: "grace",
            display_name: "Grace Hopper",
            email: "grace@example.test",
            is_active: false,
            is_initialized: false,
            is_superuser: false,
            last_login_at: null,
            membership_count: 0,
            created_at: timestamp,
          },
          action: {
            purpose: "activation",
            url: "/account-actions/activate-grace",
            expires_at: "2026-07-31T15:00:00Z",
          },
        },
        201,
      );
      return;
    }
    if (url.pathname === "/api/admin/settings" && request.method() === "GET") {
      await json(route, policy);
      return;
    }
    if (url.pathname === "/api/admin/settings" && request.method() === "PATCH") {
      policy = { ...policy, ...(request.postDataJSON() as typeof policy) };
      await json(route, policy);
      return;
    }
    await json(route, { detail: `Unhandled ${request.method()} ${url.pathname}` }, 404);
  });

  await page.goto("/admin");
  await expect(page).toHaveURL(/\/admin\/users$/);
  await expect(page.getByRole("heading", { name: "System administration" })).toBeVisible();
  await expect(page.getByText("Ada Lovelace", { exact: true })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Workspace" })).toHaveCount(0);

  await page.getByRole("button", { name: "Create account" }).click();
  await page.getByLabel("Username").fill("grace");
  await page.getByLabel("Display name").fill("Grace Hopper");
  await page.getByLabel("Email").fill("grace@example.test");
  await page.getByRole("button", { name: "Create account" }).last().click();
  await expect(page.getByText("Account created", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Activation link")).toHaveValue(
    "http://127.0.0.1:4173/account-actions/activate-grace",
  );
  await page.getByRole("button", { name: "Done" }).click();

  await page.getByRole("link", { name: "Instance settings" }).click();
  await expect(page.getByText("Runtime summary", { exact: true })).toBeVisible();
  await expect(page.getByText("azure-blob", { exact: true })).toBeVisible();
  await page.getByLabel(/Allow self-registration/).uncheck();
  await page.getByLabel(/Account-link expiry/).fill("120");
  await page.getByRole("button", { name: "Save policy" }).click();
  await expect(page.getByText(/now effective/)).toBeVisible();
  expect(policy.allow_self_registration).toBe(false);
  expect(policy.account_action_expiry_minutes).toBe(120);
});

test("an individual owner creates a separate team and manages team access", async ({
  page,
}) => {
  await installToken(page);
  let teamCreated = false;
  let joinCode = "join-old";
  let openInvites: Array<Record<string, unknown>> = [];

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (!url.pathname.startsWith("/api/")) {
      await route.continue();
      return;
    }
    if (url.pathname === "/api/auth/me") {
      await json(route, {
        user: {
          id: 7,
          username: "alice",
          display_name: "Alice",
          is_active: true,
          is_superuser: false,
        },
        memberships: [
          {
            workspace_id: 10,
            workspace_name: "Alice's Workspace",
            workspace_kind: "individual",
            role: "admin",
          },
          ...(teamCreated
            ? [
                {
                  workspace_id: 20,
                  workspace_name: "Evidence team",
                  workspace_kind: "team",
                  role: "admin",
                },
              ]
            : []),
        ],
      });
      return;
    }
    if (/^\/api\/workspaces\/(10|20)\/capabilities$/.test(url.pathname)) {
      const workspaceId = Number(url.pathname.split("/")[3]);
      await json(route, {
        workspace_id: workspaceId,
        preset: "annotate",
        overrides: [],
        effective: ["annotation"],
        blocked: {},
      });
      return;
    }
    if (/^\/api\/workspaces\/(10|20)\/members$/.test(url.pathname)) {
      const workspaceId = Number(url.pathname.split("/")[3]);
      await json(route, [
        {
          id: workspaceId,
          workspace_id: workspaceId,
          user_id: 7,
          username: "alice",
          display_name: "Alice",
          email: "alice@example.test",
          is_active: true,
          role: "admin",
        },
      ]);
      return;
    }
    if (url.pathname === "/api/workspaces" && request.method() === "POST") {
      teamCreated = true;
      await json(route, {
        id: 20,
        name: "Evidence team",
        kind: "team",
        join_code: joinCode,
        capability_preset: "annotate",
        capability_overrides: [],
      });
      return;
    }
    if (url.pathname === "/api/workspaces/20/governance" && request.method() === "GET") {
      await json(route, {
        workspace_id: 20,
        workspace_kind: "team",
        join_code: joinCode,
        default_invite_expiry_minutes: 10_080,
      });
      return;
    }
    if (url.pathname === "/api/workspaces/20/join-requests") {
      await json(route, [
        {
          id: 31,
          workspace_id: 20,
          user_id: 8,
          username: "bob",
          display_name: "Bob Researcher",
          email: "bob@example.test",
          status: "pending",
          message: "Collaborator",
          created_at: timestamp,
        },
      ]);
      return;
    }
    if (url.pathname === "/api/workspaces/20/invites" && request.method() === "GET") {
      await json(route, openInvites);
      return;
    }
    if (url.pathname === "/api/workspaces/20/invites" && request.method() === "POST") {
      openInvites = [
        {
          id: 41,
          workspace_id: 20,
          role: "annotator",
          created_by: 7,
          created_by_username: "alice",
          expires_at: "2026-08-01T14:00:00Z",
          created_at: timestamp,
        },
      ];
      await json(route, {
        id: 41,
        token: "invite-new-member",
        role: "annotator",
        workspace_id: 20,
        expires_at: "2026-08-01T14:00:00Z",
      });
      return;
    }
    if (url.pathname === "/api/workspaces/20/join-code/rotate") {
      joinCode = "join-new";
      await json(route, {
        workspace_id: 20,
        workspace_kind: "team",
        join_code: joinCode,
        default_invite_expiry_minutes: 10_080,
      });
      return;
    }
    if (url.pathname.startsWith("/api/projects")) {
      await json(route, []);
      return;
    }
    if (url.pathname.includes("round-work-contexts")) {
      await json(route, []);
      return;
    }
    await json(route, { detail: `Unhandled ${request.method()} ${url.pathname}` }, 404);
  });

  await page.goto("/workspace-settings");
  await expect(page.getByText("Alice's Workspace · Individual workspace", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Owner", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Create or join a team" }).click();
  await page.getByRole("textbox", { name: /Team name/ }).fill("Evidence team");
  await page.getByRole("button", { name: "Create team" }).click();

  await expect(page.getByRole("combobox", { name: "Workspace" })).toContainText(
    "Evidence team · Administrator",
  );
  await page.goto("/workspace-settings");
  await expect(page.getByText("Team workspace", { exact: false })).toBeVisible();
  await expect(page.getByText("join-old", { exact: true })).toBeVisible();
  await expect(page.getByText("Bob Researcher", { exact: true })).toBeVisible();

  await page.getByRole("combobox", { name: "Expires" }).selectOption("1440");
  await page.getByRole("button", { name: "Create invitation" }).click();
  await expect(page.getByText(/invite-new-member/)).toBeVisible();
  await expect(page.getByRole("cell", { name: "alice", exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Rotate join code" }).click();
  await page.getByRole("button", { name: "Rotate code" }).click();
  await expect(page.getByText("join-new", { exact: true })).toBeVisible();
});

test("public account activation completes once and adopts the new session", async ({
  page,
}) => {
  let completionCount = 0;
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (!url.pathname.startsWith("/api/")) {
      await route.continue();
      return;
    }
    if (url.pathname === "/api/account-actions/activation-token" && request.method() === "GET") {
      await json(route, {
        purpose: "activation",
        username: "new-user",
        display_name: "New User",
        expires_at: "2026-07-31T15:00:00Z",
      });
      return;
    }
    if (url.pathname === "/api/account-actions/activation-token" && request.method() === "POST") {
      completionCount += 1;
      await json(route, { access_token: "activated-session", token_type: "bearer" });
      return;
    }
    if (url.pathname === "/api/auth/me") {
      await json(route, {
        user: {
          id: 99,
          username: "new-user",
          display_name: "New User",
          is_active: true,
          is_superuser: false,
        },
        memberships: [],
      });
      return;
    }
    await json(route, { detail: `Unhandled ${request.method()} ${url.pathname}` }, 404);
  });

  await page.goto("/account-actions/activation-token");
  await expect(page.getByRole("heading", { name: "Activate your account" })).toBeVisible();
  await page.locator('input[name="newPassword"]').fill("strong-password-123");
  await page.locator('input[name="confirmPassword"]').fill("strong-password-123");
  await page.getByRole("button", { name: "Activate account" }).click();

  await expect(page.getByRole("heading", { name: "No modules available" })).toBeVisible();
  expect(completionCount).toBe(1);
  expect(await page.evaluate(() => window.localStorage.getItem("al_medlit_access_token"))).toBe(
    "activated-session",
  );
});
