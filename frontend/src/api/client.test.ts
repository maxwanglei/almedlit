import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  getSessionGeneration,
  publishSessionChange,
  SESSION_EVENT_STORAGE_KEY,
  subscribeSessionChanges,
} from "@/auth/session";

import {
  acceptInvite,
  archiveArtifactPackage,
  archiveBaseModel,
  downloadArtifactPackageFile,
  downloadExport,
  downloadSubmission,
  getMe,
  getProjectIaa,
  getWorkspaceCapabilities,
  importBaseModel,
  listAdminUsers,
  listBaseModels,
  listDocumentSummaries,
  listInferenceRuns,
  listInferenceWindows,
  login,
  request,
  requestBlob,
  setBaseModelReadiness,
  uploadBaseModel,
} from "./client";

function installStorage(): void {
  const values = new Map<string, string>();
  const localStorage = {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
    clear: () => values.clear(),
    key: (index: number) => Array.from(values.keys())[index] ?? null,
    get length() {
      return values.size;
    },
  } as Storage;

  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      localStorage,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    },
  });
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: { cookie: "" },
  });
}

const sessionProbes = [
  { name: "getMe", path: "/api/auth/me", request: getMe },
  {
    name: "getWorkspaceCapabilities",
    path: "/api/workspaces/17/capabilities",
    request: () => getWorkspaceCapabilities(17),
  },
];

describe("API authentication failures", () => {
  beforeEach(() => {
    installStorage();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    Reflect.deleteProperty(globalThis, "window");
    Reflect.deleteProperty(globalThis, "document");
  });

  it.each([
    { name: "document summaries", load: () => listDocumentSummaries(7), path: "/api/documents", marker: "summary=true" },
    { name: "inference runs", load: () => listInferenceRuns(7), path: "/api/projects/7/inference/runs", marker: "limit=100" },
    { name: "inference windows", load: () => listInferenceWindows(8), path: "/api/inference/runs/8/windows", marker: "limit=100" },
  ])("loads all bounded pages for $name", async ({ load, path, marker }) => {
    const first = Array.from({ length: 100 }, (_, id) => ({ id }));
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(first), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify([{ id: 100 }]), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    expect(await load()).toHaveLength(101);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0]?.[0]).toContain(path);
    expect(fetchMock.mock.calls[0]?.[0]).toContain(marker);
    expect(fetchMock.mock.calls[1]?.[0]).toContain("offset=100");
  });

  it.each(sessionProbes)("broadcasts one anonymous session after a $name 401 response", async ({ request, path }) => {
    publishSessionChange(true);
    const generation = getSessionGeneration();
    const observed: boolean[] = [];
    const unsubscribe = subscribeSessionChanges((authenticated) =>
      observed.push(authenticated),
    );
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Session expired" }), {
        status: 401,
        statusText: "Unauthorized",
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    try {
      await expect(request()).rejects.toMatchObject({
        name: "ApiError",
        status: 401,
        message: "Session expired",
      });

      expect(observed).toEqual([false]);
      expect(getSessionGeneration()).toBe(generation + 1);
      expect(JSON.parse(window.localStorage.getItem(SESSION_EVENT_STORAGE_KEY) ?? "null"))
        .toMatchObject({ authenticated: false });
      expect(fetchMock).toHaveBeenCalledWith(
        path,
        expect.objectContaining({ credentials: "same-origin" }),
      );
      expect(fetchMock.mock.calls[0]?.[1]?.headers).not.toHaveProperty("Authorization");
    } finally {
      unsubscribe();
    }
  });

  describe.each(sessionProbes)("$name session probe", ({ request, path }) => {
    it.each([403, 502])("preserves the session after an HTTP %i response", async (status) => {
      publishSessionChange(true);
      const generation = getSessionGeneration();
      const sessionEvent = window.localStorage.getItem(SESSION_EVENT_STORAGE_KEY);
      const observed = vi.fn();
      const unsubscribe = subscribeSessionChanges(observed);
      const fetchMock = vi.fn().mockResolvedValue(new Response(
        JSON.stringify({ detail: "Session could not be checked" }),
        { status, headers: { "Content-Type": "application/json" } },
      ));
      vi.stubGlobal("fetch", fetchMock);

      try {
        await expect(request()).rejects.toMatchObject({ name: "ApiError", status });

        expect(observed).not.toHaveBeenCalled();
        expect(getSessionGeneration()).toBe(generation);
        expect(window.localStorage.getItem(SESSION_EVENT_STORAGE_KEY)).toBe(sessionEvent);
        expect(fetchMock).toHaveBeenCalledWith(path, expect.objectContaining({ credentials: "same-origin" }));
      } finally {
        unsubscribe();
      }
    });

    it("preserves the session after a network failure", async () => {
      publishSessionChange(true);
      const generation = getSessionGeneration();
      const sessionEvent = window.localStorage.getItem(SESSION_EVENT_STORAGE_KEY);
      const observed = vi.fn();
      const unsubscribe = subscribeSessionChanges(observed);
      const failure = new TypeError("Failed to fetch");
      const fetchMock = vi.fn().mockRejectedValue(failure);
      vi.stubGlobal("fetch", fetchMock);

      try {
        await expect(request()).rejects.toBe(failure);

        expect(observed).not.toHaveBeenCalled();
        expect(getSessionGeneration()).toBe(generation);
        expect(window.localStorage.getItem(SESSION_EVENT_STORAGE_KEY)).toBe(sessionEvent);
        expect(fetchMock).toHaveBeenCalledWith(path, expect.objectContaining({ credentials: "same-origin" }));
      } finally {
        unsubscribe();
      }
    });
  });

  it.each([
    { name: "login", attempt: () => login("alice", "wrong password") },
    {
      name: "invite credentials",
      attempt: () => acceptInvite("invite-token", {
        username: "alice", password: "wrong password", create_account: false,
      }),
    },
  ])("preserves the current session after rejected $name", async ({ attempt }) => {
    publishSessionChange(true);
    const generation = getSessionGeneration();
    const storedEvent = window.localStorage.getItem(SESSION_EVENT_STORAGE_KEY);
    const observed = vi.fn();
    const unsubscribe = subscribeSessionChanges(observed);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: "Invalid username or password" }),
      { status: 401, headers: { "Content-Type": "application/json" } },
    )));
    try {
      await expect(attempt()).rejects.toMatchObject({ status: 401 });
      expect(observed).not.toHaveBeenCalled();
      expect(getSessionGeneration()).toBe(generation);
      expect(window.localStorage.getItem(SESSION_EVENT_STORAGE_KEY)).toBe(storedEvent);
    } finally {
      unsubscribe();
    }
  });

  it.each([{}, { username: "alice" }, { password: "password" }, { username: "", password: "" }])(
    "invalidates a rejected cookie-session invite with payload %j", async (payload) => {
      publishSessionChange(true);
      const observed = vi.fn();
      const unsubscribe = subscribeSessionChanges(observed);
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
        JSON.stringify({ detail: "Session expired" }),
        { status: 401, headers: { "Content-Type": "application/json" } },
      )));
      try {
        await expect(acceptInvite("invite-token", payload)).rejects.toMatchObject({ status: 401 });
        expect(observed).toHaveBeenCalledExactlyOnceWith(false);
      } finally {
        unsubscribe();
      }
    },
  );

  it("does not clear a newer session when an older request returns 401", async () => {
    publishSessionChange(true);
    const observed: boolean[] = [];
    const unsubscribe = subscribeSessionChanges((authenticated) =>
      observed.push(authenticated),
    );
    let resolveResponse!: (response: Response) => void;
    const fetchMock = vi.fn().mockImplementation(
      () =>
        new Promise<Response>((resolve) => {
          resolveResponse = resolve;
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const oldRequest = getMe();
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    publishSessionChange(true);
    resolveResponse(
      new Response(JSON.stringify({ detail: "Old session expired" }), {
        status: 401,
        statusText: "Unauthorized",
        headers: { "Content-Type": "application/json" },
      }),
    );

    await expect(oldRequest).rejects.toMatchObject({ status: 401 });
    unsubscribe();
    expect(observed).toEqual([true]);
  });

  it("omits the all-user status filter and sends explicit active filters", async () => {
    const page = { items: [], total: 0, page: 1, page_size: 20 };
    const fetchMock = vi.fn().mockImplementation(
      async () =>
        new Response(JSON.stringify(page), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await listAdminUsers({ status: "all", page: 1, pageSize: 20 });
    await listAdminUsers({ status: "active", page: 2, pageSize: 20 });

    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "/api/admin/users?page=1&page_size=20",
    );
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      "/api/admin/users?status=active&page=2&page_size=20",
    );
  });

  it("accepts an invite with the HttpOnly cookie session", async () => {
    const observed: boolean[] = [];
    const unsubscribe = subscribeSessionChanges((authenticated) =>
      observed.push(authenticated),
    );
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ authenticated: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await acceptInvite("invite-token");
    unsubscribe();

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/invites/accept",
      expect.objectContaining({
        method: "POST",
        credentials: "same-origin",
      }),
    );
    expect(fetchMock.mock.calls[0]?.[1]?.headers).not.toHaveProperty("Authorization");
    expect(observed).toEqual([true]);
  });

  it("downloads submission bytes with the cookie session", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("submission contents", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const blob = await downloadSubmission(17);

    expect(await blob.text()).toBe("submission contents");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/submissions/17/download",
      expect.objectContaining({
        credentials: "same-origin",
      }),
    );
    expect(fetchMock.mock.calls[0]?.[1]?.headers).not.toHaveProperty("Authorization");
  });

  it("downloads export bytes with the cookie session", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("training data", {
        status: 200,
        headers: { "Content-Type": "application/x-ndjson" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const blob = await downloadExport(29);

    expect(await blob.text()).toBe("training data");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/exports/29/download",
      expect.objectContaining({
        credentials: "same-origin",
      }),
    );
    expect(fetchMock.mock.calls[0]?.[1]?.headers).not.toHaveProperty("Authorization");
  });

  it("downloads an authorized package file without exposing a storage key", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response("weights", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const blob = await downloadArtifactPackageFile(41, 73);

    expect(await blob.text()).toBe("weights");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/artifact-packages/41/files/73/download",
      expect.objectContaining({
        credentials: "same-origin",
      }),
    );
    expect(fetchMock.mock.calls[0]?.[1]?.headers).not.toHaveProperty("Authorization");
  });

  it("requests the manager-approved archive lifecycle", async () => {
    document.cookie = "al_medlit_csrf=csrf-value";
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      package_id: 41,
      archived_at: "2026-01-01T00:00:00Z",
      purge_after: "2026-01-08T00:00:00Z",
    }), { status: 202, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);

    await archiveArtifactPackage(41);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/artifact-packages/41/archive",
      expect.objectContaining({
        method: "POST",
        credentials: "same-origin",
        headers: expect.objectContaining({ "X-CSRF-Token": "csrf-value" }),
      }),
    );
  });

  it("normalizes base-model packages and uses audited manager lifecycle endpoints", async () => {
    const wireAsset = {
      id: 61,
      project_id: 7,
      package_id: 51,
      provider: "huggingface",
      source_model_id: "lab/tiny",
      exact_revision: "deadbeef",
      display_name: "Tiny model",
      model_family: "llm_finetune",
      model_type: "causal_lm",
      license_name: "Apache-2.0",
      license_url: null,
      license_terms_sha256: null,
      access_mode: "execution_only",
      readiness: "ready",
      archived_at: null,
      metadata: {},
      created_at: "2026-01-01T00:00:00Z",
      package: {
        id: 51,
        project_id: 7,
        package_kind: "base_model",
        package_format: "safetensors",
        schema_version: "1",
        model_family: "llm_finetune",
        model_type: "causal_lm",
        readiness: "ready",
        deployable: true,
        manifest_digest: "a".repeat(64),
        logical_size_bytes: 42,
        file_count: 1,
        metadata: {},
        files: [],
        references: [],
        retention: { pinned: true, expires_at: null },
        created_at: "2026-01-01T00:00:00Z",
      },
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify([wireAsset]), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ...wireAsset, readiness: "quarantined" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ...wireAsset, readiness: "archived", archived_at: "2026-01-02T00:00:00Z" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }));
    vi.stubGlobal("fetch", fetchMock);

    const assets = await listBaseModels(7, { includeArchived: true });
    expect(assets[0]?.package).toMatchObject({ kind: "base_model", format: "safetensors" });
    await setBaseModelReadiness(61, { readiness: "quarantined", reason: "Review required" });
    await archiveBaseModel(61);

    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/projects/7/base-models?include_archived=true");
    expect(fetchMock.mock.calls[1]?.[0]).toBe("/api/base-models/61/readiness");
    expect(fetchMock.mock.calls[1]?.[1]).toMatchObject({
      method: "PATCH",
      body: JSON.stringify({ readiness: "quarantined", reason: "Review required" }),
    });
    expect(fetchMock.mock.calls[2]?.[0]).toBe("/api/base-models/61/archive");
    expect(fetchMock.mock.calls[2]?.[1]).toMatchObject({ method: "POST" });
  });

  it("stages an exact base revision by JSON import or multipart upload", async () => {
    const wireAsset = {
      id: 62,
      project_id: 7,
      package_id: 52,
      provider: "huggingface",
      source_model_id: "lab/tiny",
      exact_revision: "deadbeef",
      display_name: "Tiny model",
      model_family: "llm_finetune",
      model_type: "causal_lm",
      license_name: "Apache-2.0",
      license_url: null,
      license_terms_sha256: null,
      access_mode: "execution_only",
      readiness: "ready",
      archived_at: null,
      metadata: {},
      created_at: "2026-01-01T00:00:00Z",
      package: {
        id: 52,
        project_id: 7,
        package_kind: "base_model",
        package_format: "safetensors",
        schema_version: "1",
        model_family: "llm_finetune",
        model_type: "causal_lm",
        readiness: "ready",
        deployable: true,
        manifest_digest: "a".repeat(64),
        logical_size_bytes: 7,
        file_count: 1,
        files: [],
        references: [],
        retention: { pinned: true, expires_at: null },
        created_at: "2026-01-01T00:00:00Z",
      },
    };
    const fetchMock = vi.fn().mockImplementation(async () => new Response(JSON.stringify(wireAsset), {
      status: 201,
      headers: { "Content-Type": "application/json" },
    }));
    vi.stubGlobal("fetch", fetchMock);
    const common = {
      provider: "huggingface",
      source_model_id: "lab/tiny",
      exact_revision: "deadbeef",
      display_name: "Tiny model",
      model_family: "llm_finetune" as const,
      model_type: "causal_lm",
      license_name: "Apache-2.0",
      access_mode: "execution_only" as const,
    };

    await importBaseModel(7, { ...common, source_package_id: 51 });
    const file = new File(["weights"], "model.safetensors", {
      type: "application/octet-stream",
    });
    await uploadBaseModel(7, {
      ...common,
      package_format: "safetensors",
      files: [{
        relative_path: "model.safetensors",
        role: "model_file",
        content_type: "application/octet-stream",
      }],
    }, [file]);

    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/projects/7/base-models/import");
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({
      method: "POST",
      body: JSON.stringify({ ...common, source_package_id: 51 }),
      headers: { "Content-Type": "application/json" },
    });
    const uploadRequest = fetchMock.mock.calls[1]?.[1] as RequestInit;
    expect(fetchMock.mock.calls[1]?.[0]).toBe("/api/projects/7/base-models/upload");
    expect(uploadRequest.method).toBe("POST");
    expect(uploadRequest.headers).not.toHaveProperty("Content-Type");
    expect(uploadRequest.body).toBeInstanceOf(FormData);
    const multipart = uploadRequest.body as FormData;
    expect(JSON.parse(String(multipart.get("metadata")))).toMatchObject({
      exact_revision: "deadbeef",
      package_format: "safetensors",
      files: [{ relative_path: "model.safetensors" }],
    });
    expect(multipart.getAll("files")).toHaveLength(1);
  });

  it("selects legacy null-guideline IAA rounds explicitly and numeric rounds by ID", async () => {
    const fetchMock = vi.fn().mockImplementation(async () =>
      new Response("{}", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const baseParams = {
      annotationType: "evidence_block" as const,
      documentId: 11,
      targetVersionId: 13,
      structureVersionId: 17,
    };

    await getProjectIaa(7, { ...baseParams, guidelineVersionId: null });
    await getProjectIaa(7, { ...baseParams, guidelineVersionId: 19 });

    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "/api/projects/7/iaa?annotation_type=evidence_block&document_id=11&target_version_id=13&structure_version_id=17&legacy_guideline=true",
    );
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      "/api/projects/7/iaa?annotation_type=evidence_block&document_id=11&target_version_id=13&structure_version_id=17&guideline_version_id=19",
    );
  });
});

describe("API error messages", () => {
  beforeEach(() => {
    installStorage();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    Reflect.deleteProperty(globalThis, "window");
    Reflect.deleteProperty(globalThis, "document");
  });

  function respondWithError(body: string, status: number, contentType: string | null = "application/json"): void {
    const response = new Response(body, {
      status,
      statusText: "",
      headers: contentType ? { "Content-Type": contentType } : undefined,
    });
    if (contentType === null) response.headers.delete("Content-Type");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));
  }

  it.each([
    { name: "JSON request", call: () => request("/validation") },
    { name: "session probe", call: getMe },
    { name: "login", call: () => login("alice", "example-password") },
    { name: "blob download", call: () => requestBlob("/exports/17/download") },
  ])("shows FastAPI validation messages through a $name", async ({ call }) => {
    respondWithError(JSON.stringify({ detail: [
      { loc: ["body", "username"], msg: "  Field required  ", type: "missing" },
      null,
      7,
      "invalid item",
      {},
      { msg: null },
      { msg: 5 },
      { msg: {} },
      { msg: "   " },
      { loc: ["body", "count"], msg: "Input should be a valid integer", type: "int_parsing" },
    ] }), 422);

    await expect(call()).rejects.toMatchObject({
      name: "ApiError",
      status: 422,
      message: "Field required; Input should be a valid integer",
    });
  });

  it.each([
    { status: 422, message: "Some values are invalid. Check your input and try again." },
    { status: 429, message: "Too many requests. Please wait and try again." },
    { status: 500, message: "The server could not complete the request. Please try again." },
    { status: 502, message: "The server is temporarily unavailable. Please try again." },
    { status: 503, message: "The server is temporarily unavailable. Please try again." },
    { status: 504, message: "The server took too long to respond. Please try again." },
    { status: 418, message: "Request failed (HTTP 418). Please try again." },
  ])("provides a meaningful HTTP $status fallback when statusText and the body are empty", async ({ status, message }) => {
    respondWithError("", status);

    await expect(getMe()).rejects.toMatchObject({ name: "ApiError", status, message });
  });

  it.each([
    { status: 429, contentType: "text/html", body: "<!doctype html><html><body><h1>Rate limited</h1></body></html>", message: "Too many requests. Please wait and try again." },
    { status: 502, contentType: null, body: "<html><body>Bad Gateway</body></html>", message: "The server is temporarily unavailable. Please try again." },
    { status: 504, contentType: "application/json", body: "<h1>Gateway timeout</h1>", message: "The server took too long to respond. Please try again." },
    { status: 502, contentType: "text/plain", body: "<div>Upstream server failed</div>", message: "The server is temporarily unavailable. Please try again." },
  ])("replaces HTTP $status proxy markup served as $contentType with a useful message", async ({ status, contentType, body, message }) => {
    respondWithError(body, status, contentType);

    await expect(requestBlob("/exports/17/download")).rejects.toMatchObject({ status, message });
  });

  it.each([
    { detail: "<h1>Bad Gateway</h1>" },
    { detail: [{ msg: "<div>Proxy failure</div>" }, null, {}, { msg: 7 }] },
    { detail: [{}, null, 7, { msg: " " }, { msg: {} }] },
  ])("does not expose markup or invalid entries from JSON detail $detail", async (body) => {
    respondWithError(JSON.stringify(body), 502);

    await expect(getMe()).rejects.toMatchObject({
      status: 502,
      message: "The server is temporarily unavailable. Please try again.",
    });
  });

  it("preserves safe validation messages when an array also contains markup", async () => {
    respondWithError(JSON.stringify({ detail: [{ msg: "<h1>Proxy error</h1>" }, { msg: "A dataset name is required." }] }), 422);

    await expect(getMe()).rejects.toMatchObject({
      status: 422,
      message: "A dataset name is required.",
    });
  });

  it.each([
    { body: '{"detail":', contentType: "application/json" },
    { body: JSON.stringify({ detail: "   " }), contentType: "application/json" },
    { body: JSON.stringify({ unexpected: "field" }), contentType: "application/json" },
  ])("uses a stable fallback for an unusable JSON body $body", async ({ body, contentType }) => {
    respondWithError(body, 422, contentType);

    await expect(getMe()).rejects.toMatchObject({
      status: 422,
      message: "Some values are invalid. Check your input and try again.",
    });
  });

  it("preserves a structured detail string", async () => {
    respondWithError(JSON.stringify({ detail: "This task key is already in use." }), 409);

    await expect(getMe()).rejects.toMatchObject({ status: 409, message: "This task key is already in use." });
  });

  it("preserves useful plain text including comparison symbols", async () => {
    respondWithError("  Value must be < 10.  ", 422, "text/plain");

    await expect(getMe()).rejects.toMatchObject({ status: 422, message: "Value must be < 10." });
  });

  it("uses the authentication fallback and broadcasts once when reading a 401 body fails", async () => {
    publishSessionChange(true);
    const generation = getSessionGeneration();
    const observed = vi.fn();
    const unsubscribe = subscribeSessionChanges(observed);
    const response = new Response("", { status: 401, statusText: "" });
    const readBody = vi.spyOn(response, "text").mockRejectedValue(new Error("Response stream failed"));
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));

    try {
      await expect(getMe()).rejects.toMatchObject({
        name: "ApiError",
        status: 401,
        message: "Authentication is required. Please sign in again.",
      });

      expect(readBody).toHaveBeenCalledOnce();
      expect(observed).toHaveBeenCalledExactlyOnceWith(false);
      expect(getSessionGeneration()).toBe(generation + 1);
    } finally {
      unsubscribe();
      readBody.mockRestore();
    }
  });

  it("preserves validation messages when only input and context contain markup", async () => {
    respondWithError(JSON.stringify({ detail: [
      {
        loc: ["body", "text"],
        msg: "Value must be plain text",
        input: "<h1>Submitted text</h1>",
        ctx: { example: "<div>Expected text</div>" },
      },
      { loc: ["body", "name"], msg: "Field required", input: { content: "<p>Example</p>" } },
    ] }), 422);

    await expect(getMe()).rejects.toMatchObject({
      status: 422,
      message: "Value must be plain text; Field required",
    });
  });
});
