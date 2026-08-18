import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { clearToken, getToken, setToken, subscribeTokenChanges } from "@/auth/session";

import {
  acceptInvite,
  archiveArtifactPackage,
  archiveBaseModel,
  downloadArtifactPackageFile,
  downloadExport,
  downloadSubmission,
  getMe,
  getProjectIaa,
  importBaseModel,
  listAdminUsers,
  listBaseModels,
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
}

describe("API authentication failures", () => {
  beforeEach(() => {
    installStorage();
  });

  afterEach(() => {
    clearToken();
    vi.unstubAllGlobals();
    Reflect.deleteProperty(globalThis, "window");
  });

  it("clears and broadcasts the session token after a 401 response", async () => {
    setToken("expired-token");
    const observed: Array<string | null> = [];
    const unsubscribe = subscribeTokenChanges((token) => observed.push(token));
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Session expired" }), {
        status: 401,
        statusText: "Unauthorized",
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(getMe()).rejects.toMatchObject({
      name: "ApiError",
      status: 401,
      message: "Session expired",
    });
    unsubscribe();

    expect(getToken()).toBeNull();
    expect(observed).toEqual([null]);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/auth/me",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer expired-token" }),
      }),
    );
  });

  it("does not clear a newer session when an older request returns 401", async () => {
    setToken("old-token");
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
    setToken("new-token");
    resolveResponse(
      new Response(JSON.stringify({ detail: "Old session expired" }), {
        status: 401,
        statusText: "Unauthorized",
        headers: { "Content-Type": "application/json" },
      }),
    );

    await expect(oldRequest).rejects.toMatchObject({ status: 401 });
    expect(getToken()).toBe("new-token");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/auth/me",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer old-token" }),
      }),
    );
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

  it("stages existing-account invite authorization until redemption succeeds", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ access_token: "accepted-token" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await acceptInvite("invite-token", {}, "staged-login-token");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/invites/invite-token/accept",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          Authorization: "Bearer staged-login-token",
        }),
      }),
    );
    expect(getToken()).toBe("accepted-token");
  });

  it("downloads submission bytes with the bearer token", async () => {
    setToken("download-token");
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
        headers: expect.objectContaining({ Authorization: "Bearer download-token" }),
      }),
    );
  });

  it("downloads export bytes with the bearer token", async () => {
    setToken("export-token");
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
        headers: expect.objectContaining({ Authorization: "Bearer export-token" }),
      }),
    );
  });

  it("downloads an authorized package file without exposing a storage key", async () => {
    setToken("model-token");
    const fetchMock = vi.fn().mockResolvedValue(new Response("weights", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const blob = await downloadArtifactPackageFile(41, 73);

    expect(await blob.text()).toBe("weights");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/artifact-packages/41/files/73/download",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer model-token" }),
      }),
    );
  });

  it("requests the manager-approved archive lifecycle", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      package_id: 41,
      archived_at: "2026-01-01T00:00:00Z",
      purge_after: "2026-01-08T00:00:00Z",
    }), { status: 202, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);

    await archiveArtifactPackage(41);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/artifact-packages/41/archive",
      expect.objectContaining({ method: "POST" }),
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
