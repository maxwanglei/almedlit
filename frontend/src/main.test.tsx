// @vitest-environment jsdom

import { afterEach, expect, it, vi } from "vitest";

import * as preloadRecovery from "./lib/preloadErrorRecovery";

const mocks = vi.hoisted(() => ({
  createRoot: vi.fn(),
  render: vi.fn(),
}));

vi.mock("react-dom/client", () => ({
  default: { createRoot: mocks.createRoot },
}));
vi.mock("./App", () => ({ default: () => null }));

afterEach(() => {
  preloadRecovery.clearPreloadError();
  document.body.replaceChildren();
  vi.restoreAllMocks();
});

it("installs preload recovery before rendering the application", async () => {
  const root = document.createElement("div");
  root.id = "root";
  document.body.append(root);
  const register = vi.spyOn(preloadRecovery, "registerPreloadErrorHandler");
  const error = new Error("A screen chunk is no longer available");
  const failure = Object.assign(new Event("vite:preloadError", { cancelable: true }), { payload: error });
  mocks.createRoot.mockReturnValue({ render: mocks.render });
  mocks.render.mockImplementation(() => window.dispatchEvent(failure));

  try {
    await import("./main");

    expect(mocks.createRoot).toHaveBeenCalledWith(root);
    expect(mocks.render).toHaveBeenCalledOnce();
    expect(failure.defaultPrevented).toBe(true);
    expect(preloadRecovery.getPreloadError()).toBe(error);
  } finally {
    register.mock.results[0]?.value?.();
  }
});
