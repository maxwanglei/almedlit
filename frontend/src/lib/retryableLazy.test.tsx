// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { Suspense } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import RouteErrorBoundary from "@/components/RouteErrorBoundary";

import { retryableLazy } from "./retryableLazy";

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("retryableLazy", () => {
  it("loads again after a rejected import when Retry is selected", async () => {
    function LoadedScreen({ title }: { title: string }) {
      return <p>{title}</p>;
    }
    const load = vi.fn()
      .mockRejectedValueOnce(new Error("Chunk fetch failed"))
      .mockResolvedValueOnce({ default: LoadedScreen });
    const { Component: Screen, retry } = retryableLazy<{ title: string }>(load);
    render(
      <RouteErrorBoundary resetKey="/screen" onRetry={retry}>
        <Suspense fallback={<p>Loading screen</p>}>
          <Screen title="Recovered module" />
        </Suspense>
      </RouteErrorBoundary>,
    );
    await screen.findByRole("alert");
    expect(load).toHaveBeenCalledOnce();

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText("Recovered module")).toBeTruthy();
    expect(load).toHaveBeenCalledTimes(2);
  });

  it("allows another Retry after a repeated import failure", async () => {
    const load = vi.fn()
      .mockRejectedValueOnce(new Error("First fetch failed"))
      .mockRejectedValueOnce(new Error("Second fetch failed"))
      .mockResolvedValueOnce({ default: () => <p>Third attempt succeeded</p> });
    const { Component: Screen, retry } = retryableLazy(load);
    render(
      <RouteErrorBoundary resetKey="/screen" onRetry={retry}>
        <Suspense fallback={<p>Loading screen</p>}><Screen /></Suspense>
      </RouteErrorBoundary>,
    );
    await screen.findByRole("alert");

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(load).toHaveBeenCalledTimes(2));
    await screen.findByRole("alert");
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText("Third attempt succeeded")).toBeTruthy();
    expect(load).toHaveBeenCalledTimes(3);
  });

  it.each([
    { description: "undefined", loadedModule: undefined },
    { description: "missing default", loadedModule: {} },
  ])("can retry an import that resolves to $description", async ({ loadedModule }) => {
    const load = vi.fn()
      .mockResolvedValueOnce(loadedModule)
      .mockResolvedValueOnce({ default: () => <p>Valid module</p> });
    const { Component: Screen, retry } = retryableLazy(load);
    render(
      <RouteErrorBoundary resetKey="/screen" onRetry={retry}>
        <Suspense fallback={<p>Loading screen</p>}><Screen /></Suspense>
      </RouteErrorBoundary>,
    );
    await screen.findByRole("alert");

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText("Valid module")).toBeTruthy();
    expect(load).toHaveBeenCalledTimes(2);
  });

  it("remounts a resolved module after its render fails without reloading it", async () => {
    let shouldThrow = true;
    function LoadedScreen() {
      if (shouldThrow) throw new Error("Rendering failed");
      return <p>Recovered render</p>;
    }
    const load = vi.fn().mockResolvedValue({ default: LoadedScreen });
    const { Component: Screen, retry } = retryableLazy(load);
    render(
      <RouteErrorBoundary resetKey="/screen" onRetry={() => {
        shouldThrow = false;
        retry();
      }}>
        <Suspense fallback={<p>Loading screen</p>}><Screen /></Suspense>
      </RouteErrorBoundary>,
    );
    await screen.findByRole("alert");

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText("Recovered render")).toBeTruthy();
    expect(load).toHaveBeenCalledOnce();
  });
});
