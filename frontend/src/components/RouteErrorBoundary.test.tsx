// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { Profiler, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import RouteErrorBoundary from "./RouteErrorBoundary";

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("RouteErrorBoundary", () => {
  it("keeps the surrounding shell alive and retries a render failure without exposing it", () => {
    let shouldThrow = true;
    function Screen() {
      if (shouldThrow) throw new Error("Private exception details");
      return <p>Recovered screen</p>;
    }
    function Shell() {
      const [count, setCount] = useState(0);
      return (
        <>
          <button type="button" onClick={() => setCount(count + 1)}>
            Shell count {count}
          </button>
          <RouteErrorBoundary resetKey="/work" onRetry={() => { shouldThrow = false; }}>
            <Screen />
          </RouteErrorBoundary>
        </>
      );
    }
    render(<Shell />);

    const alert = screen.getByRole("alert");
    expect(screen.getByRole("heading", { name: "This screen could not be opened" })).toBeTruthy();
    expect(alert.textContent).toContain("Unsaved changes on this screen may need to be entered again.");
    expect(alert.textContent).not.toContain("Private exception details");
    const main = screen.getByRole("main");
    expect(main.id).toBe("main-content");
    expect(main.tabIndex).toBe(-1);

    fireEvent.click(screen.getByRole("button", { name: "Shell count 0" }));
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    expect(screen.getByText("Recovered screen")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Shell count 1" })).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("keeps repeated render failures recoverable", () => {
    const onRetry = vi.fn();
    function BrokenScreen(): React.ReactElement {
      throw new Error("Screen failed");
    }
    render(
      <RouteErrorBoundary resetKey="/broken" onRetry={onRetry}>
        <BrokenScreen />
      </RouteErrorBoundary>,
    );

    for (let attempt = 1; attempt <= 2; attempt += 1) {
      fireEvent.click(screen.getByRole("button", { name: "Retry" }));
      expect(onRetry).toHaveBeenCalledTimes(attempt);
      expect(screen.getByRole("alert")).toBeTruthy();
      expect(screen.getByRole("button", { name: "Reload" })).toBeTruthy();
    }
  });

  it("resets on a route key change before committing the old route's fallback", () => {
    function BrokenScreen(): React.ReactElement {
      throw new Error("Old route failed");
    }
    const committedAlerts: boolean[] = [];
    const onRender = () => committedAlerts.push(screen.queryByRole("alert") !== null);
    const { rerender } = render(
      <Profiler id="route" onRender={onRender}>
        <RouteErrorBoundary resetKey="/broken">
          <BrokenScreen />
        </RouteErrorBoundary>
      </Profiler>,
    );
    expect(screen.getByRole("alert")).toBeTruthy();
    committedAlerts.length = 0;

    rerender(
      <Profiler id="route" onRender={onRender}>
        <RouteErrorBoundary resetKey="/healthy">
          <p>Another screen</p>
        </RouteErrorBoundary>
      </Profiler>,
    );

    expect(screen.getByText("Another screen")).toBeTruthy();
    expect(committedAlerts).toEqual([false]);
  });

  it("reloads only when the user chooses Reload", () => {
    const reload = vi.fn();
    function BrokenScreen(): React.ReactElement {
      throw new Error("Screen failed");
    }
    render(
      <RouteErrorBoundary resetKey="/broken" reload={reload}>
        <BrokenScreen />
      </RouteErrorBoundary>,
    );
    expect(reload).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Reload" }));

    expect(reload).toHaveBeenCalledOnce();
  });
});
