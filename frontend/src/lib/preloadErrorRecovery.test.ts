import { afterEach, describe, expect, it, vi } from "vitest";

import {
  clearPreloadError,
  getPreloadError,
  registerPreloadErrorHandler,
  subscribePreloadErrors,
} from "./preloadErrorRecovery";

function preloadError(payload?: unknown): Event {
  return Object.assign(new Event("vite:preloadError", { cancelable: true }), { payload });
}

describe("Vite preload error recovery", () => {
  const cleanups: Array<() => void> = [];

  afterEach(() => {
    cleanups.splice(0).forEach((cleanup) => cleanup());
    clearPreloadError();
  });

  it("cancels Vite's default error and reports the failure without reloading", () => {
    const reload = vi.fn();
    const target = Object.assign(new EventTarget(), { location: { reload } });
    const onError = vi.fn();
    cleanups.push(subscribePreloadErrors(onError), registerPreloadErrorHandler(target));
    const failure = new TypeError("Failed to fetch dynamically imported module");
    const event = preloadError(failure);

    expect(target.dispatchEvent(event)).toBe(false);
    expect(event.defaultPrevented).toBe(true);
    expect(onError).toHaveBeenCalledExactlyOnceWith();
    expect(getPreloadError()).toBe(failure);
    expect(getPreloadError()).toBe(getPreloadError());
    expect(reload).not.toHaveBeenCalled();
  });

  it("reports subsequent preload failures so recovery remains available", () => {
    const target = new EventTarget();
    const onError = vi.fn();
    cleanups.push(subscribePreloadErrors(onError), registerPreloadErrorHandler(target));
    const first = new Error("Unable to preload CSS");
    const second = new Error("Failed to fetch dynamically imported module");

    target.dispatchEvent(preloadError(first));
    expect(getPreloadError()).toBe(first);
    target.dispatchEvent(preloadError(second));

    expect(getPreloadError()).toBe(second);
    expect(onError).toHaveBeenCalledTimes(2);
  });

  it.each([undefined, null, "Missing chunk"])("normalizes an error payload of %s", (payload) => {
    const target = new EventTarget();
    const onError = vi.fn();
    cleanups.push(subscribePreloadErrors(onError), registerPreloadErrorHandler(target));
    const event = preloadError(payload);

    target.dispatchEvent(event);

    expect(event.defaultPrevented).toBe(true);
    expect(onError).toHaveBeenCalledExactlyOnceWith();
    expect(getPreloadError()).toBeInstanceOf(Error);
    expect(getPreloadError()?.message).toBe(payload || "A part of the application could not be loaded.");
  });

  it("dismisses the notification and allows a later failure to be reported", () => {
    const target = new EventTarget();
    const listener = vi.fn();
    cleanups.push(subscribePreloadErrors(listener), registerPreloadErrorHandler(target));
    const failure = new Error("Missing chunk");
    target.dispatchEvent(preloadError(failure));

    clearPreloadError();

    expect(getPreloadError()).toBeNull();
    expect(listener).toHaveBeenCalledTimes(2);
    clearPreloadError();
    expect(listener).toHaveBeenCalledTimes(2);
    target.dispatchEvent(preloadError(failure));
    expect(getPreloadError()).toBe(failure);
    expect(listener).toHaveBeenCalledTimes(3);
  });

  it("lets a subscriber unsubscribe independently of event handling", () => {
    const target = new EventTarget();
    const listener = vi.fn();
    const unsubscribe = subscribePreloadErrors(listener);
    cleanups.push(unsubscribe, registerPreloadErrorHandler(target));
    unsubscribe();
    const failure = new Error("Missing chunk");

    target.dispatchEvent(preloadError(failure));

    expect(listener).not.toHaveBeenCalled();
    expect(getPreloadError()).toBe(failure);
  });

  it("removes its listener without changing other listeners", () => {
    const target = new EventTarget();
    const onError = vi.fn();
    const otherListener = vi.fn();
    target.addEventListener("vite:preloadError", otherListener);
    cleanups.push(subscribePreloadErrors(onError));
    const cleanup = registerPreloadErrorHandler(target);
    cleanups.push(cleanup);
    cleanup();
    cleanup();
    const event = preloadError(new Error("Missing chunk"));

    expect(target.dispatchEvent(event)).toBe(true);
    expect(event.defaultPrevented).toBe(false);
    expect(onError).not.toHaveBeenCalled();
    expect(getPreloadError()).toBeNull();
    expect(otherListener).toHaveBeenCalledExactlyOnceWith(event);
  });
});
