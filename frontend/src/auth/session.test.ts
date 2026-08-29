import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  getSessionGeneration,
  LEGACY_TOKEN_STORAGE_KEY,
  publishSessionChange,
  removeLegacyToken,
  SESSION_EVENT_STORAGE_KEY,
  subscribeSessionChanges,
} from "./session";

interface InstalledStorage {
  dispatchStorage: (key: string | null, newValue: string | null) => void;
  localStorage: Storage;
}

function installStorage(): InstalledStorage {
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
  const storageListeners = new Set<(event: StorageEvent) => void>();

  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      localStorage,
      addEventListener: (type: string, listener: (event: StorageEvent) => void) => {
        if (type === "storage") {
          storageListeners.add(listener);
        }
      },
      removeEventListener: (type: string, listener: (event: StorageEvent) => void) => {
        if (type === "storage") {
          storageListeners.delete(listener);
        }
      },
    },
  });

  return {
    localStorage,
    dispatchStorage: (key, newValue) => {
      storageListeners.forEach((listener) =>
        listener({ key, newValue, storageArea: localStorage } as StorageEvent),
      );
    },
  };
}

describe("session token store", () => {
  beforeEach(() => installStorage());

  afterEach(() => {
    window.localStorage.clear();
    Reflect.deleteProperty(globalThis, "window");
  });

  it("removes a legacy localStorage bearer token", () => {
    window.localStorage.setItem(LEGACY_TOKEN_STORAGE_KEY, "legacy-jwt");
    removeLegacyToken();
    expect(window.localStorage.getItem(LEGACY_TOKEN_STORAGE_KEY)).toBeNull();
  });

  it("broadcasts only non-secret session state", () => {
    const observed: boolean[] = [];
    const before = getSessionGeneration();
    const unsubscribe = subscribeSessionChanges((authenticated) =>
      observed.push(authenticated),
    );

    publishSessionChange(true);
    publishSessionChange(false);
    unsubscribe();
    publishSessionChange(true);

    expect(observed).toEqual([true, false]);
    expect(getSessionGeneration()).toBe(before + 3);
    expect(window.localStorage.getItem(LEGACY_TOKEN_STORAGE_KEY)).toBeNull();
    expect(window.localStorage.getItem(SESSION_EVENT_STORAGE_KEY)).not.toContain("jwt");
  });

  it("notifies subscribers about session changes from another tab", () => {
    const { dispatchStorage, localStorage } = installStorage();
    const observed: boolean[] = [];
    const unsubscribe = subscribeSessionChanges((authenticated) =>
      observed.push(authenticated),
    );

    const event = JSON.stringify({ authenticated: true, nonce: "other-tab" });
    localStorage.setItem(SESSION_EVENT_STORAGE_KEY, event);
    dispatchStorage(SESSION_EVENT_STORAGE_KEY, event);
    dispatchStorage(null, null);
    dispatchStorage("unrelated", "value");
    unsubscribe();

    expect(observed).toEqual([true, false]);
  });
});
