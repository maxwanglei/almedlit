import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  authHeader,
  clearToken,
  getToken,
  setToken,
  subscribeTokenChanges,
  TOKEN_STORAGE_KEY,
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
    clearToken();
    Reflect.deleteProperty(globalThis, "window");
  });

  it("stores and returns the token", () => {
    setToken("abc");
    expect(getToken()).toBe("abc");
  });

  it("builds an Authorization header when a token is present", () => {
    setToken("abc");
    expect(authHeader()).toEqual({ Authorization: "Bearer abc" });
  });

  it("returns an empty header when no token is set", () => {
    expect(authHeader()).toEqual({});
  });

  it("notifies same-tab subscribers when the token changes", () => {
    const observed: Array<string | null> = [];
    const unsubscribe = subscribeTokenChanges((token) => observed.push(token));

    setToken("abc");
    setToken("abc");
    clearToken();
    unsubscribe();
    setToken("ignored-after-unsubscribe");

    expect(observed).toEqual(["abc", null]);
  });

  it("notifies subscribers about token changes from another tab", () => {
    const { dispatchStorage, localStorage } = installStorage();
    const observed: Array<string | null> = [];
    const unsubscribe = subscribeTokenChanges((token) => observed.push(token));

    localStorage.setItem(TOKEN_STORAGE_KEY, "other-user-token");
    dispatchStorage(TOKEN_STORAGE_KEY, "other-user-token");
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    dispatchStorage(TOKEN_STORAGE_KEY, null);
    dispatchStorage("unrelated", "value");
    unsubscribe();

    expect(observed).toEqual(["other-user-token", null]);
  });
});
