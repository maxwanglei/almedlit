import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  getSessionGeneration,
  LEGACY_TOKEN_STORAGE_KEY,
  publishSessionChange,
  removeLegacyToken,
  SESSION_EVENT_STORAGE_KEY,
  subscribeSessionChanges,
  publishMembershipChange,
  subscribeMembershipChanges,
  MEMBERSHIP_EVENT_STORAGE_KEY,
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

  it("refreshes memberships without invalidating account sessions in any tab", () => {
    const { dispatchStorage, localStorage } = installStorage();
    const sessionChanged = vi.fn();
    const membershipChanged = vi.fn();
    const unsubscribeSession = subscribeSessionChanges(sessionChanged);
    const unsubscribeMemberships = subscribeMembershipChanges(membershipChanged);
    const generation = getSessionGeneration();
    publishMembershipChange();
    const event = localStorage.getItem(MEMBERSHIP_EVENT_STORAGE_KEY);
    dispatchStorage(MEMBERSHIP_EVENT_STORAGE_KEY, event);
    expect(membershipChanged).toHaveBeenCalledTimes(2);
    expect(sessionChanged).not.toHaveBeenCalled();
    expect(getSessionGeneration()).toBe(generation);
    unsubscribeSession(); unsubscribeMemberships();
  });

  it("clears app drafts on logout and cross-tab logout while retaining unrelated storage", () => {
    const { dispatchStorage } = installStorage();
    const drafts = new Map<string, string>();
    Object.defineProperty(window, "sessionStorage", { configurable: true, value: {
      get length() { return drafts.size; },
      key: (index: number) => [...drafts.keys()][index] ?? null,
      removeItem: (key: string) => drafts.delete(key),
    } });
    const seed = () => {
      drafts.set("prediction-draft:7", "private");
      drafts.set("prediction-review:7:8", "private");
      drafts.set("al-medlit:source-setup:7:new", "private");
      drafts.set("al-medlit:training-preparation:7:new", "private");
      drafts.set("other-app", "retained");
    };
    seed();
    publishSessionChange(false);
    expect([...drafts.keys()]).toEqual(["other-app"]);
    seed();
    const unsubscribe = subscribeSessionChanges(() => undefined);
    dispatchStorage(SESSION_EVENT_STORAGE_KEY, JSON.stringify({ authenticated: false }));
    expect([...drafts.keys()]).toEqual(["other-app"]);
    unsubscribe();
  });

  it("notifies this tab even when browser storage is unavailable", () => {
    const observed = vi.fn();
    const unsubscribe = subscribeSessionChanges(observed);
    Object.defineProperty(window, "localStorage", {
      configurable: true, get: () => { throw new Error("Storage denied"); },
    });
    expect(() => publishSessionChange(false)).not.toThrow();
    expect(observed).toHaveBeenCalledWith(false);
    unsubscribe();
    installStorage();
  });
});
