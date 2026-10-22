export const LEGACY_TOKEN_STORAGE_KEY = "al_medlit_access_token";
export const SESSION_EVENT_STORAGE_KEY = "al_medlit_session_event";
export const MEMBERSHIP_EVENT_STORAGE_KEY = "al_medlit_membership_event";

export type SessionChangeListener = (authenticated: boolean) => void;

const sessionChangeListeners = new Set<SessionChangeListener>();
const membershipChangeListeners = new Set<() => void>();
let sessionGeneration = 0;

function storage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch { return null; }
}

export function removeLegacyToken(): void {
  try { storage()?.removeItem(LEGACY_TOKEN_STORAGE_KEY); } catch { return; }
}

function clearSessionDrafts(): void {
  if (typeof window === "undefined") return;
  try {
    const drafts = window.sessionStorage;
    if (!drafts) return;
    for (let index = drafts.length - 1; index >= 0; index -= 1) {
      const key = drafts.key(index);
      if (key?.startsWith("al-medlit:") || key?.startsWith("prediction-draft:") || key?.startsWith("prediction-review:")) {
        drafts.removeItem(key);
      }
    }
  } catch { return; }
}

export function getSessionGeneration(): number {
  return sessionGeneration;
}

export function publishSessionChange(authenticated: boolean): void {
  sessionGeneration += 1;
  clearSessionDrafts();
  const tokenStorage = storage();
  try {
    if (tokenStorage) {
    tokenStorage.removeItem(LEGACY_TOKEN_STORAGE_KEY);
    tokenStorage.setItem(
      SESSION_EVENT_STORAGE_KEY,
      JSON.stringify({ authenticated, nonce: `${Date.now()}-${Math.random()}` }),
    );
    }
  } catch { /* Browser storage can be blocked while the cookie session remains valid. */ }
  sessionChangeListeners.forEach((listener) => listener(authenticated));
}

export function publishMembershipChange(): void {
  try {
    storage()?.setItem(MEMBERSHIP_EVENT_STORAGE_KEY, JSON.stringify({
      kind: "memberships", nonce: `${Date.now()}-${Math.random()}`,
    }));
  } catch { /* Browser storage can be blocked while the cookie session remains valid. */ }
  membershipChangeListeners.forEach((listener) => listener());
}

export function subscribeMembershipChanges(listener: () => void): () => void {
  membershipChangeListeners.add(listener);
  const handleStorage = (event: StorageEvent) => {
    if (event.key !== MEMBERSHIP_EVENT_STORAGE_KEY || event.newValue === null) return;
    if (event.storageArea !== null && event.storageArea !== storage()) return;
    try {
      const value: unknown = JSON.parse(event.newValue);
      if (typeof value === "object" && value !== null && "kind" in value && value.kind === "memberships") listener();
    } catch { return; }
  };
  if (typeof window !== "undefined") window.addEventListener("storage", handleStorage);
  return () => {
    membershipChangeListeners.delete(listener);
    if (typeof window !== "undefined") window.removeEventListener("storage", handleStorage);
  };
}

export function subscribeSessionChanges(
  listener: SessionChangeListener,
): () => void {
  sessionChangeListeners.add(listener);

  const handleStorage = (event: StorageEvent): void => {
    if (event.key === null) {
      sessionGeneration += 1;
      clearSessionDrafts();
      listener(false);
      return;
    }
    if (event.key !== SESSION_EVENT_STORAGE_KEY || event.newValue === null) {
      return;
    }
    if (event.storageArea !== null && event.storageArea !== storage()) {
      return;
    }
    try {
      const parsed: unknown = JSON.parse(event.newValue);
      if (typeof parsed === "object" && parsed !== null && "authenticated" in parsed && typeof parsed.authenticated === "boolean") {
        sessionGeneration += 1;
        clearSessionDrafts();
        listener(parsed.authenticated);
      }
    } catch {
      // Ignore malformed cross-tab notifications. Server-side cookie
      // validation remains the authority for authentication state.
    }
  };

  if (typeof window !== "undefined") {
    window.addEventListener("storage", handleStorage);
  }

  return () => {
    sessionChangeListeners.delete(listener);
    if (typeof window !== "undefined") {
      window.removeEventListener("storage", handleStorage);
    }
  };
}
