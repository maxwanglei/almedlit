export const LEGACY_TOKEN_STORAGE_KEY = "al_medlit_access_token";
export const SESSION_EVENT_STORAGE_KEY = "al_medlit_session_event";

export type SessionChangeListener = (authenticated: boolean) => void;

const sessionChangeListeners = new Set<SessionChangeListener>();
let sessionGeneration = 0;

function storage(): Storage | null {
  return typeof window === "undefined" ? null : window.localStorage;
}

export function removeLegacyToken(): void {
  storage()?.removeItem(LEGACY_TOKEN_STORAGE_KEY);
}

export function getSessionGeneration(): number {
  return sessionGeneration;
}

export function publishSessionChange(authenticated: boolean): void {
  sessionGeneration += 1;
  const tokenStorage = storage();
  if (tokenStorage) {
    tokenStorage.removeItem(LEGACY_TOKEN_STORAGE_KEY);
    tokenStorage.setItem(
      SESSION_EVENT_STORAGE_KEY,
      JSON.stringify({ authenticated, nonce: `${Date.now()}-${Math.random()}` }),
    );
  }
  sessionChangeListeners.forEach((listener) => listener(authenticated));
}

export function subscribeSessionChanges(
  listener: SessionChangeListener,
): () => void {
  sessionChangeListeners.add(listener);

  const handleStorage = (event: StorageEvent): void => {
    if (event.key === null) {
      sessionGeneration += 1;
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
      const parsed = JSON.parse(event.newValue) as { authenticated?: unknown };
      if (typeof parsed.authenticated === "boolean") {
        sessionGeneration += 1;
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
