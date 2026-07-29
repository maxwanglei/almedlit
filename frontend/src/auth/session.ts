export const TOKEN_STORAGE_KEY = "al_medlit_access_token";

export type TokenChangeListener = (token: string | null) => void;

const tokenChangeListeners = new Set<TokenChangeListener>();

function storage(): Storage | null {
  return typeof window === "undefined" ? null : window.localStorage;
}

export function getToken(): string | null {
  return storage()?.getItem(TOKEN_STORAGE_KEY) ?? null;
}

function notifyTokenChange(token: string | null): void {
  tokenChangeListeners.forEach((listener) => listener(token));
}

export function setToken(token: string): void {
  const tokenStorage = storage();
  if (!tokenStorage || tokenStorage.getItem(TOKEN_STORAGE_KEY) === token) {
    return;
  }
  tokenStorage.setItem(TOKEN_STORAGE_KEY, token);
  notifyTokenChange(token);
}

export function clearToken(): void {
  const tokenStorage = storage();
  if (!tokenStorage || tokenStorage.getItem(TOKEN_STORAGE_KEY) === null) {
    return;
  }
  tokenStorage.removeItem(TOKEN_STORAGE_KEY);
  notifyTokenChange(null);
}

export function subscribeTokenChanges(listener: TokenChangeListener): () => void {
  const sameTabListener: TokenChangeListener = (token) => listener(token);
  tokenChangeListeners.add(sameTabListener);

  const handleStorage = (event: StorageEvent): void => {
    if (event.key !== TOKEN_STORAGE_KEY && event.key !== null) {
      return;
    }
    if (event.storageArea !== null && event.storageArea !== storage()) {
      return;
    }
    listener(event.key === TOKEN_STORAGE_KEY ? event.newValue : getToken());
  };

  if (typeof window !== "undefined") {
    window.addEventListener("storage", handleStorage);
  }

  return () => {
    tokenChangeListeners.delete(sameTabListener);
    if (typeof window !== "undefined") {
      window.removeEventListener("storage", handleStorage);
    }
  };
}

export function authHeader(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}
