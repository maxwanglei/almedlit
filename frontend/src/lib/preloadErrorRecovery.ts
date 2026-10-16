let preloadError: Error | null = null;
const listeners = new Set<() => void>();

function notifyListeners(): void {
  for (const listener of [...listeners]) listener();
}

export function getPreloadError(): Error | null {
  return preloadError;
}

export function subscribePreloadErrors(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function clearPreloadError(): void {
  if (preloadError === null) return;
  preloadError = null;
  notifyListeners();
}

export function registerPreloadErrorHandler(target: EventTarget = window): () => void {
  const handlePreloadError = (event: Event): void => {
    // Recovery must be user initiated: a reload could discard unsaved edits.
    event.preventDefault();
    const payload: unknown = (event as Event & { payload?: unknown }).payload;
    preloadError = payload instanceof Error
      ? payload
      : new Error(typeof payload === "string" && payload ? payload : "A part of the application could not be loaded.");
    notifyListeners();
  };

  target.addEventListener("vite:preloadError", handlePreloadError);
  return () => {
    target.removeEventListener("vite:preloadError", handlePreloadError);
  };
}
