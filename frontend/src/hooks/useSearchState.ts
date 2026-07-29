import { useCallback, useMemo, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";

export type SearchStateUpdate = Record<string, string | number | null | undefined>;
export type SearchStateMode = "push" | "replace";
type SearchLocation = Pick<Location, "pathname" | "search" | "hash">;

export function buildSearchUrl(
  location: SearchLocation,
  update: SearchStateUpdate,
): string {
  const params = new URLSearchParams(location.search);

  Object.entries(update).forEach(([key, value]) => {
    if (value === null || value === undefined || value === "") {
      params.delete(key);
    } else {
      params.set(key, String(value));
    }
  });

  const search = params.toString();
  return `${location.pathname}${search ? `?${search}` : ""}${location.hash}`;
}

export function allowedSearchValue<T extends string>(
  params: URLSearchParams,
  key: string,
  allowed: readonly T[],
  fallback: T,
): { value: T; isValid: boolean } {
  const value = params.get(key);
  if (value === null) {
    return { value: fallback, isValid: true };
  }
  if (allowed.includes(value as T)) {
    return { value: value as T, isValid: true };
  }
  return { value: fallback, isValid: false };
}

export function useSearchState(): [
  URLSearchParams,
  (update: SearchStateUpdate, mode?: SearchStateMode) => void,
] {
  const location = useLocation();
  const navigate = useNavigate();
  const routerUrl = `${location.pathname}${location.search}${location.hash}`;
  const renderedPathnameRef = useRef(location.pathname);
  renderedPathnameRef.current = location.pathname;
  const observedRouterUrlRef = useRef(routerUrl);
  const currentLocationRef = useRef<SearchLocation>({
    pathname: location.pathname,
    search: location.search,
    hash: location.hash,
  });
  if (observedRouterUrlRef.current !== routerUrl) {
    observedRouterUrlRef.current = routerUrl;
    currentLocationRef.current = {
      pathname: location.pathname,
      search: location.search,
      hash: location.hash,
    };
  }
  const params = useMemo(
    () => new URLSearchParams(location.search),
    [location.search],
  );

  const updateSearch = useCallback(
    (update: SearchStateUpdate, mode: SearchStateMode = "push"): void => {
      // React may retain the previous route while a lazy destination suspends.
      // Ignore query updates from that retained screen so it cannot pull the
      // browser back after the user has navigated to another module.
      if (window.location.pathname !== renderedPathnameRef.current) {
        return;
      }
      const currentLocation = currentLocationRef.current;
      const nextUrl = buildSearchUrl(currentLocation, update);
      const currentUrl = `${currentLocation.pathname}${currentLocation.search}${currentLocation.hash}`;
      if (nextUrl === currentUrl) {
        return;
      }
      const parsed = new URL(nextUrl, "https://al-medlit.local");
      currentLocationRef.current = {
        pathname: parsed.pathname,
        search: parsed.search,
        hash: parsed.hash,
      };
      navigate(nextUrl, { replace: mode === "replace" });
    },
    [navigate],
  );

  return [params, updateSearch];
}
