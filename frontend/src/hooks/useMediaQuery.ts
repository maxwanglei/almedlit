import { useEffect, useState } from "react";

function getMediaQueryList(query: string): MediaQueryList | null {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return null;
  }
  return window.matchMedia(query);
}

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(
    () => getMediaQueryList(query)?.matches ?? false,
  );

  useEffect(() => {
    const media = getMediaQueryList(query);
    if (!media) {
      setMatches(false);
      return undefined;
    }
    const handleChange = (): void => setMatches(media.matches);
    handleChange();
    media.addEventListener("change", handleChange);
    return () => media.removeEventListener("change", handleChange);
  }, [query]);

  return matches;
}
