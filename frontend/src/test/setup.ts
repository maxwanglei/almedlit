if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: (query: string): MediaQueryList => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false,
    }),
  });
}

if (typeof HTMLElement !== "undefined") {
  for (const method of ["showPopover", "hidePopover"] as const) {
    if (typeof HTMLElement.prototype[method] !== "function") {
      Object.defineProperty(HTMLElement.prototype, method, {
        configurable: true,
        value() {},
      });
    }
  }
}
