import { createElement, lazy } from "react";
import type { ComponentType, ReactElement } from "react";

export function retryableLazy<Props extends object>(
  load: () => Promise<{ default: ComponentType<Props> }>,
): { Component: ComponentType<Props>; retry: () => void } {
  let rejected = false;
  const createLazyComponent = () => lazy(async () => {
    try {
      const loadedModule = await load();
      // A prevented Vite preload error can resolve an import with no module.
      if (!loadedModule?.default) throw new Error("The screen module could not be loaded.");
      return loadedModule;
    } catch (error) {
      rejected = true;
      throw error;
    }
  });
  let LazyComponent = createLazyComponent();

  function RetryableComponent(props: Props): ReactElement {
    return createElement(LazyComponent, props);
  }

  function retry(): void {
    if (!rejected) return;
    rejected = false;
    LazyComponent = createLazyComponent();
  }

  return { Component: RetryableComponent, retry };
}
