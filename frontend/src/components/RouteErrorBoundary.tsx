import { Component } from "react";
import type { ReactNode } from "react";

interface RouteErrorBoundaryProps {
  children: ReactNode;
  resetKey: string;
  onRetry?: () => void;
  reload?: () => void;
}

interface RouteErrorBoundaryState {
  failed: boolean;
  resetKey: string;
}

export default class RouteErrorBoundary extends Component<
  RouteErrorBoundaryProps,
  RouteErrorBoundaryState
> {
  state: RouteErrorBoundaryState = {
    failed: false,
    resetKey: this.props.resetKey,
  };

  static getDerivedStateFromError(): Partial<RouteErrorBoundaryState> {
    return { failed: true };
  }

  static getDerivedStateFromProps(
    props: RouteErrorBoundaryProps,
    state: RouteErrorBoundaryState,
  ): Partial<RouteErrorBoundaryState> | null {
    return props.resetKey !== state.resetKey
      ? { failed: false, resetKey: props.resetKey }
      : null;
  }

  private retry = (): void => {
    this.props.onRetry?.();
    this.setState({ failed: false });
  };

  private reload = (): void => {
    if (this.props.reload) this.props.reload();
    else window.location.reload();
  };

  render(): ReactNode {
    if (!this.state.failed) return this.props.children;

    return (
      <main id="main-content" className="route-error" tabIndex={-1}>
        <section role="alert" aria-labelledby="route-error-title">
          <h1 id="route-error-title">This screen could not be opened</h1>
          <p>
            Try opening the screen again. If it still fails, reload to get the latest version.
            {" "}Unsaved changes on this screen may need to be entered again.
          </p>
          <div className="recovery-actions">
            <button type="button" onClick={this.retry}>Retry</button>
            <button type="button" onClick={this.reload}>Reload</button>
          </div>
        </section>
      </main>
    );
  }
}
