import type {
  MouseEvent as ReactMouseEvent,
  ReactNode,
} from "react";
import { Button } from "@astryxdesign/core/Button";
import { EmptyState } from "@astryxdesign/core/EmptyState";

import { shouldHandleSpaClick } from "@/components/ModuleSwitcher";

export function statusVariant(
  status: string,
): "neutral" | "info" | "success" | "warning" | "error" {
  const normalized = status.toLowerCase();
  if (["active", "available", "completed", "succeeded", "ready"].includes(normalized)) {
    return "success";
  }
  if (["failed", "blocked", "rejected", "unavailable"].includes(normalized)) {
    return "error";
  }
  if (
    ["partial", "pilot", "queued", "running", "in_progress", "pending"].includes(
      normalized,
    )
  ) {
    return "warning";
  }
  if (["draft", "foundation", "planned", "experimental"].includes(normalized)) {
    return "info";
  }
  return "neutral";
}

export function formatStatus(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter: string) => letter.toUpperCase());
}

export function shortHash(value: string | null | undefined): string {
  return value ? value.slice(0, 10) : "Not recorded";
}

export function PlatformStatus({ value }: { value: string }): React.ReactElement {
  return (
    <span className="platform-status" data-variant={statusVariant(value)}>
      {formatStatus(value)}
    </span>
  );
}

export function PlatformRouteLink({
  href,
  className,
  children,
  ariaCurrent,
  onNavigate,
}: {
  href: string;
  className?: string;
  children: ReactNode;
  ariaCurrent?: "page";
  onNavigate: (path: string) => void;
}): React.ReactElement {
  return (
    <a
      href={href}
      className={className}
      aria-current={ariaCurrent}
      onClick={(event: ReactMouseEvent<HTMLAnchorElement>) => {
        if (!shouldHandleSpaClick(event)) return;
        event.preventDefault();
        onNavigate(href);
      }}
    >
      {children}
    </a>
  );
}

interface PageHeaderProps {
  title: string;
  description: string;
  actionLabel?: string;
  actionDisabled?: boolean;
  onAction?: () => void;
  secondary?: ReactNode;
}

export function PlatformPageHeader({
  title,
  description,
  actionLabel,
  actionDisabled,
  onAction,
  secondary,
}: PageHeaderProps): React.ReactElement {
  return (
    <header className="platform-page-header">
      <div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      <div className="platform-page-actions">
        {secondary}
        {actionLabel && onAction ? (
          <Button
            label={actionLabel}
            variant="primary"
            isDisabled={actionDisabled}
            onClick={onAction}
          />
        ) : null}
      </div>
    </header>
  );
}

interface EmptyProps {
  title: string;
  detail: string;
  actionLabel?: string;
  onAction?: () => void;
}

export function PlatformEmpty({
  title,
  detail,
  actionLabel,
  onAction,
}: EmptyProps): React.ReactElement {
  return (
    <div className="platform-empty">
      <EmptyState
        title={title}
        description={detail}
        actions={
          actionLabel && onAction ? (
            <Button label={actionLabel} variant="primary" onClick={onAction} />
          ) : undefined
        }
      />
    </div>
  );
}

interface Stat {
  label: string;
  value: string | number;
  detail?: string;
}

export function PlatformStats({ items }: { items: Stat[] }): React.ReactElement {
  return (
    <dl className="platform-stats">
      {items.map((item) => (
        <div key={item.label}>
          <dt>{item.label}</dt>
          <dd>{item.value}</dd>
          {item.detail ? <dd className="platform-stat-detail">{item.detail}</dd> : null}
        </div>
      ))}
    </dl>
  );
}

export function PlatformSection({
  title,
  description,
  action,
  children,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
}): React.ReactElement {
  return (
    <section className="platform-section">
      <div className="platform-section-header">
        <div>
          <h2>{title}</h2>
          {description ? <p>{description}</p> : null}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
