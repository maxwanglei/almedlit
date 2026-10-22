import type {
  MouseEvent as ReactMouseEvent,
  ReactNode,
} from "react";
import { Button } from "@astryxdesign/core/Button";
import { EmptyState } from "@astryxdesign/core/EmptyState";

import { shouldHandleSpaClick } from "@/components/ModuleSwitcher";

type StatusTone = "neutral" | "info" | "success" | "warning" | "error";

const STATUS_PRESENTATION: Record<string, { label: string; tone: StatusTone }> = {
  active: { label: "Active", tone: "success" },
  available: { label: "Available", tone: "success" },
  completed: { label: "Completed", tone: "success" },
  succeeded: { label: "Succeeded", tone: "success" },
  ready: { label: "Ready", tone: "success" },
  submitted: { label: "Submitted", tone: "success" },
  closed: { label: "Closed", tone: "neutral" },
  cancelled: { label: "Cancelled", tone: "neutral" },
  canceled: { label: "Cancelled", tone: "neutral" },
  failed: { label: "Failed", tone: "error" },
  blocked: { label: "Blocked", tone: "error" },
  rejected: { label: "Rejected", tone: "error" },
  unavailable: { label: "Unavailable", tone: "error" },
  partial: { label: "Partial", tone: "warning" },
  pilot: { label: "Pilot", tone: "warning" },
  queued: { label: "Queued", tone: "info" },
  running: { label: "Running", tone: "info" },
  in_progress: { label: "In progress", tone: "info" },
  pending: { label: "Pending", tone: "warning" },
  review: { label: "Review", tone: "warning" },
  adjudication_ready: { label: "Ready for adjudication", tone: "warning" },
  open: { label: "Open", tone: "info" },
  draft: { label: "Draft", tone: "neutral" },
  foundation: { label: "Foundation", tone: "info" },
  planned: { label: "Planned", tone: "neutral" },
  experimental: { label: "Experimental", tone: "warning" },
};

const TERM_LABELS: Record<string, string> = {
  conventional_ml: "Conventional ML",
  llm_finetune: "LLM fine-tuning",
  lora: "LoRA",
  minio: "MinIO",
  f1_macro: "F1 macro",
  f1_micro: "F1 micro",
};

const ACRONYMS: Record<string, string> = {
  api: "API", cpu: "CPU", csv: "CSV", f1: "F1", gpu: "GPU", id: "ID",
  json: "JSON", jsonl: "JSONL", llm: "LLM", ml: "ML", ner: "NER", nlp: "NLP",
  pmid: "PMID", pmc: "PMC", sha256: "SHA-256", tfidf: "TF-IDF", uri: "URI",
};

export function statusVariant(status: string): StatusTone {
  return STATUS_PRESENTATION[status.trim().toLowerCase()]?.tone ?? "neutral";
}

export function formatStatus(value: string): string {
  const normalized = value.trim().toLowerCase();
  const knownLabel = STATUS_PRESENTATION[normalized]?.label ?? TERM_LABELS[normalized];
  if (knownLabel) return knownLabel;
  const words = normalized.replace(/_/g, " ").split(/\s+/);
  return words.map((word, index) => ACRONYMS[word] ?? (index === 0
    ? word.charAt(0).toUpperCase() + word.slice(1)
    : word)).join(" ");
}

export function formatCount(count: number, singular: string, plural = `${singular}s`): string {
  return `${count.toLocaleString()} ${count === 1 ? singular : plural}`;
}

export function formatPercent(value: number): string {
  return new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 }).format(value);
}

export function formatDateTime(value: string | null | undefined, fallback = "Not recorded"): string {
  if (!value) return fallback;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? fallback : date.toLocaleString(undefined, {
    dateStyle: "medium", timeStyle: "short",
  });
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
