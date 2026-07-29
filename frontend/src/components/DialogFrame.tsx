import {
  useEffect,
  useRef,
  type ReactElement,
  type ReactNode,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled]):not([type='hidden'])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[contenteditable='true']",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

let scrollLockCount = 0;
let bodyOverflowBeforeLock = "";

function lockBodyScroll(): void {
  if (scrollLockCount === 0) {
    bodyOverflowBeforeLock = document.body.style.overflow;
    document.body.style.overflow = "hidden";
  }
  scrollLockCount += 1;
}

function unlockBodyScroll(): void {
  scrollLockCount = Math.max(0, scrollLockCount - 1);
  if (scrollLockCount === 0) {
    document.body.style.overflow = bodyOverflowBeforeLock;
  }
}

function focusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(
    container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
  ).filter((element) => {
    if (
      element.hidden ||
      element.getAttribute("aria-hidden") === "true" ||
      element.closest("[hidden], [aria-hidden='true']")
    ) {
      return false;
    }
    const style = window.getComputedStyle(element);
    return style.display !== "none" && style.visibility !== "hidden";
  });
}

export interface DialogFrameProps {
  open?: boolean;
  role?: "dialog" | "alertdialog";
  ariaLabel?: string;
  labelledBy?: string;
  describedBy?: string;
  busy?: boolean;
  error?: string | null;
  errorFocusSelector?: string;
  initialFocusRef?: RefObject<HTMLElement | null>;
  initialFocusSelector?: string;
  backdropClassName: string;
  dialogClassName: string;
  dialogElement?: "div" | "section";
  portal?: boolean;
  onDismiss: () => void;
  children: ReactNode;
}

export default function DialogFrame({
  open = true,
  role = "dialog",
  ariaLabel,
  labelledBy,
  describedBy,
  busy = false,
  error = null,
  errorFocusSelector = "[role='alert']",
  initialFocusRef,
  initialFocusSelector,
  backdropClassName,
  dialogClassName,
  dialogElement = "div",
  portal = true,
  onDismiss,
  children,
}: DialogFrameProps): ReactElement | null {
  const dialogRef = useRef<HTMLElement | null>(null);
  const lastErrorRef = useRef(error);

  useEffect(() => {
    if (!open) return;

    const trigger =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    lockBodyScroll();
    const frame = window.requestAnimationFrame(() => {
      const dialog = dialogRef.current;
      const initialTarget =
        initialFocusRef?.current ??
        (initialFocusSelector
          ? dialog?.querySelector<HTMLElement>(initialFocusSelector) ?? null
          : null) ??
        (dialog ? focusableElements(dialog)[0] ?? null : null) ??
        dialog;
      initialTarget?.focus();
    });

    return () => {
      window.cancelAnimationFrame(frame);
      unlockBodyScroll();
      if (trigger?.isConnected) trigger.focus();
    };
  }, [initialFocusRef, initialFocusSelector, open]);

  useEffect(() => {
    const previousError = lastErrorRef.current;
    lastErrorRef.current = error;
    if (!open || !error || error === previousError) return;

    const frame = window.requestAnimationFrame(() => {
      const target =
        dialogRef.current?.querySelector<HTMLElement>(errorFocusSelector) ??
        document.querySelector<HTMLElement>(errorFocusSelector);
      if (!target) return;
      if (!target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
      target.focus();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [error, errorFocusSelector, open]);

  if (!open) return null;

  const DialogElement = dialogElement;
  const content = (
    <div
      className={backdropClassName}
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busy) onDismiss();
      }}
    >
      <DialogElement
        ref={(element) => {
          dialogRef.current = element;
        }}
        className={dialogClassName}
        role={role}
        aria-modal="true"
        aria-label={ariaLabel}
        aria-labelledby={labelledBy}
        aria-describedby={describedBy}
        aria-busy={busy || undefined}
        tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            if (!busy) onDismiss();
            return;
          }
          if (event.key !== "Tab") return;

          const dialog = dialogRef.current;
          if (!dialog) return;
          const focusable = focusableElements(dialog);
          if (!focusable.length) {
            event.preventDefault();
            dialog.focus();
            return;
          }

          const first = focusable[0];
          const last = focusable[focusable.length - 1];
          const active = document.activeElement;
          if (event.shiftKey && (active === first || !dialog.contains(active))) {
            event.preventDefault();
            last.focus();
          } else if (
            !event.shiftKey &&
            (active === last || !dialog.contains(active))
          ) {
            event.preventDefault();
            first.focus();
          }
        }}
      >
        {children}
      </DialogElement>
    </div>
  );

  return portal ? createPortal(content, document.body) : content;
}
