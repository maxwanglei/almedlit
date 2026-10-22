import { NativeTextarea } from "@/components/NativeField";
import {
  useEffect,
  useId,
  useRef,
  useState,
  type ReactElement,
} from "react";
import { Button } from "@astryxdesign/core/Button";

import DialogFrame from "./DialogFrame";

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  description: string;
  confirmLabel: string;
  destructive?: boolean;
  busy?: boolean;
  error?: string | null;
  requiredReason?: boolean;
  reasonLabel?: string;
  onCancel: () => void;
  onConfirm: (reason: string) => void | Promise<void>;
}

export default function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  destructive = true,
  busy = false,
  error = null,
  requiredReason = false,
  reasonLabel = "Audit reason",
  onCancel,
  onConfirm,
}: ConfirmDialogProps): ReactElement | null {
  const titleId = useId();
  const descriptionId = useId();
  const cancelRef = useRef<HTMLButtonElement>(null);
  const [reason, setReason] = useState("");
  const [confirmationError, setConfirmationError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const working = busy || confirming;
  const visibleError = confirmationError ?? error;
  const hasReason = reason.trim().length > 0;

  useEffect(() => {
    if (!open) {
      setReason("");
      setConfirmationError(null);
    }
  }, [open]);

  return (
    <DialogFrame
      open={open}
      role="alertdialog"
      labelledBy={titleId}
      describedBy={descriptionId}
      busy={working}
      error={visibleError}
      initialFocusRef={cancelRef}
      backdropClassName="confirm-dialog-backdrop"
      dialogClassName="confirm-dialog"
      onDismiss={onCancel}
    >
      <div className="confirm-dialog-copy">
        <h2 id={titleId}>{title}</h2>
        <p id={descriptionId}>{description}</p>
      </div>
      {requiredReason ? (
        <label className="confirm-dialog-reason">
          <span>{reasonLabel}</span>
          <NativeTextarea
            name="auditReason"
            value={reason}
            rows={3}
            autoComplete="off"
            disabled={working}
            aria-required="true"
            onChange={(event) => setReason(event.target.value)}
          />
        </label>
      ) : null}
      {visibleError ? <p className="platform-error" role="alert">{visibleError}</p> : null}
      <div className="confirm-dialog-actions">
        <Button
          ref={cancelRef}
          label="Cancel"
          variant="ghost"
          isDisabled={working}
          onClick={onCancel}
        />
        <Button
          label={working ? "Working…" : confirmLabel}
          variant={destructive ? "destructive" : "primary"}
          isDisabled={working || (requiredReason && !hasReason)}
          isLoading={working}
          onClick={async () => {
            setConfirmationError(null);
            setConfirming(true);
            try {
              await onConfirm(reason.trim());
            } catch (caught) {
              setConfirmationError(caught instanceof Error ? caught.message : "Unable to complete this action. Try again.");
            } finally {
              setConfirming(false);
            }
          }}
        />
      </div>
    </DialogFrame>
  );
}
