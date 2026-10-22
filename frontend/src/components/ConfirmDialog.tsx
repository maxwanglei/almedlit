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
  onCancel: () => void;
  onConfirm: () => void | Promise<void>;
}

export default function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  destructive = true,
  busy = false,
  error = null,
  onCancel,
  onConfirm,
}: ConfirmDialogProps): ReactElement | null {
  const titleId = useId();
  const descriptionId = useId();
  const cancelRef = useRef<HTMLButtonElement>(null);
  const [confirmationError, setConfirmationError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const working = busy || confirming;
  const visibleError = confirmationError ?? error;

  useEffect(() => {
    if (!open) {
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
          isDisabled={working}
          isLoading={working}
          onClick={async () => {
            setConfirmationError(null);
            setConfirming(true);
            try {
              await onConfirm();
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
