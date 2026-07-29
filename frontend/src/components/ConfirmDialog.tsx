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
  requiredReason = false,
  reasonLabel = "Audit reason",
  onCancel,
  onConfirm,
}: ConfirmDialogProps): ReactElement | null {
  const titleId = useId();
  const descriptionId = useId();
  const cancelRef = useRef<HTMLButtonElement>(null);
  const [reason, setReason] = useState("");
  const hasReason = reason.trim().length > 0;

  useEffect(() => {
    if (!open) setReason("");
  }, [open]);

  return (
    <DialogFrame
      open={open}
      role="alertdialog"
      labelledBy={titleId}
      describedBy={descriptionId}
      busy={busy}
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
          <textarea
            name="auditReason"
            value={reason}
            rows={3}
            autoComplete="off"
            disabled={busy}
            aria-required="true"
            onChange={(event) => setReason(event.target.value)}
          />
        </label>
      ) : null}
      <div className="confirm-dialog-actions">
        <Button
          ref={cancelRef}
          label="Cancel"
          variant="ghost"
          isDisabled={busy}
          onClick={onCancel}
        />
        <Button
          label={busy ? "Working…" : confirmLabel}
          variant={destructive ? "destructive" : "primary"}
          isDisabled={busy || (requiredReason && !hasReason)}
          isLoading={busy}
          onClick={() => void onConfirm(reason.trim())}
        />
      </div>
    </DialogFrame>
  );
}
