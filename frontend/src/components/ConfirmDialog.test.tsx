// @vitest-environment jsdom

import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import ConfirmDialog from "./ConfirmDialog";

function DialogHarness({
  requiredReason = false,
  busy = false,
  onConfirm = vi.fn(),
}: {
  requiredReason?: boolean;
  busy?: boolean;
  onConfirm?: (reason: string) => void;
}): React.ReactElement {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        Delete item
      </button>
      <ConfirmDialog
        open={open}
        title="Delete item?"
        description="This permanently removes the selected item."
        confirmLabel="Delete item"
        busy={busy}
        requiredReason={requiredReason}
        onCancel={() => setOpen(false)}
        onConfirm={onConfirm}
      />
    </>
  );
}

afterEach(cleanup);

describe("ConfirmDialog", () => {
  it("traps focus, closes with Escape, and restores the trigger", async () => {
    render(<DialogHarness />);
    const trigger = screen.getByRole("button", { name: "Delete item" });
    trigger.focus();
    fireEvent.click(trigger);

    const dialog = screen.getByRole("alertdialog", { name: "Delete item?" });
    const cancel = screen.getByRole("button", { name: "Cancel" });
    const confirm = screen.getAllByRole("button", { name: "Delete item" })[1];
    await waitFor(() => expect(document.activeElement).toBe(cancel));

    confirm.focus();
    fireEvent.keyDown(dialog, { key: "Tab" });
    expect(document.activeElement).toBe(cancel);
    fireEvent.keyDown(dialog, { key: "Escape" });

    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  it("requires a trimmed audit reason before confirmation", () => {
    const onConfirm = vi.fn();
    render(<DialogHarness requiredReason onConfirm={onConfirm} />);
    fireEvent.click(screen.getByRole("button", { name: "Delete item" }));

    const confirm = screen.getAllByRole("button", {
      name: "Delete item",
    })[1] as HTMLButtonElement;
    expect(confirm.disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Audit reason"), {
      target: { value: "  Duplicate import  " },
    });
    expect(confirm.disabled).toBe(false);
    fireEvent.click(confirm);
    expect(onConfirm).toHaveBeenCalledWith("Duplicate import");
  });

  it("blocks cancellation and confirmation while busy", () => {
    const onConfirm = vi.fn();
    render(<DialogHarness busy onConfirm={onConfirm} />);
    fireEvent.click(screen.getByRole("button", { name: "Delete item" }));

    const dialog = screen.getByRole("alertdialog", { name: "Delete item?" });
    expect(dialog.getAttribute("aria-busy")).toBe("true");
    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(screen.getByRole("alertdialog", { name: "Delete item?" })).toBeTruthy();
    expect(
      (screen.getByRole("button", { name: "Cancel" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(
      (screen.getByRole("button", { name: "Working…" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(onConfirm).not.toHaveBeenCalled();
  });
});
