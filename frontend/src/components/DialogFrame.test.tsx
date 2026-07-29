// @vitest-environment jsdom

import { useState } from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import DialogFrame from "./DialogFrame";

function Harness({ busy = false }: { busy?: boolean }): React.ReactElement {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        Open
      </button>
      <DialogFrame
        open={open}
        ariaLabel="Shared dialog"
        busy={busy}
        error={error}
        backdropClassName="test-backdrop"
        dialogClassName="test-dialog"
        initialFocusSelector="[data-initial]"
        onDismiss={() => setOpen(false)}
      >
        <button type="button" data-initial>
          First
        </button>
        <button type="button" onClick={() => setError("Server rejected the change")}>
          Trigger error
        </button>
        {error ? <p role="alert">{error}</p> : null}
        <button type="button">Last</button>
      </DialogFrame>
    </>
  );
}

afterEach(() => {
  cleanup();
  document.body.style.overflow = "";
});

describe("DialogFrame", () => {
  it("locks scroll, traps focus, dismisses, and restores the trigger", async () => {
    render(<Harness />);
    const trigger = screen.getByRole("button", { name: "Open" });
    trigger.focus();
    fireEvent.click(trigger);

    const dialog = screen.getByRole("dialog", { name: "Shared dialog" });
    const first = screen.getByRole("button", { name: "First" });
    const last = screen.getByRole("button", { name: "Last" });
    await waitFor(() => expect(document.activeElement).toBe(first));
    expect(document.body.style.overflow).toBe("hidden");

    last.focus();
    fireEvent.keyDown(dialog, { key: "Tab" });
    expect(document.activeElement).toBe(first);
    fireEvent.keyDown(dialog, { key: "Escape" });

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.activeElement).toBe(trigger);
    expect(document.body.style.overflow).toBe("");
  });

  it("focuses a newly rendered server error", async () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    fireEvent.click(screen.getByRole("button", { name: "Trigger error" }));

    const error = await screen.findByRole("alert");
    await waitFor(() => expect(document.activeElement).toBe(error));
    expect(error.getAttribute("tabindex")).toBe("-1");
  });

  it("blocks Escape and backdrop dismissal while busy", () => {
    render(<Harness busy />);
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    const dialog = screen.getByRole("dialog", { name: "Shared dialog" });

    fireEvent.keyDown(dialog, { key: "Escape" });
    fireEvent.mouseDown(dialog.parentElement as HTMLElement);

    expect(screen.getByRole("dialog", { name: "Shared dialog" })).toBeTruthy();
    expect(dialog.getAttribute("aria-busy")).toBe("true");
  });
});
