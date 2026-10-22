// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ValidatedForm from "./ValidatedForm";
import { NativeInput, NativeSelect } from "./NativeField";

afterEach(cleanup);

describe("field validation", () => {
  it("associates errors with each invalid control and focuses the first", () => {
    const submit = vi.fn((event) => event.preventDefault());
    render(<ValidatedForm onSubmit={submit}>
      <fieldset><label>Name<NativeInput required /></label><label>Email<NativeInput type="email" required /></label></fieldset>
      <button type="submit">Save</button>
    </ValidatedForm>);
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    const name = screen.getByRole("textbox", { name: "Name" });
    const email = screen.getByRole("textbox", { name: "Email" });
    expect(submit).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(name);
    for (const field of [name, email]) {
      expect(field.getAttribute("aria-invalid")).toBe("true");
      expect(document.getElementById(field.getAttribute("aria-describedby") ?? "")?.textContent).toBeTruthy();
    }
    fireEvent.input(name, { target: { value: "Trial" } });
    expect(name.hasAttribute("aria-invalid")).toBe(false);
    expect(email.getAttribute("aria-invalid")).toBe("true");
  });

  it("keeps existing help and custom errors while validating native constraints", () => {
    render(<ValidatedForm>
      <label>Sample size<NativeInput type="number" min={1} max={10} aria-describedby="sample-help" defaultValue={20} /></label>
      <span id="sample-help">Choose up to ten records.</span>
      <label>Server field<NativeInput aria-invalid="true" aria-describedby="server-error" /></label>
      <span id="server-error">This name is taken.</span>
    </ValidatedForm>);
    const sample = screen.getByLabelText("Sample size");
    fireEvent.blur(sample);
    expect(sample.getAttribute("aria-invalid")).toBe("true");
    expect(sample.getAttribute("aria-describedby")).toContain("sample-help");
    expect(screen.getByLabelText("Server field").getAttribute("aria-invalid")).toBe("true");
  });

  it("clears a previous error when a conditional field becomes optional or disabled", () => {
    const { rerender } = render(<NativeInput required aria-label="Labels" />);
    const input = screen.getByRole("textbox", { name: "Labels" });
    fireEvent.blur(input);
    expect(input.getAttribute("aria-invalid")).toBe("true");
    rerender(<NativeInput aria-label="Labels" />);
    expect(input.hasAttribute("aria-invalid")).toBe(false);
    rerender(<NativeInput required aria-label="Labels" />);
    fireEvent.blur(input);
    rerender(<NativeInput required disabled aria-label="Labels" />);
    expect(input.hasAttribute("aria-invalid")).toBe(false);
  });

  it("updates select and checkbox errors through native interaction", () => {
    render(<ValidatedForm>
      <label>Dataset<NativeSelect required defaultValue=""><option value="">Choose a dataset</option><option value="one">Study</option></NativeSelect></label>
      <label>Approve<NativeInput type="checkbox" required /></label>
      <button type="submit">Save</button>
    </ValidatedForm>);
    const select = screen.getByRole("combobox", { name: "Dataset" });
    const checkbox = screen.getByRole("checkbox", { name: "Approve" });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(select.getAttribute("aria-invalid")).toBe("true");
    expect(checkbox.getAttribute("aria-invalid")).toBe("true");
    fireEvent.input(select, { target: { value: "one" } });
    fireEvent.click(checkbox);
    expect(select.hasAttribute("aria-invalid")).toBe(false);
    expect(checkbox.hasAttribute("aria-invalid")).toBe(false);
  });
});
