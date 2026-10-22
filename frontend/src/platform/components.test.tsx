// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { formatCount, formatDateTime, formatPercent, formatStatus, PlatformStatus, statusVariant } from "./components";

afterEach(cleanup);

describe("platform presentation", () => {
  it.each([
    ["in_progress", "In progress"],
    ["adjudication_ready", "Ready for adjudication"],
    ["conventional_ml", "Conventional ML"],
    ["llm_finetune", "LLM fine-tuning"],
    ["lora", "LoRA"],
    ["minio", "MinIO"],
    ["f1_macro", "F1 macro"],
    ["gpu_runtime", "GPU runtime"],
  ])("formats %s as %s", (value, label) => {
    expect(formatStatus(value)).toBe(label);
  });

  it("keeps lifecycle states distinct and readable", () => {
    render(<><PlatformStatus value="open" /><PlatformStatus value="closed" /><PlatformStatus value="cancelled" /></>);
    expect(screen.getByText("Open").getAttribute("data-variant")).toBe("info");
    expect(screen.getByText("Closed").getAttribute("data-variant")).toBe("neutral");
    expect(screen.getByText("Cancelled").getAttribute("data-variant")).toBe("neutral");
    expect(statusVariant("review")).toBe("warning");
    expect(statusVariant("blocked")).toBe("error");
  });

  it("formats counts and missing dates consistently", () => {
    expect(formatCount(1, "document")).toBe("1 document");
    expect(formatCount(2, "document")).toBe("2 documents");
    expect(formatCount(2, "person", "people")).toBe("2 people");
    expect(formatPercent(0.875)).toBe("87.5%");
    expect(formatDateTime(null)).toBe("Not recorded");
    expect(formatDateTime("invalid", "No date")).toBe("No date");
  });
});
