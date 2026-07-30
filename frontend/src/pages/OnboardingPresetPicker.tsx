import { useState } from "react";
import type { ReactElement } from "react";
import { Banner } from "@astryxdesign/core/Banner";
import { ClickableCard } from "@astryxdesign/core/ClickableCard";

import { setWorkspaceCapability } from "@/api/client";

const TEAM_PRESETS: { key: string; label: string; blurb: string }[] = [
  { key: "annotate", label: "Annotate", blurb: "Just annotation. The lightest setup." },
  {
    key: "train",
    label: "Train Models",
    blurb: "Independent training and model registry without annotation.",
  },
  { key: "annotate_train", label: "Annotate + Train", blurb: "Annotate and train models." },
  {
    key: "annotate_train_al",
    label: "+ Active Learning",
    blurb: "Enable the planned active-learning capability.",
  },
  { key: "full", label: "Full", blurb: "Everything your deployment supports." },
];

const PERSONAL_PRESETS: { key: string; label: string; blurb: string }[] = [
  {
    key: "annotate",
    label: "Start Annotating",
    blurb: "Private projects, PMID import, and your My Work queue.",
  },
  {
    key: "annotate_train",
    label: "Annotate + Train",
    blurb: "Add model-training tools after you build a labeled set.",
  },
  {
    key: "train",
    label: "Train Models",
    blurb: "Use uploaded or public datasets without enabling annotation.",
  },
  {
    key: "full",
    label: "Full Research Loop",
    blurb: "Enable every available tool for an advanced solo workflow.",
  },
];

export default function OnboardingPresetPicker({
  isPersonalWorkspace = false,
  workspaceId,
  onDone,
}: {
  isPersonalWorkspace?: boolean;
  workspaceId: number;
  onDone: () => Promise<void>;
}): ReactElement {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const presets = isPersonalWorkspace ? PERSONAL_PRESETS : TEAM_PRESETS;

  async function choose(preset: string): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      await setWorkspaceCapability(workspaceId, preset);
      await onDone();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not set preset");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell preset-shell">
      <section className="preset-panel" aria-labelledby="preset-title">
        <div>
          <h1 id="preset-title">
            {isPersonalWorkspace ? "Set up your personal workspace" : "Choose your workspace"}
          </h1>
          <p>
            {isPersonalWorkspace
              ? "Start with the tools you need now. You can add more later."
              : "You can change this anytime in workspace settings."}
          </p>
        </div>
        {error ? (
          <Banner
            className="form-banner"
            status="error"
            title="Preset update failed"
            description={error}
          />
        ) : null}
        <div className="preset-grid">
          {presets.map((preset, index) => (
            <ClickableCard
              key={preset.key}
              label={preset.label}
              isDisabled={busy}
              onClick={() => void choose(preset.key)}
              className={index === 0 && isPersonalWorkspace ? "preset-card recommended" : "preset-card"}
            >
              {index === 0 && isPersonalWorkspace ? <small>Recommended</small> : null}
              <strong>{preset.label}</strong>
              <span>{preset.blurb}</span>
            </ClickableCard>
          ))}
        </div>
      </section>
    </main>
  );
}
