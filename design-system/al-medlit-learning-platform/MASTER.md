# AL-MedLit design system

> This file is the source of truth for authenticated product UI. Page-specific
> files in `pages/` may refine layout, but must not replace these foundations.

**Product:** Scientific NLP and LLM data, annotation, and model-training workspace  
**Foundation:** Astryx Neutral  
**Style:** Restrained, precise, data-dense, and work-focused  
**Theme:** Light only until a complete dark theme is designed and verified

## Product principles

- Put the current task, project, dataset, and model identity ahead of decoration.
- Keep application navigation stable while independent functional workspaces change.
- Treat a project as scientific context and provenance, not as the visual owner of every
  workflow.
- Prefer compact tables and unframed sections for comparison and repeated work.
- Use cards only for repeated records, dialogs, and genuinely framed tools.
- Reveal advanced ML configuration progressively without hiding reproducibility data.
- Treat Learning Loop and Guideline Learning as planned modules until released.

## Tokens

Use Astryx theme tokens through the semantic aliases in `frontend/src/styles.css`.
Raw colors belong only in that token layer.

| Role | CSS token |
|---|---|
| Canvas | `--surface-canvas` |
| Raised surface | `--surface-raised` |
| Muted surface | `--surface-muted` |
| Primary text | `--text-primary` |
| Secondary text | `--text-secondary` |
| Muted text | `--text-muted` |
| Border | `--border-subtle` |
| Emphasized border | `--border-strong` |
| Primary action | `--action-primary` |
| Focus ring | `--focus-ring` |
| Success, warning, error, info | `--status-*-text`, `--status-*-surface`, `--status-*-border` |

Status must always be communicated with text or an icon in addition to color.
Normal text must meet WCAG AA contrast.

## Typography

- UI: the Astryx body sans family, with system sans fallbacks.
- Document reading: system serif through `--font-reading`.
- Code, identifiers, checksums, and model revisions: the system monospace family.
- Use 16px minimum for document reading and 13px minimum for compact UI body text.
- Use sentence case, zero letter spacing for headings, and no viewport-scaled type.
- Reserve serif type for source-document content, never navigation, headings, or metrics.

## Spacing and shape

- Base spacing scale: 4, 8, 12, 16, 24, 32, and 40px.
- Compact control height: 40px; touch/coarse-pointer target: at least 44x44px.
- Radii: 4px for small controls, 6px for inputs, and 8px maximum for containers.
- Pills and circles are allowed only when the shape conveys status, identity, or count.
- Shadows are reserved for menus, popovers, and dialogs. Sections and cards use borders.

## Shell and navigation

- Use one authenticated shell with global My Work, Projects, Training, and Models.
- Expose workspace governance from the workspace menu and instance Administration only
  to deployment superusers.
- Desktop uses a compact global rail plus a sticky top bar. Mobile uses a drawer with
  separate Go to and Switch workspace groups.
- Project navigation contains Overview, Data, Predictions, Tasks, Team & rounds, Quality & review,
  Activity, and Settings. Train and Models are independent global workspaces.
- Keep the global header and current functional context as one coherent sticky stack.
- Show the current global destination and project section with `aria-current`.
- Preserve deep links, browser Back/Forward behavior, route titles, and focus placement.
- On small screens, replace the project sidebar with one labeled section selector.

## Data and training UI

- Training has its own workspace-level Runs, New Training, Training Data, Runtimes, and
  Run Detail routes. Project is a visible context selector/filter.
- Models has its own workspace-wide registry and immutable version detail routes.
- Tables use sticky identity columns on small screens and retain visible scroll affordance.
- Model rows show name, family, base model, task, version, training method, dataset,
  status, and evaluation without opening a detail view.
- Training uses five steps: Task & data, Model & recipe, Environment,
  Evaluation & storage, and Review.
- Mobile training shows `Step X of 5: Name`; Back and Continue remain visible.
- Loading, empty, unavailable-runtime, validation, failure, and success states are required.
- Technical JSON, hashes, package lists, and storage paths use progressive disclosure.

## Accessibility and interaction

- All workflows must be keyboard complete with visible `:focus-visible` treatment.
- Hover-revealed actions must also appear with `:focus-within` and on touch devices.
- Icon-only actions require a Lucide icon, accessible name, and tooltip.
- Forms use persistent labels, field-level errors, and focus the first invalid field.
- Dialogs trap focus, support Escape when safe, restore trigger focus, and lock page scroll.
- Respect reduced motion; interaction transitions should be 150-300ms and must not shift layout.
- Verify at 375, 768, 1024, and 1440px, landscape mobile, and 200% zoom.

## Component ownership

- `frontend/src/components/AppShell.tsx` owns the shared shell, responsive navigation,
  route announcements, and workspace context. Pages reuse this shell.
- Use Astryx buttons, dialogs, status, and form components where they support the
  interaction. Retain semantic native tables and labeled controls in existing workflows.
  Apply their shared styles in `frontend/src/styles.css`.
- Use semantic HTML for document text, annotation spans, tables, and landmarks.
  Use Astryx layout components for new general-purpose composition when they fit.
- Treat the semantic aliases above as the application token contract. Legacy aliases
  resolve to the same values while existing selectors migrate.
- `frontend/src/platform/navigation.ts` owns released project routes.
  `frontend/src/navigation/moduleNavigation.ts` owns global routes.
- `frontend/src/platform/components.tsx` owns status labels, tones, counts, and dates.

## Do not use

- Marketing heroes, community/forum patterns, ornamental gradients, or decorative blobs.
- Health, longevity, or evidence-specific styling as the platform-wide visual identity.
- Green as a default action color; reserve semantic colors for actual status.
- Serif dashboard headings or metrics.
- Nested cards, radii above 8px, hover-only commands, or color-only status.
- Placeholder-only labels, hidden focus rings, emoji icons, or unsupported dark-mode claims.
