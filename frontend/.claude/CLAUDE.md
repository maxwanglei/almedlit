# Frontend UI changes

Read `../../design-system/al-medlit-learning-platform/MASTER.md` from this file
before changing layout, typography, controls, status, or navigation. It owns the
application's theme, tokens, component boundaries, and interaction rules.

Discover installed Astryx APIs before using them. Run `npx astryx build "<idea>"`
for a component kit, then `npx astryx component <Name>` for the props and examples.
Use `npx astryx template <name>` for composition examples.

Keep the Astryx reset, component styles, theme styles, and application styles loaded
in that order in `src/main.tsx`. The application uses a light theme.

Preserve semantic native elements in document editors and existing data workflows.
Reuse shared presentation helpers and controls. Update the shared style rules when
fixing repeated controls rather than adding page-specific overrides.

Verify UI changes in the running browser at desktop and mobile widths. Test loading,
failure, keyboard interaction, and focus when changing those behaviors.
