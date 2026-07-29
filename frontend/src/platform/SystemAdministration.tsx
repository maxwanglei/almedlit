import { ServerCog } from "lucide-react";

import { shouldHandleSpaClick } from "@/components/ModuleSwitcher";

import { PlatformEmpty, PlatformPageHeader, PlatformSection } from "./components";

const SECTIONS = [
  { id: "health", label: "System health" },
  { id: "plugins", label: "Plugins" },
  { id: "audit", label: "Audit" },
  { id: "settings", label: "Instance settings" },
] as const;

export default function SystemAdministration({
  pathname,
  onNavigate,
}: {
  pathname: string;
  onNavigate: (path: string) => void;
}): React.ReactElement {
  const requested = pathname.match(/^\/admin\/([^/]+)$/)?.[1];
  const active =
    SECTIONS.find((section) => section.id === requested) ?? SECTIONS[0];

  return (
    <main id="main-content" className="module-workspace-main" tabIndex={-1}>
      <div className="platform-page system-administration-page">
        <PlatformPageHeader
          title="System administration"
          description="Deployment-wide health, extensions, audit, and instance policy."
        />
        <nav className="platform-subnav" aria-label="System administration">
          {SECTIONS.map((section) => (
            <a
              key={section.id}
              href={`/admin/${section.id}`}
              aria-current={active.id === section.id ? "page" : undefined}
              onClick={(event) => {
                if (!shouldHandleSpaClick(event)) {
                  return;
                }
                event.preventDefault();
                onNavigate(`/admin/${section.id}`);
              }}
            >
              {section.label}
            </a>
          ))}
        </nav>
        <PlatformSection
          title={active.label}
          description="This surface is reserved for deployment superusers and never grants workspace permissions."
        >
          <div className="system-administration-placeholder">
            <ServerCog size={22} aria-hidden="true" />
            <PlatformEmpty
              title={`${active.label} integration is not configured`}
              detail="Connect the corresponding deployment service before exposing operational commands."
            />
          </div>
        </PlatformSection>
      </div>
    </main>
  );
}
