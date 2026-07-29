import type { ReactNode } from "react";
import { Button } from "@astryxdesign/core/Button";

import BrandLogo from "@/components/BrandLogo";
import ModuleSwitcher, {
  shouldHandleSpaClick,
} from "@/components/ModuleSwitcher";
import type {
  ModuleDestination,
  ModuleId,
  UtilityDestination,
} from "@/navigation/moduleNavigation";

interface AppShellProps {
  modules: ModuleDestination[];
  currentModuleId: ModuleId | null;
  workspaceSettings: UtilityDestination | null;
  workspaceSettingsCurrent: boolean;
  homePath: string;
  workspaceSwitcher: ReactNode;
  mobileWorkspaceSwitcher: ReactNode;
  announcement: string;
  onNavigate: (path: string) => void;
  onLogout: () => void;
  children: ReactNode;
}

export default function AppShell({
  modules,
  currentModuleId,
  workspaceSettings,
  workspaceSettingsCurrent,
  homePath,
  workspaceSwitcher,
  mobileWorkspaceSwitcher,
  announcement,
  onNavigate,
  onLogout,
  children,
}: AppShellProps): React.ReactElement {
  return (
    <div className="app-shell unified-app-shell">
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <header className="topbar unified-topbar">
        <div className="topbar-brand-group">
          <a
            className="brand-identity"
            href={homePath}
            onClick={(event) => {
              if (!shouldHandleSpaClick(event)) {
                return;
              }
              event.preventDefault();
              onNavigate(homePath);
            }}
          >
            <BrandLogo />
            <span className="brand-block">
              <strong className="brand-title">AL-MedLit</strong>
              <span className="brand-subtitle">NLP learning workspace</span>
            </span>
          </a>
          <div className="unified-mobile-navigation">
            <ModuleSwitcher
              presentation="mobile"
              modules={modules}
              currentModuleId={currentModuleId}
              workspaceSwitcher={mobileWorkspaceSwitcher}
              workspaceSettings={workspaceSettings}
              workspaceSettingsCurrent={workspaceSettingsCurrent}
              onNavigate={onNavigate}
              onLogout={onLogout}
            />
          </div>
        </div>
        <div className="unified-topbar-actions unified-topbar-actions--desktop">
          {workspaceSwitcher}
          {workspaceSettings ? (
            <a
              className="logout-button"
              href={workspaceSettings.path}
              aria-current={workspaceSettingsCurrent ? "page" : undefined}
              onClick={(event) => {
                if (!shouldHandleSpaClick(event)) {
                  return;
                }
                event.preventDefault();
                onNavigate(workspaceSettings.path);
              }}
            >
              {workspaceSettings.label}
            </a>
          ) : null}
          <Button
            className="logout-button"
            label="Sign out"
            size="sm"
            onClick={onLogout}
          />
        </div>
      </header>
      <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {announcement}
      </p>
      <div className="unified-shell-frame">
        <aside className="global-navigation-rail">
          <ModuleSwitcher
            presentation="desktop"
            modules={modules}
            currentModuleId={currentModuleId}
            onNavigate={onNavigate}
          />
        </aside>
        <div className="unified-shell-content">{children}</div>
      </div>
    </div>
  );
}
