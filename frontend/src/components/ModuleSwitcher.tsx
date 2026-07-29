import {
  useEffect,
  useRef,
  useState,
  type ReactNode,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { Button } from "@astryxdesign/core/Button";
import { IconButton } from "@astryxdesign/core/IconButton";
import {
  BrainCircuit,
  ClipboardList,
  FolderKanban,
  Menu,
  PackageOpen,
  ShieldCheck,
  X,
  type LucideIcon,
} from "lucide-react";

import type {
  ModuleDestination,
  ModuleId,
  UtilityDestination,
} from "@/navigation/moduleNavigation";

interface ModuleSwitcherProps {
  modules: ModuleDestination[];
  currentModuleId: ModuleId | null;
  onNavigate: (path: string) => void;
  presentation?: "responsive" | "desktop" | "mobile";
  workspaceSwitcher?: ReactNode;
  workspaceSettings?: UtilityDestination | null;
  workspaceSettingsCurrent?: boolean;
  onLogout?: () => void;
}

const MODULE_ICONS: Record<ModuleId, LucideIcon> = {
  "my-work": ClipboardList,
  projects: FolderKanban,
  training: BrainCircuit,
  models: PackageOpen,
  administration: ShieldCheck,
};

export function shouldHandleSpaClick(
  event: Pick<
    ReactMouseEvent<HTMLAnchorElement>,
    "button" | "defaultPrevented" | "metaKey" | "ctrlKey" | "shiftKey" | "altKey"
  >,
): boolean {
  return (
    event.button === 0 &&
    !event.defaultPrevented &&
    !event.metaKey &&
    !event.ctrlKey &&
    !event.shiftKey &&
    !event.altKey
  );
}

export default function ModuleSwitcher({
  modules,
  currentModuleId,
  onNavigate,
  presentation = "responsive",
  workspaceSwitcher,
  workspaceSettings = null,
  workspaceSettingsCurrent = false,
  onLogout,
}: ModuleSwitcherProps): React.ReactElement | null {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRootRef = useRef<HTMLDivElement>(null);
  const menuButtonRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!menuOpen) {
      return;
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.requestAnimationFrame(() => {
      menuRootRef.current
        ?.querySelector<HTMLAnchorElement>("[data-drawer-initial='true']")
        ?.focus();
    });
    const handleDrawerKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        event.preventDefault();
        setMenuOpen(false);
        menuButtonRef.current?.focus();
        return;
      }
      if (event.key !== "Tab") {
        return;
      }
      const drawer = menuRootRef.current?.querySelector<HTMLElement>(
        ".module-navigation-drawer",
      );
      const focusable = Array.from(
        drawer?.querySelectorAll<HTMLElement>(
          "a[href], button:not([disabled]), select:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex='-1'])",
        ) ?? [],
      );
      if (focusable.length === 0) {
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleDrawerKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", handleDrawerKeyDown);
    };
  }, [menuOpen]);

  if (modules.length === 0) {
    return null;
  }

  const renderLinks = (drawer = false): React.ReactElement[] =>
    modules.map((module, index) => {
      const ModuleIcon = MODULE_ICONS[module.id];
      return (
        <a
          key={module.id}
          href={module.path}
          aria-current={module.id === currentModuleId ? "page" : undefined}
          data-drawer-initial={drawer && index === 0 ? "true" : undefined}
          onClick={(event) => {
            if (!shouldHandleSpaClick(event)) {
              return;
            }
            event.preventDefault();
            setMenuOpen(false);
            onNavigate(module.path);
          }}
        >
          <ModuleIcon size={18} strokeWidth={1.8} aria-hidden="true" />
          <span>{module.label}</span>
        </a>
      );
    });

  const desktopNavigation = (
    <nav className="module-navigation module-navigation--rail" aria-label="Primary">
      <span className="module-navigation-label">Workspace</span>
      <div className="module-navigation-desktop">{renderLinks()}</div>
    </nav>
  );

  const mobileNavigation = (
    <div ref={menuRootRef} className="module-navigation module-navigation--mobile">
      <div className="module-navigation-mobile">
        <IconButton
          ref={menuButtonRef}
          label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
          icon={menuOpen ? <X size={20} /> : <Menu size={20} />}
          variant="ghost"
          size="sm"
          aria-expanded={menuOpen}
          aria-controls="module-navigation-drawer"
          onClick={() => setMenuOpen((current) => !current)}
        />
        {menuOpen ? (
          <>
            <button
              type="button"
              className="module-navigation-backdrop"
              aria-label="Close navigation menu"
              tabIndex={-1}
              onClick={() => {
                setMenuOpen(false);
                menuButtonRef.current?.focus();
              }}
            />
            <section
              id="module-navigation-drawer"
              className="module-navigation-drawer"
              role="dialog"
              aria-modal="true"
              aria-labelledby="module-navigation-title"
            >
              <header>
                <div>
                  <strong id="module-navigation-title">Navigation</strong>
                  <span>AL-MedLit workspace</span>
                </div>
                <IconButton
                  label="Close navigation menu"
                  icon={<X size={20} />}
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setMenuOpen(false);
                    menuButtonRef.current?.focus();
                  }}
                />
              </header>
              <div className="module-navigation-drawer-section">
                <h2>Go to</h2>
                <nav aria-label="Mobile primary navigation">
                  {renderLinks(true)}
                </nav>
              </div>
              <div className="module-navigation-drawer-section">
                <h2>Switch workspace</h2>
                {workspaceSwitcher}
              </div>
              <footer>
                {workspaceSettings ? (
                  <a
                    href={workspaceSettings.path}
                    aria-current={
                      workspaceSettingsCurrent ? "page" : undefined
                    }
                    onClick={(event) => {
                      if (!shouldHandleSpaClick(event)) {
                        return;
                      }
                      event.preventDefault();
                      setMenuOpen(false);
                      onNavigate(workspaceSettings.path);
                    }}
                  >
                    {workspaceSettings.label}
                  </a>
                ) : null}
                {onLogout ? (
                  <Button
                    label="Sign out"
                    size="sm"
                    onClick={() => {
                      setMenuOpen(false);
                      onLogout();
                    }}
                  />
                ) : null}
              </footer>
            </section>
          </>
        ) : null}
      </div>
    </div>
  );

  if (presentation === "desktop") {
    return desktopNavigation;
  }
  if (presentation === "mobile") {
    return mobileNavigation;
  }
  return (
    <>
      {desktopNavigation}
      {mobileNavigation}
    </>
  );
}
