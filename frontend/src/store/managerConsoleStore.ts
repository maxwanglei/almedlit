import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

export type ManagerRole = "manager" | "admin";
export type ManagerAppMode = "project" | "admin";
export type ManagerTab =
  | "overview"
  | "documents"
  | "tasks"
  | "progress"
  | "loop"
  | "models"
  | "guidelines"
  | "export";
export type LoopPhase = "select" | "adjudicate" | "reflect";
export type TasksSubView = "schema" | "assign";
export type AdminSection = "users" | "plugins" | "health" | "audit" | "settings";
export type ManagerModal = "newProject" | null;

interface ManagerConsoleState {
  role: ManagerRole;
  appMode: ManagerAppMode;
  currentTab: ManagerTab;
  loopPhase: LoopPhase;
  tasksSubView: TasksSubView;
  adminSection: AdminSection;
  modal: ManagerModal;
  setRole: (role: ManagerRole) => void;
  setAppMode: (appMode: ManagerAppMode) => void;
  setCurrentTab: (tab: ManagerTab) => void;
  setLoopPhase: (phase: LoopPhase) => void;
  setTasksSubView: (subView: TasksSubView) => void;
  setAdminSection: (section: AdminSection) => void;
  setModal: (modal: ManagerModal) => void;
  openProjectTab: (tab: ManagerTab, phase?: LoopPhase) => void;
}

export const useManagerConsoleStore = create<ManagerConsoleState>()(
  immer((set) => ({
    role: "manager",
    appMode: "project",
    currentTab: "overview",
    loopPhase: "select",
    tasksSubView: "schema",
    adminSection: "users",
    modal: null,
    setRole: (role) =>
      set((state) => {
        state.role = role;
        if (role !== "admin" && state.appMode === "admin") {
          state.appMode = "project";
        }
      }),
    setAppMode: (appMode) =>
      set((state) => {
        state.appMode = appMode;
      }),
    setCurrentTab: (tab) =>
      set((state) => {
        state.currentTab = tab;
      }),
    setLoopPhase: (phase) =>
      set((state) => {
        state.loopPhase = phase;
      }),
    setTasksSubView: (subView) =>
      set((state) => {
        state.tasksSubView = subView;
      }),
    setAdminSection: (section) =>
      set((state) => {
        state.adminSection = section;
      }),
    setModal: (modal) =>
      set((state) => {
        state.modal = modal;
      }),
    openProjectTab: (tab, phase) =>
      set((state) => {
        state.appMode = "project";
        state.currentTab = tab;
        if (phase) {
          state.loopPhase = phase;
        }
      }),
  })),
);
