import React from "react";
import ReactDOM from "react-dom/client";
import { Theme } from "@astryxdesign/core/theme";
import { neutralTheme } from "@astryxdesign/theme-neutral/built";
import App from "./App";
import { registerPreloadErrorHandler } from "./lib/preloadErrorRecovery";
import "@astryxdesign/core/reset.css";
import "@astryxdesign/core/astryx.css";
import "@astryxdesign/theme-neutral/theme.css";
import "./styles.css";

const unregisterPreloadErrorHandler = registerPreloadErrorHandler();
if (import.meta.hot) {
  import.meta.hot.dispose(unregisterPreloadErrorHandler);
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Theme theme={neutralTheme} mode="light">
      <App />
    </Theme>
  </React.StrictMode>
);
