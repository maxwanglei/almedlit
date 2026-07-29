// @vitest-environment jsdom

import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { BrowserRouter, useNavigate } from "react-router-dom";

import {
  allowedSearchValue,
  buildSearchUrl,
  useSearchState,
} from "./useSearchState";

function SearchHarness(): React.ReactElement {
  const [params, updateSearch] = useSearchState();
  return (
    <div>
      <output aria-label="Search value">{params.get("view") ?? "missing"}</output>
      <button type="button" onClick={() => updateSearch({ view: "annotate" })}>
        Push
      </button>
      <button
        type="button"
        onClick={() => updateSearch({ view: "progress", stale: null }, "replace")}
      >
        Replace
      </button>
      <button
        type="button"
        onClick={() => {
          updateSearch({ project: 7 });
          updateSearch({ document: 11 }, "replace");
          updateSearch({ view: "annotate" }, "replace");
        }}
      >
        Compose updates
      </button>
    </div>
  );
}

function CrossRouteHarness(): React.ReactElement {
  const [, updateSearch] = useSearchState();
  const navigate = useNavigate();
  return (
    <button
      type="button"
      onClick={() => {
        navigate("/projects");
        updateSearch({ view: "annotate" }, "replace");
      }}
    >
      Navigate with retained callback
    </button>
  );
}

beforeEach(() => {
  window.history.replaceState(null, "", "/annotator/workbench?view=progress&stale=1");
});

afterEach(cleanup);

describe("search state", () => {
  it("builds URLs without losing unrelated state or hashes", () => {
    expect(
      buildSearchUrl(
        {
          pathname: "/annotator/workbench",
          search: "?project=7&pane=queue",
          hash: "#selection",
        },
        { pane: "document", assignment: 12, project: null },
      ),
    ).toBe(
      "/annotator/workbench?pane=document&assignment=12#selection",
    );
  });

  it("falls back safely for invalid enumerated values", () => {
    const invalid = allowedSearchValue(
      new URLSearchParams("pane=unknown"),
      "pane",
      ["document", "queue"] as const,
      "document",
    );
    expect(invalid).toEqual({ value: "document", isValid: false });
  });

  it("pushes, replaces, and restores state on popstate", () => {
    render(
      <BrowserRouter>
        <SearchHarness />
      </BrowserRouter>,
    );
    expect(screen.getByLabelText("Search value").textContent).toBe("progress");

    fireEvent.click(screen.getByRole("button", { name: "Push" }));
    expect(window.location.search).toContain("view=annotate");
    expect(screen.getByLabelText("Search value").textContent).toBe("annotate");

    fireEvent.click(screen.getByRole("button", { name: "Replace" }));
    expect(window.location.search).toBe("?view=progress");

    act(() => {
      window.history.pushState(null, "", "/annotator/workbench?view=annotate");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(screen.getByLabelText("Search value").textContent).toBe("annotate");
  });

  it("composes sequential updates without dropping earlier query keys", () => {
    render(
      <BrowserRouter>
        <SearchHarness />
      </BrowserRouter>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Compose updates" }));

    expect(window.location.search).toBe(
      "?view=annotate&stale=1&project=7&document=11",
    );
    expect(screen.getByLabelText("Search value").textContent).toBe("annotate");
  });

  it("ignores a retained screen update after navigation changes the route", () => {
    render(
      <BrowserRouter>
        <CrossRouteHarness />
      </BrowserRouter>,
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Navigate with retained callback" }),
    );

    expect(window.location.pathname).toBe("/projects");
    expect(window.location.search).toBe("");
  });
});
