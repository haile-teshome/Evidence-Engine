// Tests for the sidebar, and specifically its automatic model selection.
//
// Run: pnpm test
//
// This component has already cost real work. Its preference chain puts
// LEADS-mistral first, so the moment that tag is pulled it is auto-selected on
// every load. Two full screening runs in this project were executed under LEADS
// while the reviewer believed they were testing qwen, and the only symptom was
// that the exported decisions disagreed with the PICO panel beside them.
//
// Nothing about that is visible in the UI, which is exactly why it needs tests.
// Every case below pins one rule of the selection chain so a change to the
// ordering has to be deliberate.
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { useEffect, useState } from "react";

import { StoreProvider, useStore } from "../../lib/store";
import { Sidebar } from "../Sidebar";

function mockModels(models: string[], running = true) {
  vi.stubGlobal("fetch", vi.fn(async (url: any) => {
    if (String(url).includes("/api/models/local")) {
      return {
        ok: true, status: 200, headers: new Headers(),
        json: async () => ({ models, running }),
        text: async () => JSON.stringify({ models, running }),
      };
    }
    return {
      ok: true, status: 200, headers: new Headers(),
      json: async () => ({}), text: async () => "{}",
    };
  }));
}

function mountSidebar(initialModel?: string) {
  let api: any;
  // Gate: the Sidebar's model-selection effect runs on mount, so the initial
  // model has to be in the store BEFORE it mounts. Rendering both together
  // races the seed against the effect.
  function Harness() {
    const s = useStore();
    api = s;
    const [ready, setReady] = useState(!initialModel);
    useEffect(() => {
      if (initialModel && s.model !== initialModel) s.setModel(initialModel);
      setReady(true);
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    return ready ? <Sidebar /> : null;
  }
  const utils = render(
    <StoreProvider>
      <Harness />
    </StoreProvider>,
  );
  return { ...utils, model: () => api?.model, store: () => api };
}

const LEADS = "hf.co/mradermacher/leads-mistral-7b-v1-GGUF:latest";

beforeEach(() => {
  localStorage.clear();
  vi.unstubAllGlobals();
});

// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------

describe("Sidebar rendering", () => {
  it("renders without a running Ollama", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("connection refused"))));
    const { container } = mountSidebar();
    await waitFor(() => expect(container.textContent?.length).toBeGreaterThan(0));
  });

  it("renders with no local models installed", async () => {
    mockModels([]);
    const { container } = mountSidebar();
    await waitFor(() => expect(container.textContent?.length).toBeGreaterThan(0));
  });

  it("renders the navigation", async () => {
    mockModels(["qwen2.5:7b"]);
    const { container } = mountSidebar();
    await waitFor(() => expect(container.querySelectorAll("button").length).toBeGreaterThan(3));
  });

  it("offers open-access databases and marks subscription ones as keyed", async () => {
    /* Both kinds are listed. What matters is that a source needing an API key
       is grouped as such, so the reviewer knows why it returns nothing. */
    mockModels(["qwen2.5:7b"]);
    const { container } = mountSidebar();
    await waitFor(() => expect(container.textContent).toBeTruthy());
    const text = container.textContent ?? "";
    expect(text).toContain("PubMed");
    if (text.includes("Scopus")) expect(text).toMatch(/Subscription/i);
  });

  it("does not offer Local PDFs as a searchable database", async () => {
    /* PDFs are attached from the chat, not searched. */
    mockModels(["qwen2.5:7b"]);
    const { container } = mountSidebar();
    await waitFor(() => expect(container.textContent).toBeTruthy());
    expect(container.textContent).not.toContain("Local PDFs");
  });
});

// --------------------------------------------------------------------------
// Automatic model selection. The rules, one test each.
// --------------------------------------------------------------------------

describe("automatic model selection", () => {
  it("resolves the bare 'leads' alias to the real Ollama tag", async () => {
    /* Otherwise the dropdown selection matches no SelectItem and renders blank. */
    mockModels([LEADS, "qwen2.5:7b"]);
    const view = mountSidebar("leads");
    await waitFor(() => expect(view.model()).toBe(LEADS));
  });

  it("auto-selects LEADS over qwen when both are installed", async () => {
    /* THE REGRESSION. Documented here as current behaviour, not as desirable:
       leadsTag is first in the preference chain, so a fresh load silently
       switches away from qwen. If this test starts failing because the chain
       was reordered, that is a deliberate improvement, not a break. */
    mockModels([LEADS, "qwen2.5:7b"], true);
    const view = mountSidebar("some-uninstalled-model");
    await waitFor(() => expect(view.model()).toBe(LEADS));
  });

  it("falls back to medgemma when LEADS is absent", async () => {
    mockModels(["medgemma:27b", "qwen2.5:7b"]);
    const view = mountSidebar("some-uninstalled-model");
    await waitFor(() => expect(view.model()).toBe("medgemma:27b"));
  });

  it("prefers qwen2.5 over a bare qwen tag", async () => {
    mockModels(["qwen:7b", "qwen2.5:7b"]);
    const view = mountSidebar("some-uninstalled-model");
    await waitFor(() => expect(view.model()).toBe("qwen2.5:7b"));
  });

  it("falls back to any qwen when qwen2.5 is absent", async () => {
    mockModels(["qwen:7b", "mistral:7b"]);
    const view = mountSidebar("some-uninstalled-model");
    await waitFor(() => expect(view.model()).toBe("qwen:7b"));
  });

  it("NEVER auto-selects a llama tag when another model exists", async () => {
    /* An explicit instruction in this project: llama is not the house default. */
    mockModels(["llama3.1:8b", "qwen2.5:7b"]);
    const view = mountSidebar("some-uninstalled-model");
    await waitFor(() => expect(view.model()).toBe("qwen2.5:7b"));
  });

  it("falls back to the first installed model when nothing is preferred", async () => {
    mockModels(["mistral:7b", "phi3:mini"]);
    const view = mountSidebar("some-uninstalled-model");
    await waitFor(() => expect(view.model()).toBe("mistral:7b"));
  });
});

// --------------------------------------------------------------------------
// When selection must NOT fire. These protect a deliberate choice.
// --------------------------------------------------------------------------

describe("selection leaves a deliberate choice alone", () => {
  it("keeps a model that is already installed", async () => {
    /* Picking qwen while LEADS is installed must survive a re-render. */
    mockModels([LEADS, "qwen2.5:7b"]);
    const view = mountSidebar("qwen2.5:7b");
    await waitFor(() => expect(view.model()).toBe("qwen2.5:7b"));
  });

  it.each(["claude-opus-5", "gpt-4o", "gemini-2.0-flash"])(
    "does not override the cloud model %s with a local one",
    async (cloud) => {
      mockModels([LEADS, "qwen2.5:7b"]);
      const view = mountSidebar(cloud);
      await waitFor(() => expect(view.model()).toBe(cloud));
    },
  );

  it("does not change the model when no models are installed", async () => {
    mockModels([]);
    const view = mountSidebar("qwen2.5:7b");
    await waitFor(() => expect(view.model()).toBe("qwen2.5:7b"));
  });

  it("does not change the model when the lookup fails", async () => {
    /* Ollama being down must not silently reassign the model. */
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("offline"))));
    const view = mountSidebar("qwen2.5:7b");
    await waitFor(() => expect(view.model()).toBe("qwen2.5:7b"));
  });

  it("does not change the model when the response is malformed", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true, status: 200, headers: new Headers(),
      json: async () => ({ unexpected: "shape" }), text: async () => "{}",
    })));
    const view = mountSidebar("qwen2.5:7b");
    await waitFor(() => expect(view.model()).toBe("qwen2.5:7b"));
  });

  it("selects only once, not on every render", async () => {
    mockModels([LEADS, "qwen2.5:7b"]);
    const view = mountSidebar("some-uninstalled-model");
    await waitFor(() => expect(view.model()).toBe(LEADS));
    // A later manual choice must stick rather than being re-overridden.
    view.store().setModel("qwen2.5:7b");
    await waitFor(() => expect(view.model()).toBe("qwen2.5:7b"));
  });
});

// --------------------------------------------------------------------------
// Source selection
// --------------------------------------------------------------------------

describe("source selection", () => {
  it("starts with at least one source so a search is possible", async () => {
    mockModels(["qwen2.5:7b"]);
    const view = mountSidebar();
    await waitFor(() => expect(view.store()).toBeTruthy());
    expect((view.store().sources ?? []).length).toBeGreaterThan(0);
  });

  it("toggling a source off removes exactly that one", async () => {
    mockModels(["qwen2.5:7b"]);
    const view = mountSidebar();
    await waitFor(() => expect(view.store()).toBeTruthy());
    const s = view.store();
    s.setSources(["PubMed", "Europe PMC", "OpenAlex"]);
    await waitFor(() => expect(view.store().sources).toHaveLength(3));
    s.setSources(view.store().sources.filter((x: string) => x !== "Europe PMC"));
    await waitFor(() => {
      expect(view.store().sources).toContain("PubMed");
      expect(view.store().sources).not.toContain("Europe PMC");
    });
  });

  it("allows every source to be deselected", async () => {
    /* The Planning tab guards on an empty source list, so it must be reachable. */
    mockModels(["qwen2.5:7b"]);
    const view = mountSidebar();
    await waitFor(() => expect(view.store()).toBeTruthy());
    view.store().setSources([]);
    await waitFor(() => expect(view.store().sources).toHaveLength(0));
  });
});

// --------------------------------------------------------------------------
// Credential status. The sidebar shows whether a cloud model can actually run.
// --------------------------------------------------------------------------

describe("credential status", () => {
  it("renders with a cloud model selected and no key configured", async () => {
    mockModels([]);
    const { container } = mountSidebar("claude-opus-5");
    await waitFor(() => expect(container.textContent?.length).toBeGreaterThan(0));
  });

  it("never renders a stored key into the DOM", async () => {
    const { setDbKey } = await import("../../lib/dbKeys");
    setDbKey("core", "core-secret-value-12345");
    mockModels(["qwen2.5:7b"]);
    const { container } = mountSidebar();
    await waitFor(() => expect(container.textContent).toBeTruthy());
    expect(container.innerHTML).not.toContain("core-secret-value-12345");
  });

  it("renders for a local model, which needs no key at all", async () => {
    mockModels(["qwen2.5:7b"]);
    const { container } = mountSidebar("qwen2.5:7b");
    await waitFor(() => expect(container.textContent?.length).toBeGreaterThan(0));
  });
});
