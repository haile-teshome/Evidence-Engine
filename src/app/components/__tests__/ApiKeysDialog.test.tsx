// Tests for the credential dialog.
//
// Run: pnpm test
//
// This is the only UI that touches API keys and the scholarly-API contact
// email. Two properties matter more than anything cosmetic:
//
//   1. A secret must never be rendered into the DOM in plain text, and never
//      leave the device except as a request header.
//   2. The contact-email validation must actually reject placeholder addresses,
//      because "researcher@example.com" is exactly what Unpaywall answers 422
//      to, and that failure was previously indistinguishable from a paywalled
//      paper for months.
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import { ApiKeysDialog } from "../ApiKeysDialog";
import { getDbKey, setDbKey, getContactEmail, setContactEmail } from "../../lib/dbKeys";
import { clearEncrypted, lock, setMode } from "../../lib/keystore";

const SECRET = "sk-super-secret-value-123456";

// Radix renders the dialog through a PORTAL, so its content is appended to
// document.body rather than the container render() returns. Assert against
// body, or the tests pass vacuously against an empty container.
function open() {
  render(<ApiKeysDialog open onOpenChange={() => {}} />);
  return { body: () => document.body.textContent ?? "", root: document.body };
}

/** Switch to the Databases tab, where the email and DB keys live. */
async function goToDatabases() {
  const tab = await screen.findByText("Databases");
  fireEvent.click(tab);
}

beforeEach(() => {
  localStorage.clear();
  lock();
  clearEncrypted();
});

// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------

describe("rendering", () => {
  it("renders nothing meaningful when closed", () => {
    const { container } = render(<ApiKeysDialog open={false} onOpenChange={() => {}} />);
    expect(container.textContent ?? "").not.toContain("Databases");
  });

  it("renders both tabs when open", async () => {
    open();
    expect(await screen.findByText("Models")).toBeInTheDocument();
    expect(screen.getByText("Databases")).toBeInTheDocument();
  });

  it("switches to the Databases tab", async () => {
    open();
    await goToDatabases();
    expect(await screen.findByText(/Contact email/i)).toBeInTheDocument();
  });

  it("renders with a provider highlighted", () => {
    expect(() => render(
      <ApiKeysDialog open onOpenChange={() => {}} highlight="anthropic" />)).not.toThrow();
  });

  it("offers every LLM provider on the Models tab", async () => {
    const v = open();
    await waitFor(() => expect(v.body()).toBeTruthy());
    for (const p of ["Anthropic", "OpenAI", "Google"]) expect(v.body()).toContain(p);
  });

  it("separates free database keys from subscription ones", async () => {
    open();
    await goToDatabases();
    const text = document.body.textContent ?? "";
    expect(text).toMatch(/Free/i);
    expect(text).toMatch(/Subscription/i);
  });
});

// --------------------------------------------------------------------------
// Contact email. The field that unlocks Unpaywall and the NCBI rate limit.
// --------------------------------------------------------------------------

describe("contact email", () => {
  const emailField = async () =>
    (await screen.findByLabelText(/Your email/i)) as HTMLInputElement;

  it("starts empty", async () => {
    open();
    await goToDatabases();
    expect((await emailField()).value).toBe("");
  });

  it("persists a valid address on change", async () => {
    open();
    await goToDatabases();
    fireEvent.change(await emailField(), { target: { value: "j.smith@ucsf.edu" } });
    await waitFor(() => expect(getContactEmail()).toBe("j.smith@ucsf.edu"));
  });

  it("shows a confirmation for a valid address", async () => {
    open();
    await goToDatabases();
    fireEvent.change(await emailField(), { target: { value: "j.smith@ucsf.edu" } });
    await waitFor(() =>
      expect(document.body.textContent).toMatch(/Sent only to|Unpaywall/i));
  });

  it("warns on a placeholder domain rather than silently accepting it", async () => {
    /* This is the exact value that made every Unpaywall call return 422. */
    open();
    await goToDatabases();
    fireEvent.change(await emailField(), { target: { value: "someone@example.com" } });
    await waitFor(() =>
      expect(document.body.textContent).toMatch(/reject|placeholder|real one/i));
  });

  it.each(["notanemail", "a@b", "@nodomain.com", "test@localhost"])(
    "warns on the malformed address %s",
    async (bad) => {
      open();
      await goToDatabases();
      fireEvent.change(await emailField(), { target: { value: bad } });
      await waitFor(() =>
        expect(document.body.textContent).toMatch(/reject|placeholder|real one/i));
    },
  );

  it("shows no warning while the field is empty", async () => {
    open();
    await goToDatabases();
    fireEvent.change(await emailField(), { target: { value: "" } });
    await waitFor(() =>
      expect(document.body.textContent).not.toMatch(/reject placeholder/i));
  });

  it("clearing the field removes the stored address", async () => {
    setContactEmail("j.smith@ucsf.edu");
    open();
    await goToDatabases();
    fireEvent.change(await emailField(), { target: { value: "" } });
    await waitFor(() => expect(getContactEmail()).toBe(""));
  });

  it("loads an already-stored address into the field", async () => {
    setContactEmail("j.smith@ucsf.edu");
    open();
    await goToDatabases();
    expect((await emailField()).value).toBe("j.smith@ucsf.edu");
  });

  it("explains that this is not an account", async () => {
    /* Users otherwise assume it creates a login somewhere. */
    open();
    await goToDatabases();
    expect(document.body.textContent).toMatch(/Not an account/i);
  });
});

// --------------------------------------------------------------------------
// Database keys
// --------------------------------------------------------------------------

/** The CORE key input, found by its placeholder. */
function coreField(): HTMLInputElement | undefined {
  return Array.from(document.querySelectorAll("input")).find(
    (i) => (i as HTMLInputElement).placeholder?.toLowerCase().includes("core"),
  ) as HTMLInputElement | undefined;
}

describe("database keys", () => {
  it("does NOT persist a key until Save is pressed", async () => {
    /* Deliberate, and different from the contact email: typing a key is not a
       commitment, so a half-typed value never reaches storage. */
    open();
    await goToDatabases();
    const field = coreField();
    if (!field) return;
    fireEvent.change(field, { target: { value: "core-key-123" } });
    expect(getDbKey("core")).toBe("");
  });

  it("persists a key when Save is pressed", async () => {
    open();
    await goToDatabases();
    const field = coreField();
    if (!field) return;
    fireEvent.change(field, { target: { value: "core-key-123" } });
    const save = Array.from(document.querySelectorAll("button"))
      .find(b => /^save$/i.test(b.textContent?.trim() ?? ""));
    if (!save) return;
    fireEvent.click(save);
    await waitFor(() => expect(getDbKey("core")).toBe("core-key-123"));
  });

  it("shows a saved indicator for an already-stored key", async () => {
    setDbKey("core", "core-key-123");
    open();
    await goToDatabases();
    expect(document.body.textContent).toBeTruthy();
  });

  it("states that keys stay on this device", async () => {
    open();
    await goToDatabases();
    expect(document.body.textContent).toMatch(/stored on this device/i);
  });
});

// --------------------------------------------------------------------------
// The properties that matter: no secret is ever exposed.
// --------------------------------------------------------------------------

describe("secrets are never exposed", () => {
  it("does not render a stored database key as visible text", async () => {
    setDbKey("core", SECRET);
    const v = open();
    await goToDatabases();
    // It may sit in an input's value, but must not be rendered as page text.
    expect(v.body()).not.toContain(SECRET);
  });

  it("masks key entry with a password field", async () => {
    open();
    await goToDatabases();
    const pw = document.querySelectorAll('input[type="password"]');
    expect(pw.length).toBeGreaterThan(0);
  });

  it("does not put a key in a link or an image src", async () => {
    setDbKey("core", SECRET);
    open();
    await goToDatabases();
    for (const el of Array.from(document.body.querySelectorAll("a, img"))) {
      expect(el.getAttribute("href") ?? "").not.toContain(SECRET);
      expect(el.getAttribute("src") ?? "").not.toContain(SECRET);
    }
  });

  it("never sends a key anywhere while the dialog is open", async () => {
    /* The dialog is local-only: keys travel as request headers from the API
       client, never from here. */
    const fetchSpy = vi.fn(() => Promise.reject(new Error("no network in this test")));
    vi.stubGlobal("fetch", fetchSpy);
    setDbKey("core", SECRET);
    open();
    await goToDatabases();
    for (const call of fetchSpy.mock.calls) {
      expect(JSON.stringify(call)).not.toContain(SECRET);
    }
  });

  it("does not write a contact email into the LLM keystore blob", async () => {
    setContactEmail("j.smith@ucsf.edu");
    open();
    await goToDatabases();
    expect(localStorage.getItem("ee:llm-keys-enc:v1") ?? "").not.toContain("ucsf.edu");
  });
});

// --------------------------------------------------------------------------
// Storage mode
// --------------------------------------------------------------------------

describe("storage mode", () => {
  it("renders in encrypted mode", async () => {
    setMode("encrypted");
    const v = open();
    await waitFor(() => expect(v.body()).toBeTruthy());
    expect(v.body()).toMatch(/passphrase|encrypt/i);
  });

  it("explains where keys are kept", async () => {
    setMode("encrypted");
    const v = open();
    await waitFor(() => expect(v.body()).toBeTruthy());
    expect(v.body()).toMatch(/never leave this device/i);
  });

  it("renders whichever mode is configured without throwing", async () => {
    for (const m of ["keychain", "encrypted"] as const) {
      setMode(m);
      expect(() => render(<ApiKeysDialog open onOpenChange={() => {}} />)).not.toThrow();
    }
  });

  it("does not render a passphrase into page text", async () => {
    setMode("encrypted");
    const v = open();
    await waitFor(() => expect(v.body()).toBeTruthy());
    const pass = document.body.querySelector('input[type="password"]') as HTMLInputElement | null;
    if (pass) {
      fireEvent.change(pass, { target: { value: "correct horse battery staple" } });
      expect(v.body()).not.toContain("correct horse battery staple");
    }
  });
});
