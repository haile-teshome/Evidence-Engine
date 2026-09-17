// Tests for the API client's transport layer and request contracts.
//
// Run: pnpm test
//
// `fetch` is mocked throughout. What is under test is the code around it: how a
// failing response becomes an error the UI can show, whether credentials are
// attached, whether abort signals are honoured, and whether a non-JSON body
// from a crashed backend produces a usable message rather than a parse error.
//
// The credential assertions matter most: these headers carry the user's API
// keys, and a request that attaches them to the wrong place, or fails to
// attach them at all, is a security or a support problem respectively.
import { describe, it, expect, beforeEach, vi } from "vitest";

import { AIService } from "../mockServices";
import { apiConfig, keyHeaders } from "../apiClient";
import { setDbKey, setContactEmail } from "../dbKeys";

type FetchCall = { url: string; init: RequestInit };
// Not Partial<Response>: Response.body is a ReadableStream, and these
// fixtures carry the body as the string the client will read via text().
type FakeRes = { ok?: boolean; status?: number; body?: string };

function mockFetch(handler: (call: FetchCall) => FakeRes | Promise<FakeRes>) {
  const calls: FetchCall[] = [];
  const fn = vi.fn(async (url: any, init: any = {}) => {
    const call = { url: String(url), init };
    calls.push(call);
    const res = await handler(call);
    return {
      ok: res.ok ?? true,
      status: res.status ?? 200,
      text: async () => res.body ?? "{}",
      json: async () => JSON.parse(res.body ?? "{}"),
      headers: new Headers(),
    } as unknown as Response;
  });
  vi.stubGlobal("fetch", fn);
  return calls;
}

const ok = (body: any) => ({ ok: true, status: 200, body: JSON.stringify(body) });

beforeEach(() => {
  localStorage.clear();
  vi.unstubAllGlobals();
});

// --------------------------------------------------------------------------
// Transport: URLs, methods, bodies
// --------------------------------------------------------------------------

describe("request shape", () => {
  it("posts to the configured base url", async () => {
    const calls = mockFetch(() => ok({ status: "missing", reason: "x" }));
    await AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" });
    expect(calls[0].url.startsWith(apiConfig.baseUrl)).toBe(true);
  });

  it("uses POST with a JSON content type", async () => {
    const calls = mockFetch(() => ok({ status: "missing" }));
    await AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" });
    expect(calls[0].init.method).toBe("POST");
    expect((calls[0].init.headers as any)["Content-Type"]).toBe("application/json");
  });

  it("serialises the request body", async () => {
    const calls = mockFetch(() => ok({ status: "missing" }));
    await AIService.fetchFullText({ Title: "A study", URL: "u", Source: "PubMed", paper_id: "p1" });
    const sent = JSON.parse(String(calls[0].init.body));
    expect(sent.Title).toBe("A study");
    expect(sent.paper_id).toBe("p1");
  });

  it("returns the parsed JSON body", async () => {
    mockFetch(() => ok({ status: "found", text: "full text", source: "Europe PMC (XML)" }));
    const out = await AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" });
    expect(out.status).toBe("found");
    expect(out.source).toBe("Europe PMC (XML)");
  });

  it("passes an abort signal through so a long run can be cancelled", async () => {
    const calls = mockFetch(() => ok({ status: "missing" }));
    const ctrl = new AbortController();
    await AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" }, ctrl.signal);
    expect(calls[0].init.signal).toBe(ctrl.signal);
  });
});

// --------------------------------------------------------------------------
// Error handling. A backend failure must become a readable message.
// --------------------------------------------------------------------------

describe("error handling", () => {
  it("throws with the backend's detail message on a 4xx", async () => {
    mockFetch(() => ({ ok: false, status: 422, body: JSON.stringify({ detail: "paper is required" }) }));
    await expect(AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" }))
      .rejects.toThrow(/paper is required/);
  });

  it("throws on a 500 rather than returning a broken object", async () => {
    mockFetch(() => ({ ok: false, status: 500, body: JSON.stringify({ detail: "boom" }) }));
    await expect(AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" }))
      .rejects.toThrow();
  });

  it("produces a usable message when the body is not JSON", async () => {
    /* A crashed backend or a proxy error page returns HTML. The user should
       see something, not a JSON parse failure. */
    mockFetch(() => ({ ok: false, status: 502, body: "<html>Bad Gateway</html>" }));
    await expect(AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" }))
      .rejects.toThrow(/Bad Gateway|502/);
  });

  it("produces a message when the error body is empty", async () => {
    mockFetch(() => ({ ok: false, status: 503, body: "" }));
    await expect(AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" }))
      .rejects.toThrow(/503|failed/i);
  });

  it("propagates a network failure", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("offline"))));
    await expect(AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" }))
      .rejects.toThrow(/offline/);
  });

  it("an aborted request rejects rather than hanging", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(
      Object.assign(new Error("The operation was aborted"), { name: "AbortError" }))));
    await expect(AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" }))
      .rejects.toThrow();
  });

  it("a 200 with an empty body does not throw", async () => {
    mockFetch(() => ({ ok: true, status: 200, body: "" }));
    await expect(AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" }))
      .resolves.toBeDefined();
  });
});

// --------------------------------------------------------------------------
// Credentials. These headers carry the user's keys on every request.
// --------------------------------------------------------------------------

describe("credential headers", () => {
  it("sends no key headers when nothing is configured", () => {
    const h = keyHeaders();
    expect(JSON.stringify(h)).not.toMatch(/sk-|AIza/);
  });

  it("attaches a configured database key to the request", async () => {
    setDbKey("core", "core-secret-key");
    const calls = mockFetch(() => ok({ status: "missing" }));
    await AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" });
    expect((calls[0].init.headers as any)["X-DB-Core-Key"]).toBe("core-secret-key");
  });

  it("attaches a valid contact email", async () => {
    setContactEmail("j.smith@ucsf.edu");
    const calls = mockFetch(() => ok({ status: "missing" }));
    await AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" });
    expect((calls[0].init.headers as any)["X-User-Contact-Email"]).toBe("j.smith@ucsf.edu");
  });

  it("does NOT attach a placeholder contact email", async () => {
    setContactEmail("someone@example.com");
    const calls = mockFetch(() => ok({ status: "missing" }));
    await AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" });
    expect((calls[0].init.headers as any)["X-User-Contact-Email"]).toBeUndefined();
  });

  it("never puts a credential in the URL, where it would be logged", async () => {
    setDbKey("core", "core-secret-key");
    setContactEmail("j.smith@ucsf.edu");
    const calls = mockFetch(() => ok({ status: "missing" }));
    await AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" });
    expect(calls[0].url).not.toContain("core-secret-key");
    expect(calls[0].url).not.toContain("ucsf.edu");
  });

  it("never puts a credential in the request body", async () => {
    setDbKey("core", "core-secret-key");
    const calls = mockFetch(() => ok({ status: "missing" }));
    await AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" });
    expect(String(calls[0].init.body)).not.toContain("core-secret-key");
  });
});

// --------------------------------------------------------------------------
// Full-text acquisition contract. The miss payload drives the manual-retrieval
// UI, so its shape is load-bearing.
// --------------------------------------------------------------------------

describe("fetchFullText miss payload", () => {
  it("carries a reason, a reason code, and links", async () => {
    mockFetch(() => ok({
      status: "missing",
      reason: "Paywalled. Needs library or interlibrary loan access.",
      reason_code: "paywalled",
      oa_status: "closed",
      doi: "10.1/abc",
      pmid: "31946617",
      links: { doi: "https://doi.org/10.1/abc", pubmed: "https://pubmed.ncbi.nlm.nih.gov/31946617/" },
    }));
    const out = await AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" });
    expect(out.reason_code).toBe("paywalled");
    expect(out.links?.doi).toContain("doi.org");
    expect(out.doi).toBe("10.1/abc");
  });

  it("a found result carries the text and its source tier", async () => {
    mockFetch(() => ok({ status: "found", text: "body", source: "Open access (nature.com)",
                         pdf_key: "abc123" }));
    const out = await AIService.fetchFullText({ Title: "T", URL: "", Source: "PubMed" });
    expect(out.status).toBe("found");
    expect(out.pdf_key).toBe("abc123");
  });
});

// --------------------------------------------------------------------------
// A representative sample of the other endpoints, checked for path, payload
// shaping, and error propagation.
// --------------------------------------------------------------------------

describe("endpoint methods", () => {
  it("detectFramework posts the text and returns the framework", async () => {
    const calls = mockFetch(() => ok({ framework: "pcc" }));
    const out: any = await AIService.detectFramework("A scoping review of AI");
    expect(calls[0].url).toContain("/framework/detect");
    expect(JSON.stringify(out)).toContain("pcc");
  });

  it("buildSearch posts the PICO frame", async () => {
    const calls = mockFetch(() => ok({ query: '("dental"[tiab])' }));
    await AIService.buildSearch({ population: "adults", intervention: "metformin" } as any);
    expect(calls[0].url).toContain("/search/build");
    expect(String(calls[0].init.body)).toContain("metformin");
  });

  it("extractFromText posts the text and query", async () => {
    const calls = mockFetch(() => ok({ summary: "s", spans: [], values: [], evidence: [] }));
    await AIService.extractFromText("We randomised 128 adults.", "sample size");
    expect(calls[0].url).toContain("/extract/text");
    const sent = JSON.parse(String(calls[0].init.body));
    expect(sent.text).toContain("128 adults");
    expect(sent.query).toBe("sample size");
  });

  it("extractFromText includes the configured model", async () => {
    const calls = mockFetch(() => ok({ summary: "s", spans: [], values: [] }));
    await AIService.extractFromText("t", "q");
    expect(JSON.parse(String(calls[0].init.body)).model).toBe(apiConfig.model);
  });

  it("fetchCitations posts the snowball direction", async () => {
    const calls = mockFetch(() => ok({ papers: [] }));
    await AIService.fetchCitations("A title", "Backward (References)", 10, ["PubMed"]);
    expect(calls[0].url).toContain("/citations");
    expect(String(calls[0].init.body)).toContain("Backward");
  });

  it("checkIntegrity posts the papers", async () => {
    const calls = mockFetch(() => ok({ results: [] }));
    await AIService.checkIntegrity([{ paper_id: "p1", Title: "T" } as any]);
    expect(calls[0].url).toContain("/integrity/check");
  });

  it("an endpoint failure surfaces as a rejected promise", async () => {
    mockFetch(() => ({ ok: false, status: 500, body: JSON.stringify({ detail: "model down" }) }));
    await expect(AIService.extractFromText("t", "q")).rejects.toThrow(/model down/);
  });

  it("detectFramework deliberately falls back instead of throwing", async () => {
    /* Framework detection is advisory. Failing it must not block the user from
       starting a review, so it defaults to PICO rather than propagating. */
    mockFetch(() => ({ ok: false, status: 500, body: JSON.stringify({ detail: "down" }) }));
    await expect(AIService.detectFramework("text")).resolves.toBe("pico");
  });

  it("detectFramework returns pcc when the backend says so", async () => {
    mockFetch(() => ok({ framework: "pcc" }));
    await expect(AIService.detectFramework("A scoping review")).resolves.toBe("pcc");
  });

  it("detectFramework normalises an unexpected value to pico", async () => {
    mockFetch(() => ok({ framework: "nonsense" }));
    await expect(AIService.detectFramework("text")).resolves.toBe("pico");
  });

  it("every method attaches credentials, not just the first", async () => {
    setDbKey("core", "k");
    const calls = mockFetch(() => ok({}));
    await AIService.detectFramework("t");
    await AIService.extractFromText("t", "q");
    for (const c of calls) {
      expect((c.init.headers as any)["X-DB-Core-Key"]).toBe("k");
    }
  });
});
