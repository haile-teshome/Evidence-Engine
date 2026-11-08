// Which model each stage's requests actually carry.
//
// Run: pnpm test
//
// Choosing a model per stage is only real if the request for that stage sends
// it. The failure this guards against is silent: a call site left reading the
// default still works, still returns values, and still looks right in the UI,
// while the reviewer's choice for that stage was never applied. Nothing in the
// app would show it, so it is asserted here, endpoint by endpoint.
//
// The three functions reached from two stages are the sharpest case. Screening
// a snowball candidate posts to /screen/abstract, the same endpoint abstract
// screening uses, so the stage has to come from the caller rather than from the
// endpoint.
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { readFileSync } from "node:fs";

import { AIService, DataAggregator, MetaAnalysisService, QualityService } from "../mockServices";
import { apiConfig, modelFor } from "../apiClient";
import type { ModelStage } from "../stageModels";

type Sent = { url: string; model: string };

let sent: Sent[] = [];

beforeEach(() => {
  sent = [];
  vi.stubGlobal("fetch", vi.fn(async (url: any, init: any = {}) => {
    let model = "";
    try { model = JSON.parse(String(init.body || "{}")).model || ""; } catch { /* non-JSON */ }
    sent.push({ url: String(url), model });
    return {
      ok: true, status: 200, headers: new Headers(),
      text: async () => "{}", json: async () => ({}),
    } as unknown as Response;
  }));
  apiConfig.model = "default-model";
  apiConfig.stageModels = {};
});

afterEach(() => {
  apiConfig.model = "qwen2.5:7b";
  apiConfig.stageModels = {};
  vi.unstubAllGlobals();
});

/** The model the one request made by `run` carried. */
async function modelSentBy(run: () => Promise<unknown>): Promise<string> {
  sent = [];
  await run().catch(() => { /* the fake reply is empty; only the request matters */ });
  expect(sent.length).toBeGreaterThan(0);
  return sent[0].model;
}

const paper = { id: "p1", title: "T", abstract: "A", source: "PubMed", url: "u" } as any;
const pico = { population: "", intervention: "", comparator: "", outcome: "" } as any;
const row = { Title: "T", URL: "u", Source: "PubMed", Abstract: "A", paper_id: "p1" };
const fields = [{ id: "n", label: "Sample size", type: "number" }];

// Each stage, the call that runs it, and what that call posts to.
const CASES: { stage: ModelStage; what: string; run: () => Promise<unknown> }[] = [
  { stage: "home", what: "PICO inference",
    run: () => AIService.inferPicoAndQuery("a question") },
  { stage: "home", what: "ranking search results",
    run: () => DataAggregator.rerankByRelevance([paper], pico) },
  { stage: "planning", what: "per-database query adaptation",
    run: () => AIService.adaptQueriesPerSource("query", ["PubMed"]) },
  { stage: "abstract-screening", what: "abstract screening",
    run: () => AIService.screenPaperMultiAgent(paper, pico) },
  { stage: "full-text-screening", what: "full-text screening",
    run: () => AIService.screenFullTextMultiAgent(row, [], []) },
  { stage: "table-extraction", what: "table extraction",
    run: () => AIService.extractTablesTraced(row) },
  { stage: "text-extraction", what: "filling the extraction form",
    run: () => AIService.extractFieldsRich("text", fields) },
  { stage: "text-extraction", what: "extracting from pasted text",
    run: () => AIService.extractFromText("text", "query") },
  { stage: "quality-assessment", what: "risk of bias",
    run: () => QualityService.assessPaper(paper) },
  { stage: "meta-analysis", what: "pulling effect sizes",
    run: () => MetaAnalysisService.extract([paper], "mortality") },
  { stage: "writing", what: "drafting the protocol",
    run: () => AIService.generateProtocol({ title: "t" }) },
];

describe("every stage sends its own model", () => {
  for (const { stage, what, run } of CASES) {
    it(`${what} uses the ${stage} model`, async () => {
      apiConfig.stageModels = { [stage]: `model-for-${stage}` };
      expect(await modelSentBy(run)).toBe(`model-for-${stage}`);
    });

    it(`${what} falls back to the default when ${stage} has no choice`, async () => {
      apiConfig.stageModels = {};
      expect(await modelSentBy(run)).toBe("default-model");
    });
  }

  it("leaves the other stages alone", async () => {
    // Setting one stage must not change any other. This is the whole promise.
    apiConfig.stageModels = { "text-extraction": "only-here" };
    expect(await modelSentBy(() => AIService.extractFieldsRich("text", fields))).toBe("only-here");
    expect(await modelSentBy(() => AIService.screenPaperMultiAgent(paper, pico)))
      .toBe("default-model");
    expect(await modelSentBy(() => AIService.inferPicoAndQuery("q"))).toBe("default-model");
  });
});

describe("a call reached from two stages takes the stage from its caller", () => {
  beforeEach(() => {
    apiConfig.stageModels = {
      "abstract-screening": "screening-model",
      "citation-snowball": "snowball-model",
      "text-extraction": "text-model",
      "table-extraction": "table-model",
      home: "home-model",
    };
  });

  it("screens a snowball candidate with the snowball model", async () => {
    // Same endpoint as abstract screening. The stage is the reviewer's stage.
    expect(await modelSentBy(() =>
      AIService.screenPaperMultiAgent(paper, pico, [], [], undefined, "", "citation-snowball")))
      .toBe("snowball-model");
    expect(await modelSentBy(() => AIService.screenPaperMultiAgent(paper, pico)))
      .toBe("screening-model");
  });

  it("ranks snowball citations with the snowball model", async () => {
    expect(await modelSentBy(() =>
      DataAggregator.rerankByRelevance([paper], pico, [], [], -1, undefined, undefined, undefined,
                                       "citation-snowball")))
      .toBe("snowball-model");
    expect(await modelSentBy(() => DataAggregator.rerankByRelevance([paper], pico)))
      .toBe("home-model");
  });

  it("fills a form from tables with the table-extraction model", async () => {
    expect(await modelSentBy(() =>
      AIService.extractFieldsRich("text", fields, "", undefined, "", "table-extraction")))
      .toBe("table-model");
    expect(await modelSentBy(() => AIService.extractFieldsRich("text", fields)))
      .toBe("text-model");
  });
});

describe("calls that belong to no stage", () => {
  it("name a session and read a PDF's own metadata with the default", async () => {
    // Housekeeping, not a review judgement: a title read by another model is
    // not a different review, and nobody would set a model for it.
    apiConfig.stageModels = { home: "home-model", writing: "writing-model" };
    expect(await modelSentBy(() => AIService.generateSessionTitle("goal"))).toBe("default-model");
    expect(await modelSentBy(() => AIService.extractPdfMetadata("text", "f.pdf")))
      .toBe("default-model");
  });
});

describe("modelFor", () => {
  it("reads the live config, so the store's updates take effect at once", () => {
    apiConfig.model = "a";
    apiConfig.stageModels = {};
    expect(modelFor("writing")).toBe("a");
    apiConfig.stageModels = { writing: "b" };
    expect(modelFor("writing")).toBe("b");
    expect(modelFor("home")).toBe("a");
  });
});

describe("no call site is left reading the default", () => {
  it("routes every model through modelFor or defaultModel", () => {
    // The regression this file exists for. A new endpoint that writes
    // `model: apiConfig.model` would work, look right, and quietly ignore the
    // reviewer's choice for its stage; only its absence from the source is
    // evidence that no such site exists.
    const src = readFileSync("src/app/lib/apiClient.ts", "utf8");
    const offenders = src
      .split("\n")
      .map((line, i) => ({ n: i + 1, line: line.trim() }))
      .filter(({ line }) => /apiConfig\.model\b/.test(line))
      // The two resolvers are where reading it is the point, and a comment
      // mentioning it is not a call site.
      .filter(({ line }) => !line.startsWith("//"))
      .filter(({ line }) => !/return (modelForStage|apiConfig\.model;)/.test(line));

    expect(offenders).toEqual([]);
  });
});
