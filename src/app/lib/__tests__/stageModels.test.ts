// Choosing a model per stage.
//
// Run: pnpm test
//
// The point of the per-stage map is that a stage the reviewer set runs what
// they set, and a stage they never thought about runs the default. Both halves
// matter: an override that silently stops applying sends the wrong model to a
// stage, and a default that stops applying sends an empty model name, which
// fails the request rather than falling back.
import { describe, it, expect } from "vitest";

import {
  MODEL_STAGES, ModelStage, cleanStageModels, isScreeningStage, modelForStage,
} from "../stageModels";

describe("modelForStage", () => {
  it("uses the stage's own model when one is set", () => {
    expect(modelForStage("text-extraction", { "text-extraction": "qwen2.5:7b" }, "leads"))
      .toBe("qwen2.5:7b");
  });

  it("falls back to the default for a stage with no choice", () => {
    expect(modelForStage("writing", { "text-extraction": "qwen2.5:7b" }, "leads")).toBe("leads");
  });

  it("falls back when there are no choices at all", () => {
    expect(modelForStage("home", {}, "leads")).toBe("leads");
    expect(modelForStage("home", null, "leads")).toBe("leads");
    expect(modelForStage("home", undefined, "leads")).toBe("leads");
  });

  it("treats a blank override as no override", () => {
    // What a half-finished edit leaves behind. Sending "" as a model name
    // fails the request, so it must read as absent rather than as a choice.
    expect(modelForStage("home", { home: "" }, "leads")).toBe("leads");
    expect(modelForStage("home", { home: "   " }, "leads")).toBe("leads");
  });

  it("keeps the stages independent", () => {
    const picks = { "abstract-screening": "leads", "text-extraction": "qwen2.5:7b" } as const;
    expect(modelForStage("abstract-screening", picks, "gpt-4o")).toBe("leads");
    expect(modelForStage("text-extraction", picks, "gpt-4o")).toBe("qwen2.5:7b");
    expect(modelForStage("quality-assessment", picks, "gpt-4o")).toBe("gpt-4o");
  });
});

describe("cleanStageModels", () => {
  it("keeps known stages", () => {
    expect(cleanStageModels({ "text-extraction": "qwen2.5:7b" }))
      .toEqual({ "text-extraction": "qwen2.5:7b" });
  });

  it("drops a stage this build no longer has", () => {
    // A saved session written by an older build. A stale key is invisible in
    // the UI and could never be cleared from there.
    expect(cleanStageModels({ "full-text-acquisition": "qwen2.5:7b", home: "gpt-4o" }))
      .toEqual({ home: "gpt-4o" });
  });

  it("drops empty and non-string values", () => {
    expect(cleanStageModels({ home: "", writing: "  ", planning: 7, "meta-analysis": null }))
      .toEqual({});
  });

  it("survives anything that is not an object", () => {
    for (const junk of [null, undefined, "leads", 3, []]) {
      expect(cleanStageModels(junk)).toEqual({});
    }
  });
});

describe("the stage list", () => {
  it("has no duplicates and labels every stage", () => {
    const ids = MODEL_STAGES.map(s => s.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const stage of MODEL_STAGES) {
      expect(stage.label.length).toBeGreaterThan(0);
    }
  });

  it("marks exactly the stages that ask a model for a screening verdict", () => {
    // LEADS-Mistral is offered only here, because it returns nothing usable
    // when asked to do anything else.
    const screening = MODEL_STAGES.filter(s => isScreeningStage(s.id)).map(s => s.id);
    expect(screening).toEqual(["abstract-screening", "full-text-screening", "citation-snowball"]);
  });

  it("covers every stage the type allows", () => {
    // A stage missing from the list is a stage with no picker: its model could
    // never be set, and nothing would say why.
    const all: ModelStage[] = [
      "home", "planning", "abstract-screening", "full-text-screening",
      "citation-snowball", "table-extraction", "text-extraction",
      "quality-assessment", "meta-analysis", "writing",
    ];
    expect(MODEL_STAGES.map(s => s.id).sort()).toEqual([...all].sort());
  });
});
