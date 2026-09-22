// Tests for folding clarifying answers back into the question frame.
//
// Run: pnpm test
//
// The reviewer is asked a direct question ("what population?"), answers it, and
// then the PCC card reads "None specified" for that very element. The answers
// were only ever appended to the inference prompt as prose, so whether they
// survived was up to the model, and on scoping questions it routinely decided
// the population was not restrictive and returned nothing.
//
// Being ignored after being asked is worse than not being asked.
import { describe, it, expect } from "vitest";

import { applyClarifyAnswers } from "../HomePage";

const empty = {
  population: "", intervention: "", comparator: "", outcome: "",
  concept: "", context: "", framework: "pcc",
};

describe("applyClarifyAnswers", () => {
  it("fills an element the model left empty", () => {
    const out = applyClarifyAnswers(empty, { population: "any patients, any age" });
    expect(out.population).toBe("any patients, any age");
  });

  it("does NOT overwrite an element the model already operationalised", () => {
    /* "adults aged 18 and over with type 2 diabetes" is the same answer, stated
       better, and is what actually gets searched and screened. */
    const rich = { ...empty, population: "adults aged 18 and over with type 2 diabetes" };
    const out = applyClarifyAnswers(rich, { population: "adults" });
    expect(out.population).toBe("adults aged 18 and over with type 2 diabetes");
  });

  it("treats a whitespace-only element as empty", () => {
    const out = applyClarifyAnswers({ ...empty, population: "   " }, { population: "adults" });
    expect(out.population).toBe("adults");
  });

  it("maps every question id that names an element", () => {
    const out = applyClarifyAnswers(empty, {
      population: "adults", intervention: "metformin", comparator: "placebo",
      outcome: "HbA1c", concept: "AI tools", context: "primary care",
    });
    expect(out).toMatchObject({
      population: "adults", intervention: "metformin", comparator: "placebo",
      outcome: "HbA1c", concept: "AI tools", context: "primary care",
    });
  });

  it("ignores free-form question ids that name no element", () => {
    /* Clarifying questions are not all about elements; some ask about scope or
       study design and have no field to land in. */
    const out = applyClarifyAnswers(empty, { "study design": "RCTs only" });
    expect(out).toEqual(empty);
  });

  it("is case and whitespace insensitive about the id", () => {
    expect(applyClarifyAnswers(empty, { " Population ": "adults" }).population).toBe("adults");
  });

  it("ignores a skipped question", () => {
    /* Skipping records an empty string; that is not an answer. */
    expect(applyClarifyAnswers(empty, { population: "" }).population).toBe("");
  });

  it("ignores a whitespace-only answer", () => {
    expect(applyClarifyAnswers(empty, { population: "   " }).population).toBe("");
  });

  it("does not mutate the input", () => {
    const src = { ...empty };
    applyClarifyAnswers(src, { population: "adults" });
    expect(src.population).toBe("");
  });

  it("leaves the frame untouched when there are no answers", () => {
    expect(applyClarifyAnswers(empty, {})).toEqual(empty);
  });

  it("never drops the framework", () => {
    expect(applyClarifyAnswers(empty, { population: "adults" }).framework).toBe("pcc");
  });
});
