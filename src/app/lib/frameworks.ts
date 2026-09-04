// Question-frame registry (mirrors Backend/frameworks.py). The clarifier pills,
// the frame editor, the PICO/PCC cards and the screening columns all read the
// element list from here instead of hardcoding P/I/C/O. Adding a frame is a new
// entry, not a code change.

export type FrameworkId = "pico" | "pcc";

export interface FrameworkElement {
  id: string;      // canonical key stored on the frame object
  label: string;   // display noun
  letter: string;  // badge letter
  desc: string;    // one-line guidance (tooltip / clarifier)
}

export interface FrameworkDef {
  id: FrameworkId;
  label: string;       // "PICO" / "PCC"
  reviewType: string;  // short plain-language review type
  blurb: string;       // one short, plain sentence on when to use it
  elements: FrameworkElement[];
}

export const FRAMEWORKS: Record<FrameworkId, FrameworkDef> = {
  pico: {
    id: "pico",
    label: "PICO",
    reviewType: "Does a treatment work?",
    blurb: "Use this when you're comparing a treatment or exposure to something else to see if it helps.",
    elements: [
      { id: "population", label: "Population", letter: "P", desc: "Who you're studying." },
      { id: "intervention", label: "Intervention", letter: "I", desc: "The treatment or thing being tested." },
      { id: "comparator", label: "Comparator", letter: "C", desc: "What you compare it against (placebo, usual care, another option)." },
      { id: "outcome", label: "Outcome", letter: "O", desc: "The result you're measuring." },
    ],
  },
  pcc: {
    id: "pcc",
    label: "PCC",
    reviewType: "What's out there on a topic?",
    blurb: "Use this to survey a broad topic and see what research exists, instead of testing a treatment.",
    elements: [
      { id: "population", label: "Population", letter: "P", desc: "Who or what group you're interested in." },
      { id: "concept", label: "Concept", letter: "C", desc: "The main idea or topic you're looking at." },
      { id: "context", label: "Context", letter: "C", desc: "The setting: place, time, or situation." },
    ],
  },
};

export const FRAMEWORK_IDS: FrameworkId[] = ["pico", "pcc"];

export function isFrameworkId(x: unknown): x is FrameworkId {
  return x === "pico" || x === "pcc";
}

export function normalizeFramework(x: unknown): FrameworkId {
  return isFrameworkId(x) ? x : "pico";
}

export function frameworkOf(id: unknown): FrameworkDef {
  return FRAMEWORKS[normalizeFramework(id)];
}

export function elementIds(id: unknown): string[] {
  return frameworkOf(id).elements.map((e) => e.id);
}

export function labelFor(id: unknown, elementId: string): string {
  const e = frameworkOf(id).elements.find((x) => x.id === elementId);
  return e ? e.label : elementId.charAt(0).toUpperCase() + elementId.slice(1);
}
