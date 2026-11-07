// Tests for what a PDF turns into on the way in.
//
// Run: pnpm test
//
// Both cases here come from real articles a reviewer dropped into the app. They
// matter because neither failure announces itself: the study is imported, it
// just carries a typesetter's job number as its name, or carries "¼" through
// every extracted cell where the page says "=".
import { describe, it, expect } from "vitest";

import { guessTitle, repairTextLayer } from "../pdfImport";

// The first page of Bhat et al., JNCI 2021, as pdf.js reads it: the title, two
// spaces, then the author list.
const JNCI_FIRST_PAGE =
  "Dopamine Receptor Antagonists, Radiation, and Cholesterol Biosynthesis in Mouse Models "
  + "of Glioblastoma  Kruttika Bhat, PhD, 1,‡  Mohammad Saki, PhD, 1,‡  Fei Cheng, PhD, 1  "
  + "Ling He, DDS, PhD, 1  Le Zhang, MD, PhD, 1 Angeliki Ioannidis MS, 1";

describe("the title a PDF arrives with", () => {
  it("uses the metadata title when it reads like a title", () => {
    const real = "Effects of Levetiracetam and Lacosamide on survival and seizure control in "
               + "IDH-wild type glioblastoma during temozolomide plus radiation adjuvant therapy";
    expect(guessTitle(real, "", "1-s2.0-S2772529423010202-main.pdf")).toBe(real);
  });

  it("refuses the typesetter's job number and reads the page instead", () => {
    // "OP-JNCI210018 1094..1104" is this article's metadata Title. It passes
    // every length check, so it used to name the study everywhere in the app.
    expect(guessTitle("OP-JNCI210018 1094..1104", JNCI_FIRST_PAGE, "djab018.pdf"))
      .toBe("Dopamine Receptor Antagonists, Radiation, and Cholesterol Biosynthesis "
            + "in Mouse Models of Glioblastoma");
  });

  it("does not mistake a running head for the title", () => {
    expect(guessTitle("", "Brain and Spine 4 (2024) 102732", "my_paper.pdf")).toBe("my paper");
  });

  it("falls back to the filename when the page yields nothing", () => {
    expect(guessTitle("", "", "Smith_2020_final.pdf")).toBe("Smith 2020");
  });
});

describe("a broken text layer", () => {
  it("restores the equals signs a bad character map replaced", () => {
    const broken = "hazard ratio ¼ 3.96, 95% CI ¼ 0.29 to 12.40, n ¼ 8 mice, dose ¼ 10 Gy";
    expect(repairTextLayer(broken)).toBe(
      "hazard ratio = 3.96, 95% CI = 0.29 to 12.40, n = 8 mice, dose = 10 Gy");
  });

  it("restores plus signs in the same documents", () => {
    const broken = "a ¼ 1, b ¼ 2, c ¼ 3 and quetiapine þ atorvastatin";
    expect(repairTextLayer(broken)).toContain("quetiapine + atorvastatin");
  });

  it("leaves a genuine fraction alone", () => {
    // One occurrence is a quarter, not a mangled "=".
    const fine = "about ¼ of the cohort reported an adverse event";
    expect(repairTextLayer(fine)).toBe(fine);
  });
});
