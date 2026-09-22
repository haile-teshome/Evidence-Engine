// Component tests. jsdom, no network.
//
// Run: pnpm test
//
// The PRISMA flow diagram gets the most attention here because its numbers are
// published verbatim in the paper. A wrong count, or a stage that silently
// drops studies so the arithmetic no longer reconciles, is a reporting error
// that a reader can catch but the app cannot.
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { createRef } from "react";
import { FileText, Search } from "lucide-react";

import { EmptyState } from "../EmptyState";
import { PicoCards } from "../PicoCards";
import { ControlPane, InlineStat, PaneDivider } from "../ControlPane";
import { QueryDiff } from "../QueryDiff";
import { PrismaFlow } from "../PrismaFlow";
import { TaskProgressCard } from "../TaskProgressCard";
import { RapidScreen } from "../RapidScreen";
import { Button } from "../ui/button";
import {
  DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem,
} from "../ui/dropdown-menu";

// --------------------------------------------------------------------------
// EmptyState
// --------------------------------------------------------------------------

describe("EmptyState", () => {
  it("renders the headline", () => {
    render(<EmptyState icon={FileText} title="No studies yet" />);
    expect(screen.getByText("No studies yet")).toBeInTheDocument();
  });

  it("renders the description when given", () => {
    render(<EmptyState icon={FileText} title="T" description="Run a search to begin." />);
    expect(screen.getByText("Run a search to begin.")).toBeInTheDocument();
  });

  it("omits the description cleanly when absent", () => {
    const { container } = render(<EmptyState icon={FileText} title="T" />);
    expect(container.querySelectorAll("p")).toHaveLength(0);
  });

  it("renders an action button and fires its handler", () => {
    const onClick = vi.fn();
    render(<EmptyState icon={FileText} title="T" action={{ label: "Search", onClick }} />);
    fireEvent.click(screen.getByRole("button", { name: /search/i }));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("renders no button when there is no action", () => {
    render(<EmptyState icon={FileText} title="T" />);
    expect(screen.queryByRole("button")).toBeNull();
  });
});

// --------------------------------------------------------------------------
// PicoCards: shows the frame the review is screened against.
// --------------------------------------------------------------------------

describe("PicoCards", () => {
  const pico = { population: "adults", intervention: "metformin",
                 comparator: "placebo", outcome: "HbA1c" };

  it("renders every PICO value", () => {
    render(<PicoCards pico={pico as any} framework="pico" />);
    for (const v of Object.values(pico)) {
      expect(screen.getByText(new RegExp(v, "i"))).toBeInTheDocument();
    }
  });

  it("renders PCC elements when the framework is PCC", () => {
    render(<PicoCards pico={{ population: "any patients", concept: "AI on records",
                              context: "any setting" } as any} framework="pcc" />);
    expect(screen.getByText(/AI on records/i)).toBeInTheDocument();
    expect(screen.getByText(/any setting/i)).toBeInTheDocument();
  });

  it("does not show intervention or outcome under PCC", () => {
    /* Showing a PICO-only element on a scoping review misrepresents the frame
       the studies were actually judged against. */
    const { container } = render(
      <PicoCards pico={{ population: "p", concept: "c", context: "x" } as any} framework="pcc" />);
    expect(container.textContent).not.toMatch(/Comparator/i);
  });

  it("renders with empty values without crashing", () => {
    expect(() => render(<PicoCards pico={{} as any} framework="pico" />)).not.toThrow();
  });

  it("defaults to a frame when none is given", () => {
    expect(() => render(<PicoCards pico={pico as any} />)).not.toThrow();
  });
});

// --------------------------------------------------------------------------
// ControlPane
// --------------------------------------------------------------------------

describe("ControlPane", () => {
  it("renders stats and actions", () => {
    render(<ControlPane stats={<span>56 included</span>} actions={<button>Fetch all</button>} />);
    expect(screen.getByText("56 included")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /fetch all/i })).toBeInTheDocument();
  });

  it("renders with an empty action area", () => {
    // `actions` is a required prop, so the empty case is null, not omitted.
    expect(() => render(<ControlPane stats={<span>x</span>} actions={null} />)).not.toThrow();
  });
});

describe("InlineStat", () => {
  it("shows the value and label", () => {
    render(<InlineStat icon={FileText} value={56} label="Included" />);
    expect(screen.getByText("56")).toBeInTheDocument();
    expect(screen.getByText("Included")).toBeInTheDocument();
  });

  it("shows a hint when given", () => {
    const { container } = render(
      <InlineStat icon={FileText} value={21} label="Acquired" hint="38%" />);
    expect(container.textContent).toContain("38%");
  });

  it("renders a zero value rather than hiding it", () => {
    /* "0 missing" and "missing not computed" mean different things. */
    render(<InlineStat icon={FileText} value={0} label="Missing" />);
    expect(screen.getByText("0")).toBeInTheDocument();
  });

  it("renders a divider", () => {
    expect(() => render(<PaneDivider />)).not.toThrow();
  });
});

// --------------------------------------------------------------------------
// QueryDiff: shows what an optimisation changed in the search string.
// --------------------------------------------------------------------------

describe("QueryDiff", () => {
  it("renders both queries", () => {
    const { container } = render(
      <QueryDiff previous='("dental"[tiab])' current='("dental"[tiab] OR "oral"[tiab])' />);
    expect(container.textContent).toContain("dental");
    expect(container.textContent).toContain("oral");
  });

  it("handles an unchanged query", () => {
    expect(() => render(<QueryDiff previous="same" current="same" />)).not.toThrow();
  });

  it("handles an empty previous query, which is the first run", () => {
    expect(() => render(<QueryDiff previous="" current='("dental"[tiab])' />)).not.toThrow();
  });

  it("handles both empty", () => {
    expect(() => render(<QueryDiff previous="" current="" />)).not.toThrow();
  });
});

// --------------------------------------------------------------------------
// TaskProgressCard
// --------------------------------------------------------------------------

describe("TaskProgressCard", () => {
  // Matches TaskRecord exactly: `startedAt` drives the elapsed-time ticker and
  // `log` is read without a guard, so both are required.
  const task = {
    kind: "screen" as const,
    taskId: "task-1",
    status: "running" as const,
    startedAt: Date.now() - 5000,
    stages: [{ id: "s1", label: "Screening", status: "running" as const }],
    log: [],
    progress: { done: 33, total: 120, label: "A study" },
    detail: "A study",
  };

  it("renders the supplied title", () => {
    render(<TaskProgressCard task={task as any} title="Abstract screening" />);
    expect(screen.getByText(/Abstract screening/i)).toBeInTheDocument();
  });

  it("shows progress out of the total", () => {
    const { container } = render(<TaskProgressCard task={task as any} title="T" />);
    expect(container.textContent).toContain("33");
    expect(container.textContent).toContain("120");
  });

  it("renders a task with no progress yet", () => {
    expect(() => render(
      <TaskProgressCard task={{ ...task, progress: undefined } as any} title="T" />)).not.toThrow();
  });

  it("does not divide by zero when the total is zero", () => {
    const { container } = render(<TaskProgressCard
      task={{ ...task, progress: { done: 0, total: 0 } } as any} title="T" />);
    expect(container.textContent).not.toContain("NaN");
    expect(container.textContent).not.toContain("Infinity");
  });

  it("renders a completed task", () => {
    expect(() => render(
      <TaskProgressCard task={{ ...task, status: "done" } as any} title="T" />)).not.toThrow();
  });

  it("renders a cancelled task", () => {
    expect(() => render(
      <TaskProgressCard task={{ ...task, status: "canceled" } as any} title="T" />)).not.toThrow();
  });

  it("fires the cancel handler", () => {
    const onCancel = vi.fn();
    render(<TaskProgressCard task={task as any} title="T" onCancel={onCancel} />);
    const btn = screen.queryByRole("button", { name: /cancel/i });
    if (btn) {
      fireEvent.click(btn);
      expect(onCancel).toHaveBeenCalled();
    }
  });
});

// --------------------------------------------------------------------------
// PrismaFlow: the numbers that get published.
// --------------------------------------------------------------------------

const counts = {
  identified: 6440,
  source_counts: { PubMed: 5200, "Europe PMC": 1240 },
  duplicates_removed: 393,
  after_duplicates: 6047,
  screened: 120,
  excluded_total: 64,
  exclusion_breakdown: { "No AI/ML model": 27, "No linked records": 21, "Wrong design": 16 },
  ft_exclusion_breakdown: { "Full text unavailable": 5 },
  included_final: 51,
};

describe("PrismaFlow", () => {
  it("renders the identified count", () => {
    const { container } = render(<PrismaFlow counts={counts as any} />);
    expect(container.textContent).toMatch(/6[,.\s]?440/);
  });

  it("renders the duplicates removed", () => {
    const { container } = render(<PrismaFlow counts={counts as any} />);
    expect(container.textContent).toContain("393");
  });

  it("renders the number screened", () => {
    const { container } = render(<PrismaFlow counts={counts as any} />);
    expect(container.textContent).toContain("120");
  });

  it("renders every exclusion reason with its count", () => {
    /* PRISMA requires reasons for exclusion to be reported, not just a total. */
    const { container } = render(<PrismaFlow counts={counts as any} />);
    for (const [reason, n] of Object.entries(counts.exclusion_breakdown)) {
      expect(container.textContent).toContain(reason);
      expect(container.textContent).toContain(String(n));
    }
  });

  it("the arithmetic reconciles: screened minus excluded equals sought", () => {
    /* PRISMA 2020 does not print an "after duplicates" box, so the check that
       matters is that the stages add up. A reader will do this subtraction. */
    const { container } = render(<PrismaFlow counts={counts as any} />);
    const sought = counts.screened - counts.excluded_total;   // 120 - 64 = 56
    expect(container.textContent).toContain(`n = ${sought}`);
  });

  it("the exclusion reasons sum to the excluded total", () => {
    const sum = Object.values(counts.exclusion_breakdown).reduce((a, b) => a + b, 0);
    expect(sum).toBe(counts.excluded_total);
  });

  it("never renders a negative count", () => {
    /* Inconsistent inputs used to be able to produce a negative "assessed". */
    const { container } = render(<PrismaFlow counts={{
      ...counts, screened: 10, excluded_total: 500,
    } as any} />);
    expect(container.textContent).not.toMatch(/-\d/);
  });

  it("renders with a minimal counts object", () => {
    expect(() => render(<PrismaFlow counts={{
      identified: 0, duplicates_removed: 0, screened: 0,
      excluded_total: 0, exclusion_breakdown: {},
    } as any} />)).not.toThrow();
  });

  it("renders with no exclusion reasons", () => {
    const { container } = render(<PrismaFlow counts={{
      ...counts, exclusion_breakdown: {},
    } as any} />);
    expect(container.textContent).toContain("120");
  });

  it("renders the per-source identified breakdown", () => {
    const { container } = render(<PrismaFlow counts={counts as any} />);
    expect(container.textContent).toContain("PubMed");
  });

  it("renders full-text exclusion reasons separately from abstract ones", () => {
    const { container } = render(<PrismaFlow counts={counts as any} />);
    expect(container.textContent).toContain("Full text unavailable");
  });

  it("handles full-text results being absent", () => {
    expect(() => render(
      <PrismaFlow counts={counts as any} fullTextResults={null} />)).not.toThrow();
  });

  it("handles zero identified without dividing by zero", () => {
    const { container } = render(<PrismaFlow counts={{
      identified: 0, duplicates_removed: 0, screened: 0,
      excluded_total: 0, exclusion_breakdown: {},
    } as any} />);
    expect(container.textContent).not.toContain("NaN");
    expect(container.textContent).not.toContain("Infinity");
  });
});

// --------------------------------------------------------------------------
// RapidScreen keyboard shortcuts
// --------------------------------------------------------------------------

describe("RapidScreen shortcuts", () => {
  const items = [
    { id: "a", title: "First study", source: "PubMed", url: "", text: "abstract a", aiInclude: true, reason: "" },
    { id: "b", title: "Second study", source: "PubMed", url: "", text: "abstract b", aiInclude: false, reason: "" },
  ];
  const mount = (onDecide: (id: string, d: "include" | "exclude") => void = () => {}) => render(
    <RapidScreen open items={items} onClose={() => {}} onDecide={onDecide} hideAI />,
  );

  it("moves forward on the right arrow", () => {
    mount();
    expect(screen.getByText("First study")).toBeTruthy();
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(screen.getByText("Second study")).toBeTruthy();
  });

  it("moves back on the left arrow", () => {
    mount();
    fireEvent.keyDown(window, { key: "ArrowRight" });
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(screen.getByText("First study")).toBeTruthy();
  });

  it("includes on a and excludes on s", () => {
    const calls: [string, string][] = [];
    mount((id, d) => calls.push([id, d]));
    fireEvent.keyDown(window, { key: "a" });
    fireEvent.keyDown(window, { key: "s" });
    expect(calls).toEqual([["a", "include"], ["b", "exclude"]]);
  });

  it("no longer answers to the old i, e and b letters", () => {
    const calls: [string, string][] = [];
    mount((id, d) => calls.push([id, d]));
    fireEvent.keyDown(window, { key: "i" });
    fireEvent.keyDown(window, { key: "e" });
    expect(calls).toEqual([]);
    fireEvent.keyDown(window, { key: "ArrowRight" });
    fireEvent.keyDown(window, { key: "b" });
    expect(screen.getByText("Second study")).toBeTruthy();
  });

  it("labels every button with the key it actually listens for", () => {
    mount();
    expect(screen.getByText("Include").closest("button")?.textContent).toContain("a");
    expect(screen.getByText("Exclude").closest("button")?.textContent).toContain("s");
    expect(screen.getByText("Skip").closest("button")?.textContent).toContain("→");
    expect(screen.getByText("Back").closest("button")?.textContent).toContain("←");
  });
});

// --------------------------------------------------------------------------
// Button ref forwarding.
//
// Not a detail. On React 18 a ref given to a plain function component is
// dropped silently, and every Radix primitive that wraps a trigger in
// `asChild` measures its anchor through that ref. Without it the Export menu
// mounted with both its items and positioned itself off the top of the screen:
// the button looked dead and nothing threw. jsdom has no layout, so the
// positioning itself is not observable here; the ref is, and it is the cause.
// --------------------------------------------------------------------------

describe("Button", () => {
  it("forwards its ref to the underlying DOM node", () => {
    const ref = createRef<HTMLButtonElement>();
    render(<Button ref={ref}>Press</Button>);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
    expect(ref.current?.textContent).toBe("Press");
  });

  it("forwards its ref through asChild, the path Radix triggers use", () => {
    const ref = createRef<HTMLAnchorElement>();
    render(<Button asChild><a ref={ref} href="#x">Link</a></Button>);
    expect(ref.current).toBeInstanceOf(HTMLAnchorElement);
  });

  it("gives a Radix trigger a real anchor element to measure", () => {
    const ref = createRef<HTMLButtonElement>();
    render(
      <DropdownMenu>
        <DropdownMenuTrigger asChild><Button ref={ref}>Export</Button></DropdownMenuTrigger>
        <DropdownMenuContent><DropdownMenuItem>CSV</DropdownMenuItem></DropdownMenuContent>
      </DropdownMenu>,
    );
    // A trigger Radix cannot measure leaves the popper at its pre-measurement
    // transform, which is what put the menu off-screen.
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
    expect(ref.current?.getAttribute("aria-haspopup")).toBe("menu");
  });
});
