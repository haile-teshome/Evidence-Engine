import { Pico } from "../lib/mockServices";
import { frameworkOf, type FrameworkId } from "../lib/frameworks";

// Per-element accent so each frame element is instantly distinguishable. Keyed by
// element id, shared across PICO (population/intervention/comparator/outcome) and
// PCC (population/concept/context).
const ELEM_STYLE: Record<string, { badge: string; bar: string }> = {
  population:   { badge: "bg-blue-100 text-blue-700",     bar: "bg-blue-400" },
  intervention: { badge: "bg-violet-100 text-violet-700", bar: "bg-violet-400" },
  comparator:   { badge: "bg-amber-100 text-amber-700",   bar: "bg-amber-400" },
  outcome:      { badge: "bg-emerald-100 text-emerald-700", bar: "bg-emerald-400" },
  concept:      { badge: "bg-violet-100 text-violet-700", bar: "bg-violet-400" },
  context:      { badge: "bg-amber-100 text-amber-700",   bar: "bg-amber-400" },
};

type AnyFrame = Partial<Pico> & { p?: string; i?: string; c?: string; o?: string;
  concept?: string; context?: string; framework?: FrameworkId };

// Read an element's value from either the full-name keys (population/…) or the
// legacy single-letter keys (p/i/c/o) so history entries render too.
function valueFor(pico: AnyFrame, elementId: string): string {
  const short: Record<string, string | undefined> = {
    population: pico.p, intervention: pico.i, comparator: pico.c, outcome: pico.o,
  };
  const v = (pico as Record<string, string | undefined>)[elementId] ?? short[elementId];
  return (v || "").trim();
}

export function PicoCards({ pico, framework }: { pico: AnyFrame; framework?: FrameworkId }) {
  const fw = framework || pico.framework || "pico";
  const elements = frameworkOf(fw).elements;
  const cols = elements.length <= 3 ? "xl:grid-cols-3" : "xl:grid-cols-4";

  return (
    <div className={`grid grid-cols-1 sm:grid-cols-2 ${cols} gap-3`}>
      {elements.map((el) => {
        const style = ELEM_STYLE[el.id] || ELEM_STYLE.population;
        const value = valueFor(pico, el.id);
        return (
          <div key={el.id} className="relative overflow-hidden rounded-xl border bg-card p-4 min-h-[132px] flex flex-col gap-2.5 shadow-sm">
            <span className={`absolute inset-x-0 top-0 h-1 ${style.bar}`} />
            <div className="flex items-center gap-2">
              <span className={`flex items-center justify-center size-6 rounded-md text-xs font-bold ${style.badge}`}>{el.letter}</span>
              <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">{el.label}</span>
            </div>
            <div className={`text-sm leading-relaxed ${value ? "text-foreground" : "text-muted-foreground italic"}`}>
              {value || "None specified"}
            </div>
          </div>
        );
      })}
    </div>
  );
}
