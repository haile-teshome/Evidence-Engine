import { Pico } from "../lib/mockServices";

// Per-element accent so Population / Intervention / Comparator / Outcome are
// instantly distinguishable.
const PICO_META = [
  { key: "Population",   letter: "P", badge: "bg-blue-100 text-blue-700",     bar: "bg-blue-400" },
  { key: "Intervention", letter: "I", badge: "bg-violet-100 text-violet-700", bar: "bg-violet-400" },
  { key: "Comparator",   letter: "C", badge: "bg-amber-100 text-amber-700",   bar: "bg-amber-400" },
  { key: "Outcome",      letter: "O", badge: "bg-emerald-100 text-emerald-700", bar: "bg-emerald-400" },
] as const;

export function PicoCards({ pico }: { pico: Pico | { p?: string; i?: string; c?: string; o?: string } }) {
  const values: Record<string, string> = "population" in pico
    ? {
        Population: pico.population, Intervention: pico.intervention,
        Comparator: pico.comparator, Outcome: pico.outcome,
      }
    : {
        Population: (pico as any).p || "", Intervention: (pico as any).i || "",
        Comparator: (pico as any).c || "", Outcome: (pico as any).o || "",
      };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
      {PICO_META.map(({ key, letter, badge, bar }) => {
        const value = (values[key] || "").trim();
        return (
          <div key={key} className="relative overflow-hidden rounded-xl border bg-card p-4 min-h-[132px] flex flex-col gap-2.5 shadow-sm">
            <span className={`absolute inset-x-0 top-0 h-1 ${bar}`} />
            <div className="flex items-center gap-2">
              <span className={`flex items-center justify-center size-6 rounded-md text-xs font-bold ${badge}`}>{letter}</span>
              <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">{key}</span>
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
