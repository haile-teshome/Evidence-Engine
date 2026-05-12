import { Pico } from "../lib/mockServices";

export function PicoCards({ pico }: { pico: Pico | { p?: string; i?: string; c?: string; o?: string } }) {
  const get = (a: any, b: string) => a[b] || a[b.toUpperCase()] || "None specified";
  const map: any = "population" in pico
    ? { Population: pico.population, Intervention: pico.intervention, Comparator: pico.comparator, Outcome: pico.outcome }
    : { Population: (pico as any).p, Intervention: (pico as any).i, Comparator: (pico as any).c, Outcome: (pico as any).o };

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
      {Object.entries(map).map(([label, value]) => (
        <div key={label} className="bg-card rounded-xl p-4 border-l-4 border-primary min-h-[140px] shadow-sm">
          <div className="uppercase tracking-wider text-primary text-xs mb-2">{label}</div>
          <div className="text-sm leading-relaxed text-foreground">{(value as string)?.trim() || "None specified"}</div>
        </div>
      ))}
    </div>
  );
}
