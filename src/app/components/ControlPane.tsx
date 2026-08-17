import { ReactNode } from "react";
import { Card } from "./ui/card";

// Shared "control pane": a Card that groups summary counts (left) with the
// view's action buttons (right), used at the top of results views so every
// stage looks consistent. Compose with <InlineStat> and <PaneDivider>.
export function ControlPane({ stats, actions, className = "", singleLine = true }: {
  stats: ReactNode;
  actions: ReactNode;
  className?: string;
  // Keep stats and actions on one row (scrolls horizontally only if the window is
  // genuinely too narrow) instead of wrapping the actions below the stats. On by
  // default so the top bar stays a single tidy line. Pass singleLine={false} to
  // allow wrapping.
  singleLine?: boolean;
}) {
  return (
    <Card className={`p-0 overflow-hidden shadow-sm ring-1 ring-black/[0.02] ${className}`}>
      {/* Stats left, actions right. When the row can't fit (large counts, long
          labels, many buttons) the actions wrap to their own line and stay
          right-aligned via ml-auto, rather than jumping under the stats. */}
      <div className={`flex items-center gap-x-3 gap-y-3 px-4 py-3 bg-gradient-to-b from-muted/40 to-transparent ${singleLine ? "flex-nowrap" : "flex-wrap"}`}>
        {/* Stats flex and scroll INTERNALLY (min-w-0 + overflow-x-auto) so that
            large counts or many stats never push the action buttons off-screen;
            the actions stay pinned right (shrink-0) and always visible. */}
        <div className={`flex items-center gap-2.5 sm:gap-3.5 ${singleLine ? "flex-nowrap min-w-0 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden" : "flex-wrap"}`}>{stats}</div>
        <div className={`flex items-center gap-1.5 ml-auto ${singleLine ? "flex-nowrap shrink-0" : "flex-wrap justify-end"}`}>{actions}</div>
      </div>
    </Card>
  );
}

// A thin vertical rule to separate groups of stats. Hidden when the row wraps
// on narrow screens. shrink-0 so it keeps its 1px width instead of collapsing
// when the stats row is tight.
export function PaneDivider() {
  return <span className="w-px self-stretch bg-border/60 hidden sm:block shrink-0" aria-hidden="true" />;
}

// Keep stat values narrow no matter how big the corpus gets: 1,000-9,999 get
// thousands separators, and 10,000+ collapse to compact notation (48.4K, 1.2M)
// so the tile can't blow out the control-pane layout. Non-numbers pass through.
function fmtStatValue(v: any): { display: any; full?: string } {
  if (typeof v === "number" && isFinite(v)) {
    const full = v.toLocaleString("en-US");
    if (Math.abs(v) >= 10000) {
      return { display: new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v), full };
    }
    return { display: full, full };
  }
  return { display: v };
}

// Compact inline stat: icon tile + value + label (with an optional hint like a
// percentage), sized to sit alongside the action buttons in a ControlPane.
export function InlineStat({ icon: Icon, value, label, hint, tone = "default" }: {
  icon: any;
  value: any;
  label: string;
  hint?: string;
  tone?: "default" | "success" | "danger" | "neutral" | "amber";
}) {
  const t = {
    default: { text: "text-foreground", chip: "bg-primary/10 text-primary" },
    success: { text: "text-emerald-600 dark:text-emerald-400", chip: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300" },
    danger: { text: "text-rose-600 dark:text-rose-400", chip: "bg-rose-100 text-rose-700 dark:bg-rose-950/50 dark:text-rose-300" },
    neutral: { text: "text-foreground", chip: "bg-muted text-muted-foreground" },
    amber: { text: "text-amber-600 dark:text-amber-400", chip: "bg-amber-100 text-amber-700 dark:bg-amber-950/50 dark:text-amber-300" },
  }[tone];
  return (
    <div className="flex items-center gap-2 shrink-0">
      <div className={`size-8 rounded-lg grid place-items-center shrink-0 ${t.chip}`}>
        <Icon className="size-3.5" />
      </div>
      <div className="leading-tight">
        {(() => { const { display, full } = fmtStatValue(value); return (
          <div className={`text-lg font-bold tabular-nums whitespace-nowrap ${t.text}`} title={full && full !== String(display) ? full : undefined}>{display}</div>
        ); })()}
        <div className="text-[11px] text-muted-foreground whitespace-nowrap">{label}{hint ? <span className="text-muted-foreground/70"> · {hint}</span> : null}</div>
      </div>
    </div>
  );
}
