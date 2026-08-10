import { ReactNode } from "react";
import { Card } from "./ui/card";

// Shared "control pane": a Card that groups summary counts (left) with the
// view's action buttons (right), used at the top of results views so every
// stage looks consistent. Compose with <InlineStat> and <PaneDivider>.
export function ControlPane({ stats, actions, className = "", singleLine = false }: {
  stats: ReactNode;
  actions: ReactNode;
  className?: string;
  // Keep stats and actions on one row (scrolls horizontally if the window is
  // too narrow) instead of wrapping the actions below the stats.
  singleLine?: boolean;
}) {
  return (
    <Card className={`p-0 overflow-hidden shadow-sm ring-1 ring-black/[0.02] ${className}`}>
      <div className={`flex items-center justify-between gap-x-6 gap-y-4 p-4 bg-gradient-to-b from-muted/40 to-transparent ${singleLine ? "flex-nowrap overflow-x-auto" : "flex-wrap"}`}>
        <div className={`flex items-stretch gap-4 sm:gap-5 ${singleLine ? "flex-nowrap shrink-0" : "flex-wrap"}`}>{stats}</div>
        <div className={`flex items-center gap-2 justify-end ${singleLine ? "flex-nowrap shrink-0" : "flex-wrap"}`}>{actions}</div>
      </div>
    </Card>
  );
}

// A thin vertical rule to separate groups of stats. Hidden when the row wraps
// on narrow screens.
export function PaneDivider() {
  return <span className="w-px self-stretch bg-border/60 hidden sm:block" aria-hidden="true" />;
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
    <div className="flex items-center gap-2.5">
      <div className={`size-9 rounded-lg grid place-items-center shrink-0 ${t.chip}`}>
        <Icon className="size-4" />
      </div>
      <div className="leading-tight">
        <div className={`text-xl font-bold tabular-nums ${t.text}`}>{value}</div>
        <div className="text-[11px] text-muted-foreground">{label}{hint ? <span className="text-muted-foreground/70"> · {hint}</span> : null}</div>
      </div>
    </div>
  );
}
