import { useEffect, useState } from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Check, Loader2, Circle, Sparkles, X } from "lucide-react";

export type StageId = string;

export type Stage = {
  id: StageId;
  label: string;
  status: "pending" | "running" | "done" | "error" | "canceled";
  detail?: string;
};

function formatElapsed(ms: number) {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
}

export function AnalysisProgress({
  stages,
  startedAt,
  onCancel,
  title = "Analyzing your research question",
}: {
  stages: Stage[];
  startedAt: number;
  onCancel?: () => void;
  title?: string;
}) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const i = setInterval(() => setTick(t => t + 1), 500);
    return () => clearInterval(i);
  }, []);

  const running = stages.find(s => s.status === "running");
  const doneCount = stages.filter(s => s.status === "done").length;
  const allDone = doneCount === stages.length && stages.length > 0;
  // Overall progress, not a repeat of the running step (the stage list already
  // names and spins that): how far through the pipeline we are.
  const statusText = allDone
    ? "Analysis complete"
    : running
    ? `${doneCount} of ${stages.length} steps complete`
    : "Preparing analysis…";
  const pct = Math.round((doneCount / stages.length) * 100);
  const elapsed = formatElapsed(Date.now() - startedAt);

  return (
    <Card className="overflow-hidden border-primary/20">
      {/* Top bar */}
      <div className="px-5 py-4 bg-gradient-to-br from-primary/5 via-card to-card border-b border-border/60">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Sparkles className="size-5 text-primary" />
            <span className="absolute inset-0 animate-ping opacity-40">
              <Sparkles className="size-5 text-primary" />
            </span>
          </div>
          <div className="flex-1">
            <div className="text-sm font-medium">{title}</div>
            <div className="text-xs text-muted-foreground min-h-[1.1em] truncate">{statusText}</div>
          </div>
          <div className="text-xs tabular-nums text-muted-foreground">{elapsed}</div>
          {onCancel && (
            <Button size="sm" variant="ghost" onClick={onCancel}>
              <X className="size-4 mr-1" />Cancel
            </Button>
          )}
        </div>
        {/* Progress bar */}
        <div className="mt-3 h-1.5 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full bg-primary transition-all duration-500 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Stage list */}
      <div className="p-4 space-y-2">
        {stages.map(stage => (
          <div key={stage.id} className="flex items-center gap-3 text-sm" data-tick={tick}>
            <span className="shrink-0 size-5 flex items-center justify-center">
              {stage.status === "done" && <Check className="size-4 text-primary" />}
              {stage.status === "running" && <Loader2 className="size-4 text-primary animate-spin" />}
              {stage.status === "pending" && <Circle className="size-3 text-muted-foreground/40" />}
              {stage.status === "error" && <span className="size-2 rounded-full bg-destructive" />}
              {stage.status === "canceled" && <X className="size-3 text-muted-foreground" />}
            </span>
            <span
              className={
                stage.status === "done"
                  ? "text-foreground"
                  : stage.status === "running"
                  ? "text-foreground font-medium"
                  : stage.status === "error"
                  ? "text-destructive"
                  : "text-muted-foreground"
              }
            >
              {stage.label}
            </span>
            {stage.detail && (
              <span className="text-xs text-muted-foreground truncate">{stage.detail}</span>
            )}
          </div>
        ))}
      </div>

      {/* Tip */}
      <div className="px-5 py-3 border-t bg-muted/30 text-xs text-muted-foreground">
        Tip: Local models can take 30-60s per step on first run while the model loads into memory. Subsequent steps are faster.
      </div>
    </Card>
  );
}
