import { useEffect, useState } from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { Loader2, X, Sparkles } from "lucide-react";
import { TaskRecord } from "../lib/store";

function fmt(ms: number) {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
}

/**
 * Compact progress card for sequential tasks: shows title, elapsed time,
 * progress bar, current item, and a cancel button. Reads everything from a
 * task in the global store so the UI survives page navigation.
 */
export function TaskProgressCard({
  task,
  title,
  onCancel,
}: {
  task: TaskRecord;
  title: string;
  onCancel?: () => void;
}) {
  const [, setTick] = useState(0);
  useEffect(() => {
    const i = setInterval(() => setTick(t => t + 1), 500);
    return () => clearInterval(i);
  }, []);

  const elapsed = fmt(Date.now() - task.startedAt);
  const pct =
    task.progress && task.progress.total > 0
      ? Math.round((task.progress.done / task.progress.total) * 100)
      : null;

  // Live ETA from throughput so far: elapsed / items done → time per item,
  // projected over the remaining items. Recomputes on the 500ms tick above.
  const done = task.progress?.done ?? 0;
  const total = task.progress?.total ?? 0;
  let eta: string | null = null;
  if (task.status === "running" && total > 0) {
    if (done > 0 && done < total) {
      eta = `~${fmt(((Date.now() - task.startedAt) / done) * (total - done))} left`;
    } else if (done === 0) {
      eta = "estimating…";
    }
  }

  return (
    <Card className="overflow-hidden border-primary/20">
      <div className="px-5 py-4 bg-gradient-to-br from-primary/5 via-card to-card border-b border-border/60">
        <div className="flex items-center gap-3">
          <div>
            {task.status === "running" ? (
              <Loader2 className="size-5 text-primary animate-spin" />
            ) : (
              <Sparkles className="size-5 text-primary" />
            )}
          </div>
          <div className="flex-1">
            <div className="text-sm font-medium">{title}</div>
            <div className="text-xs text-muted-foreground min-h-[1.1em]">
              {task.detail || task.progress?.label || (task.status === "running" ? "Working…" : task.status)}
            </div>
          </div>
          <div className="text-xs tabular-nums text-muted-foreground">{elapsed}</div>
          {onCancel && task.status === "running" && (
            <Button size="sm" variant="ghost" onClick={onCancel}>
              <X className="size-4 mr-1" />Cancel
            </Button>
          )}
        </div>
        {task.progress && task.progress.total > 0 && (
          <div className="mt-3 space-y-1">
            <Progress value={pct ?? 0} />
            <div className="text-xs text-muted-foreground">
              {task.progress.done} / {task.progress.total}{eta ? ` · ${eta}` : ""}
            </div>
          </div>
        )}
      </div>
      {task.log.length > 0 && (
        <div className="max-h-32 overflow-auto text-xs font-mono bg-muted/30 px-3 py-2 space-y-0.5">
          {task.log.slice(-50).map((line, i) => (
            <div key={i} className="text-muted-foreground">{line}</div>
          ))}
        </div>
      )}
    </Card>
  );
}
