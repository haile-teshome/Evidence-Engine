import { Button } from "./ui/button";

/** Footer for a chunked list: how much is shown, and two ways to see more. */
export function ShowMore({ shown, total, step, onMore, onAll, className = "" }: {
  shown: number; total: number; step: number; onMore: () => void; onAll: () => void; className?: string;
}) {
  if (shown >= total) return null;
  return (
    <div className={`flex items-center justify-center gap-3 py-3 text-xs text-muted-foreground ${className}`}>
      <span className="tabular-nums">Showing {shown.toLocaleString()} of {total.toLocaleString()}</span>
      <Button type="button" variant="outline" size="sm" className="h-7 text-xs" onClick={onMore}>
        Show {Math.min(step, total - shown).toLocaleString()} more
      </Button>
      <Button type="button" variant="ghost" size="sm" className="h-7 text-xs" onClick={onAll}>Show all</Button>
    </div>
  );
}
