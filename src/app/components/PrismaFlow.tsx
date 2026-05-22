import { ArrowDown } from "lucide-react";

type Counts = {
  identified: number;
  source_counts?: Record<string, number>;
  duplicates_removed: number;
  screened: number;
  excluded_total: number;
  exclusion_breakdown: Record<string, number>;
  ft_exclusion_breakdown?: Record<string, number>;
  included_final?: number;
};

export function PrismaFlow({ counts }: { counts: Counts }) {
  const Box = ({ title, n, children }: { title: string; n?: number; children?: React.ReactNode }) => (
    <div className="bg-card border-2 border-primary/30 rounded-lg p-4 text-center shadow-sm">
      <div className="font-medium">{title}</div>
      {n !== undefined && <div className="text-2xl font-bold text-primary mt-1">{n.toLocaleString()}</div>}
      {children}
    </div>
  );

  // Aside is sorted by count (desc) and wraps long labels instead of
  // truncating them. Items are typically short categorical buckets like
  // "Population mismatch · 8" produced by the screener category mappers.
  const Aside = ({ title, items }: { title: string; items: Record<string, number> }) => {
    const entries = Object.entries(items).sort((a, b) => b[1] - a[1]);
    return (
      <div className="bg-muted/50 border rounded-lg p-3 text-sm">
        <div className="font-medium mb-2">{title}</div>
        {entries.length === 0 ? (
          <div className="text-muted-foreground">—</div>
        ) : (
          <ul className="space-y-1.5">
            {entries.map(([k, v]) => (
              <li key={k} className="flex justify-between items-baseline gap-3">
                <span className="leading-snug break-words">{k}</span>
                <span className="font-medium tabular-nums shrink-0">{v}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  };

  // Top-aligned rows so a tall aside doesn't push the central box into the
  // middle of empty space. The arrows are absorbed into the central column's
  // flex layout below; we don't render separate Arrow rows that span the
  // full grid width.
  const Row = ({ box, aside, withArrow }: {
    box: React.ReactNode;
    aside?: React.ReactNode;
    withArrow?: boolean;
  }) => (
    <div className="grid grid-cols-[1fr_28rem_1fr] gap-4 justify-center items-start">
      <div />
      <div className="flex flex-col items-stretch">
        {withArrow && (
          <div className="flex justify-center py-1">
            <ArrowDown className="size-5 text-muted-foreground" />
          </div>
        )}
        {box}
      </div>
      <div>{aside}</div>
    </div>
  );

  return (
    <div className="py-2 space-y-3">
      <Row box={
        <Box title="Records identified from databases" n={counts.identified}>
          {counts.source_counts && (
            <div className="text-xs text-muted-foreground mt-2 space-y-0.5">
              {Object.entries(counts.source_counts).map(([k, v]) => <div key={k}>{k}: {v}</div>)}
            </div>
          )}
        </Box>
      } />
      <Row
        withArrow
        box={<Box title="Records after duplicates removed" n={Math.max(0, counts.identified - counts.duplicates_removed)} />}
        aside={<Aside title="Duplicates removed" items={{ "Duplicate records": counts.duplicates_removed }} />}
      />
      <Row
        withArrow
        box={<Box title="Records screened (Title/Abstract)" n={counts.screened} />}
        aside={<Aside title="Excluded at screening" items={counts.exclusion_breakdown} />}
      />
      <Row
        withArrow
        box={<Box title="Reports assessed for eligibility" n={Math.max(0, counts.screened - counts.excluded_total)} />}
        aside={counts.ft_exclusion_breakdown ? <Aside title="Excluded at full-text" items={counts.ft_exclusion_breakdown} /> : undefined}
      />
      <Row
        withArrow
        box={<Box title="Studies included in review" n={counts.included_final ?? 0} />}
      />
    </div>
  );
}
