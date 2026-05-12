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
  const Aside = ({ title, items }: { title: string; items: Record<string, number> }) => (
    <div className="bg-muted/50 border rounded-lg p-3 text-sm">
      <div className="font-medium mb-2">{title}</div>
      {Object.entries(items).length === 0 ? <div className="text-muted-foreground">—</div> :
        Object.entries(items).map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2"><span className="truncate">{k}</span><span className="font-medium shrink-0">{v}</span></div>
        ))
      }
    </div>
  );
  const Arrow = () => (
    <div className="grid grid-cols-[16rem_28rem_16rem] gap-4 justify-center">
      <div />
      <div className="flex justify-center"><ArrowDown className="size-6 text-muted-foreground my-1" /></div>
      <div />
    </div>
  );
  const Row = ({ box, aside }: { box: React.ReactNode; aside?: React.ReactNode }) => (
    <div className="grid grid-cols-[16rem_28rem_16rem] gap-4 justify-center items-center">
      <div />
      <div>{box}</div>
      <div>{aside}</div>
    </div>
  );

  return (
    <div className="py-4">
      <Row box={
        <Box title="Records identified from databases" n={counts.identified}>
          {counts.source_counts && (
            <div className="text-xs text-muted-foreground mt-2 space-y-0.5">
              {Object.entries(counts.source_counts).map(([k, v]) => <div key={k}>{k}: {v}</div>)}
            </div>
          )}
        </Box>
      } />
      <Arrow />
      <Row
        box={<Box title="Records after duplicates removed" n={counts.screened} />}
        aside={<Aside title="Duplicates Removed" items={{ "Duplicate records": counts.duplicates_removed }} />}
      />
      <Arrow />
      <Row
        box={<Box title="Records screened (Title/Abstract)" n={counts.screened} />}
        aside={<Aside title="Excluded at Screening" items={counts.exclusion_breakdown} />}
      />
      <Arrow />
      <Row
        box={<Box title="Reports assessed for eligibility" n={Math.max(0, counts.screened - counts.excluded_total)} />}
        aside={counts.ft_exclusion_breakdown ? <Aside title="Excluded at Full-Text" items={counts.ft_exclusion_breakdown} /> : undefined}
      />
      <Arrow />
      <Row box={<Box title="Studies included in review" n={counts.included_final ?? 0} />} />
    </div>
  );
}
