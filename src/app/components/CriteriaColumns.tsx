import { frameworkOf, FrameworkId } from "../lib/frameworks";

// The rules a stage is about to judge every article by: the review question on
// the left, then include and exclude side by side and colour-coded.
//
// Shared by Abstract Screening and Full-Text Screening because it answers the
// same question at both stages, and because two copies drift. These are far
// more worth checking before a run than a list of titles: a wrong criterion
// mis-sorts the whole corpus, while a title tells the reader nothing they
// cannot see afterwards.
export function CriteriaColumns({
  framework, pico, inclusion, exclusion, emptyHint,
}: {
  framework: FrameworkId;
  /** Loose on purpose: callers hold it as a plain record keyed by element id. */
  pico: Record<string, string>;
  inclusion: string[];
  exclusion: string[];
  /** What to say when nothing is defined. Each stage words this for itself. */
  emptyHint?: string;
}) {
  const elements = frameworkOf(framework).elements
    .map(e => ({ label: e.label, value: (pico?.[e.id] || "").trim() }))
    .filter(e => e.value);

  return (
    <div className="grid border-t border-border md:grid-cols-3 md:divide-x divide-border">
      <div className="min-w-0 p-4">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Review question</div>
        {elements.length === 0 && inclusion.length === 0 && exclusion.length === 0 ? (
          // Worth saying plainly: with nothing to judge against, every model
          // call is guesswork and the run is wasted.
          <p className="mt-1.5 text-xs text-amber-600">
            {emptyHint || "No criteria set. Define the question on the Home page first."}
          </p>
        ) : (
          <dl className="mt-2.5 space-y-2.5">
            {elements.map(e => (
              <div key={e.label} className="min-w-0">
                <dt className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  {e.label}
                </dt>
                <dd className="mt-0.5 text-xs leading-snug text-foreground">{e.value}</dd>
              </div>
            ))}
          </dl>
        )}
      </div>

      <div className="min-w-0 bg-emerald-50 p-4 dark:bg-emerald-950/30">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-emerald-800 dark:text-emerald-300">Inclusion criteria</div>
        {inclusion.length === 0 ? (
          <p className="mt-2.5 text-xs text-muted-foreground">None set.</p>
        ) : (
          <ul className="mt-2.5 space-y-2">
            {inclusion.map((c, i) => (
              <li key={i} className="flex gap-2 text-xs leading-snug">
                <span aria-hidden className="mt-[5px] size-1.5 shrink-0 rounded-full bg-emerald-600" />
                <span className="min-w-0">{c}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="min-w-0 bg-rose-50 p-4 dark:bg-rose-950/30">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-rose-800 dark:text-rose-300">Exclusion criteria</div>
        {exclusion.length === 0 ? (
          <p className="mt-2.5 text-xs text-muted-foreground">None set.</p>
        ) : (
          <ul className="mt-2.5 space-y-2">
            {exclusion.map((c, i) => (
              <li key={i} className="flex gap-2 text-xs leading-snug">
                <span aria-hidden className="mt-[5px] size-1.5 shrink-0 rounded-full bg-rose-600" />
                <span className="min-w-0">{c}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
