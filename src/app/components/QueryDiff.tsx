// Token-level diff between two search queries.
// Highlights added tokens (green) inline, and lists Added / Removed terms below.

function tokenize(q: string): string[] {
  // Split on whitespace, keeping the original tokens. Boolean operators and
  // bracketed tags survive because we don't split inside them.
  return q.split(/(\s+)/);
}

function normalize(tok: string) {
  return tok.trim().toLowerCase();
}

// Dedupe tokens by normalized form, preserving the first original spelling.
function uniqueTerms(tokens: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const t of tokens) {
    const k = normalize(t);
    if (k && !seen.has(k)) { seen.add(k); out.push(t.trim()); }
  }
  return out;
}

export function QueryDiff({ previous, current }: { previous: string; current: string }) {
  const prevTokens = tokenize(previous);
  const curTokens = tokenize(current);
  const prevSet = new Set(prevTokens.map(normalize).filter(Boolean));
  const curSet = new Set(curTokens.map(normalize).filter(Boolean));

  // Only treat tokens as "added" when there's a previous query to diff against,
  // otherwise the whole (initial) query would light up as new.
  const added = previous ? uniqueTerms(curTokens.filter(t => normalize(t) && !prevSet.has(normalize(t)))) : [];
  const removed = uniqueTerms(prevTokens.filter(t => normalize(t) && !curSet.has(normalize(t))));

  return (
    <div className="space-y-2 text-xs font-mono">
      <pre className="bg-muted rounded p-2 whitespace-pre-wrap break-words leading-relaxed">
        {curTokens.map((tok, i) => {
          const key = normalize(tok);
          if (!key) return <span key={i}>{tok}</span>;
          const isNew = previous && !prevSet.has(key);
          return (
            <span
              key={i}
              className={isNew ? "bg-emerald-100 text-emerald-900 rounded px-0.5" : ""}
            >
              {tok}
            </span>
          );
        })}
      </pre>

      {(added.length > 0 || removed.length > 0) && (
        <div className="space-y-1">
          {added.length > 0 && (
            <div className="flex flex-wrap gap-1 items-baseline">
              <span className="text-muted-foreground w-16 shrink-0">Added:</span>
              {added.map((t, i) => (
                <span key={i} className="bg-emerald-50 text-emerald-700 rounded px-1">{t}</span>
              ))}
            </div>
          )}
          {removed.length > 0 && (
            <div className="flex flex-wrap gap-1 items-baseline">
              <span className="text-muted-foreground w-16 shrink-0">Removed:</span>
              {removed.map((t, i) => (
                <span key={i} className="bg-rose-50 text-rose-700 line-through rounded px-1">{t}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
