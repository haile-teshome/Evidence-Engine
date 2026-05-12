// Token-level diff between two search queries.
// Highlights added tokens (green) and removed tokens (red strike-through).

function tokenize(q: string): string[] {
  // Split on whitespace, keeping the original tokens. Boolean operators and
  // bracketed tags survive because we don't split inside them.
  return q.split(/(\s+)/);
}

function normalize(tok: string) {
  return tok.trim().toLowerCase();
}

export function QueryDiff({ previous, current }: { previous: string; current: string }) {
  const prevTokens = tokenize(previous);
  const curTokens = tokenize(current);
  const prevSet = new Set(prevTokens.map(normalize).filter(Boolean));
  const curSet = new Set(curTokens.map(normalize).filter(Boolean));

  const removed = prevTokens.filter(t => normalize(t) && !curSet.has(normalize(t)));

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
      {removed.filter(t => t.trim()).length > 0 && (
        <div className="flex flex-wrap gap-1 items-baseline">
          <span className="text-muted-foreground">Removed:</span>
          {removed
            .filter(t => t.trim())
            .map((t, i) => (
              <span
                key={i}
                className="bg-rose-50 text-rose-700 line-through rounded px-1"
              >
                {t.trim()}
              </span>
            ))}
        </div>
      )}
    </div>
  );
}
