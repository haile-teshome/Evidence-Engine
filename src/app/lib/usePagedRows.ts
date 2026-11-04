import { useEffect, useState } from "react";

/** Render a long list in chunks. The tables that show a screening run were mapping
 *  every row (1,400 results, 3,900 records), so each tab open built thousands of
 *  DOM rows and every store update rebuilt them.
 *
 *  The first commit is kept small (`initial` rows) so the page's entrance fade is
 *  never fighting a large DOM insertion; the list then grows to `step` rows on its
 *  own once the browser is idle, and from there in steps on request.
 *
 *  `rows` must be referentially stable across renders (memoised): a new array
 *  resets the view to the first page, which is what a changed filter or a new
 *  run should do. */
export function usePagedRows<T>(rows: T[], step = 100, initial = 25) {
  const [limit, setLimit] = useState(initial);
  useEffect(() => {
    setLimit(initial);
    const w = typeof window !== "undefined" ? (window as any) : null;
    const grow = () => setLimit(l => Math.max(l, step));
    const idle = w && typeof w.requestIdleCallback === "function";
    const id: number = idle ? w.requestIdleCallback(grow, { timeout: 1500 }) : setTimeout(grow, 600) as unknown as number;
    return () => { if (idle) w.cancelIdleCallback(id); else clearTimeout(id); };
  }, [rows, step, initial]);
  const total = rows.length;
  const visible = total > limit ? rows.slice(0, limit) : rows;
  return {
    visible,
    total,
    shown: visible.length,
    step,
    hasMore: total > limit,
    showMore: () => setLimit(l => Math.min(total, l + step)),
    showAll: () => setLimit(total),
  };
}
