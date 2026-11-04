import { startTransition, useEffect, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Skeleton } from "./ui/skeleton";

/** The grey placeholder a page shows while its content is on the way. Shaped like
 *  the pages it stands in for (title row, toolbar, table rows) so the hand-off to
 *  real content is a crossfade between two similar layouts, not a jump. */
export function PageSkeleton() {
  return (
    <div className="space-y-4" role="status" aria-label="Loading">
      <div className="flex items-center justify-between gap-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-8 w-40" />
      </div>
      <Skeleton className="h-24 w-full rounded-xl" />
      <div className="space-y-2">
        {Array.from({ length: 8 }, (_, i) => <Skeleton key={i} className="h-10 w-full" />)}
      </div>
    </div>
  );
}

/** How long to leave the skeleton alone before building the page. The page wrapper
 *  in App fades in over 120 ms; starting the heavy render after that fade has
 *  finished means no animation is running while React commits the new DOM. */
const SETTLE_MS = 180;

/** Smoothness here comes from sequencing, not speed: a frame only drops when a
 *  big DOM commit lands while something is animating. So the skeleton appears
 *  with no animation of its own (the wrapper fades it in), the page is rendered
 *  only after that fade has settled, inside a transition so input stays live, and
 *  the crossfade to content starts only once the commit is done. Opacity is
 *  compositor work, so the fades themselves stay smooth. Give it a key per tab. */
export function DeferredPage({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const id = setTimeout(() => startTransition(() => setReady(true)), SETTLE_MS);
    return () => clearTimeout(id);
  }, []);
  return (
    // The min-height holds the page's footprint while nothing is ready, so the
    // scrollbar does not vanish and reappear across the switch.
    <div className={`relative ${ready ? "" : "min-h-[60vh]"}`}>
      <AnimatePresence initial={false}>
        {!ready && (
          <motion.div
            key="skeleton"
            className="absolute inset-x-0 top-0"
            exit={{ opacity: 0, transition: { duration: 0.18, ease: "easeOut" } }}
          >
            <PageSkeleton />
          </motion.div>
        )}
      </AnimatePresence>
      {ready && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.22, ease: "easeOut" }}>
          {children}
        </motion.div>
      )}
    </div>
  );
}
