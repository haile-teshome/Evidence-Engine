// Shown in the sidebar when a newer release exists.
//
// "Later" snoozes it rather than silencing it, because a reviewer who was busy
// the one time they were asked should hear about it again. "Update and
// restart" does the whole thing: the app downloads the release itself, checks
// it against the published checksum, and swaps it in on the way back up.
//
// That the app downloads it, rather than the browser, is the point. A file a
// browser downloads is quarantined and meets Gatekeeper's "cannot check this
// for malicious software" dialog on every single update. One the app fetches
// is not, so this path has no dialog, no disk image and no drag.
//
// Reviews, decisions, extracted values and cached text live in
// ~/.evidence-engine, outside the app, so only the app itself is replaced.
import { useEffect, useRef, useState } from "react";
import { ArrowUpCircle, X, Loader2 } from "lucide-react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import {
  getUpdate, dismiss, shouldNotify, startInstall, installProgress, applyAndRestart,
  type UpdateInfo, type UpdateProgress,
} from "../lib/updates";

/** Bytes as the download is actually experienced. */
function mb(n?: number): string {
  return n ? `${Math.round(n / 1048576)} MB` : "";
}

export function UpdateBanner() {
  const [info, setInfo] = useState<UpdateInfo | null>(null);
  const [hidden, setHidden] = useState(false);
  const [progress, setProgress] = useState<UpdateProgress | null>(null);
  const [restarting, setRestarting] = useState(false);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let live = true;
    // Deliberately not retried: if the backend is still starting, the next
    // app load asks again, and the disk cache makes that cheap.
    getUpdate().then((out) => {
      if (live) setInfo(out);
    });
    return () => {
      live = false;
      if (poll.current) clearInterval(poll.current);
    };
  }, []);

  // Once a download is under way the banner follows it, and stops as soon as
  // it has somewhere to stop: ready, or failed.
  useEffect(() => {
    if (!progress || ["ready", "failed", "idle"].includes(progress.state)) return;
    poll.current = setInterval(async () => {
      const next = await installProgress();
      if (next) setProgress(next);
    }, 1000);
    return () => { if (poll.current) clearInterval(poll.current); };
  }, [progress?.state]);

  // Downloaded and unpacked: the only thing left is to go down and come back.
  useEffect(() => {
    if (progress?.state !== "ready" || restarting) return;
    setRestarting(true);
    void applyAndRestart().then((ok) => {
      if (!ok) setProgress({ state: "failed", error: "The update could not be applied." });
    });
  }, [progress?.state, restarting]);

  if (hidden || !shouldNotify(info) || !info) return null;

  const busy = !!progress && ["downloading", "verifying", "staging", "ready"].includes(progress.state);
  const failed = progress?.state === "failed";

  const snooze = () => {
    dismiss(info.latest);
    setHidden(true);
  };

  return (
    <Card className="p-3 mb-3 border-primary/40 bg-primary/5">
      <div className="flex items-start gap-2">
        <ArrowUpCircle className="size-4 mt-0.5 shrink-0 text-primary" />
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium">Update available</div>
          <div className="text-xs text-muted-foreground mt-0.5">
            {info.latest} is out. You have {info.current}.
          </div>

          {busy ? (
            <div className="mt-2 space-y-1.5">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Loader2 className="size-3.5 animate-spin shrink-0" />
                {progress!.state === "downloading"
                  ? `Downloading${progress!.total
                      ? ` ${mb(progress!.received)} of ${mb(progress!.total)}`
                      : "…"}`
                  : progress!.state === "verifying" ? "Checking the download…"
                  : progress!.state === "staging" ? "Unpacking…"
                  : "Restarting…"}
              </div>
              {!!progress!.total && (
                <div className="h-1 w-full overflow-hidden rounded-full bg-primary/15">
                  <div className="h-full bg-primary transition-[width] duration-500"
                       style={{ width: `${Math.min(100, Math.round(((progress!.received || 0) / progress!.total) * 100))}%` }} />
                </div>
              )}
            </div>
          ) : (
            <div className="mt-2 flex gap-2">
              <Button
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={async () => {
                  const out = await startInstall(info.latest).catch(() => null);
                  // No background updater available: the release page always
                  // works, so the reviewer is never stuck with a dead button.
                  if (!out) window.open(info.url, "_blank", "noopener,noreferrer");
                  else setProgress(out);
                }}>
                Update and restart
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className="h-7 px-2 text-xs"
                onClick={snooze}>
                Later
              </Button>
            </div>
          )}

          {/* Only when it failed. There is nothing to say in the ordinary
              case that the two buttons do not already say. */}
          {failed && (
            <div className="text-[11px] text-muted-foreground mt-2">
              That did not work. <a className="underline" href={info.url} target="_blank"
                 rel="noopener noreferrer">Download it here</a> instead.
            </div>
          )}
        </div>
        <button
          aria-label="Dismiss update notice"
          className="shrink-0 text-muted-foreground hover:text-foreground"
          onClick={snooze}>
          <X className="size-3.5" />
        </button>
      </div>
    </Card>
  );
}
