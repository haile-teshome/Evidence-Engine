// Shown in the sidebar only when a newer release exists. Dismissal is
// remembered per version, so this notifies once per release rather than once
// per launch. Download opens the release page; the app never replaces itself.
import { useEffect, useState } from "react";
import { ArrowUpCircle, X } from "lucide-react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { getUpdate, dismiss, shouldNotify, type UpdateInfo } from "../lib/updates";

export function UpdateBanner() {
  const [info, setInfo] = useState<UpdateInfo | null>(null);
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    let live = true;
    // Deliberately not retried: if the backend is still starting, the next
    // app load asks again, and the disk cache makes that cheap.
    getUpdate().then((out) => {
      if (live) setInfo(out);
    });
    return () => {
      live = false;
    };
  }, []);

  if (hidden || !shouldNotify(info) || !info) return null;

  return (
    <Card className="p-3 mb-3 border-primary/40 bg-primary/5">
      <div className="flex items-start gap-2">
        <ArrowUpCircle className="size-4 mt-0.5 shrink-0 text-primary" />
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium">Update available</div>
          <div className="text-xs text-muted-foreground mt-0.5">
            {info.latest} is out. You have {info.current}.
          </div>
          <div className="mt-2 flex gap-2">
            <Button
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={() => window.open(info.url, "_blank", "noopener,noreferrer")}>
              Download
            </Button>
            <Button
              size="sm"
              variant="ghost"
              className="h-7 px-2 text-xs"
              onClick={() => {
                dismiss(info.latest);
                setHidden(true);
              }}>
              Later
            </Button>
          </div>
          <div className="text-[11px] text-muted-foreground mt-2">
            Your reviews are stored outside the app, so replacing it keeps everything.
          </div>
        </div>
        <button
          aria-label="Dismiss update notice"
          className="shrink-0 text-muted-foreground hover:text-foreground"
          onClick={() => {
            dismiss(info.latest);
            setHidden(true);
          }}>
          <X className="size-3.5" />
        </button>
      </div>
    </Card>
  );
}
