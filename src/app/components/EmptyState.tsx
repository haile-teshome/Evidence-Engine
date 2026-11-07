import { ReactNode } from "react";
import { LucideIcon } from "lucide-react";
import { Button } from "./ui/button";

// A designed empty/placeholder state: an icon, a headline, a short line of
// guidance, and an optional next action. Use instead of a bare alert so
// "nothing here yet" reads as intentional rather than unfinished.
export function EmptyState({
  icon: Icon, title, description, action, extra,
}: {
  icon: LucideIcon;
  title: string;
  description?: ReactNode;
  action?: { label: string; onClick: () => void; icon?: LucideIcon };
  // A second way out, beside the action. Every stage that reads an earlier one
  // offers its own import here, so a reviewer who did the earlier work
  // elsewhere can start from this stage instead of walking back through it.
  extra?: ReactNode;
}) {
  const AIcon = action?.icon;
  return (
    <div className="flex flex-col items-center justify-center text-center rounded-xl border border-dashed bg-muted/20 px-6 py-12">
      <div className="grid place-items-center size-12 rounded-full bg-primary/10 text-primary mb-3">
        <Icon className="size-6" />
      </div>
      <h3 className="font-medium">{title}</h3>
      {description && <p className="text-sm text-muted-foreground mt-1 max-w-md">{description}</p>}
      {(action || extra) && (
        <div className="flex flex-wrap items-center justify-center gap-2 mt-4">
          {action && (
            <Button size="sm" onClick={action.onClick}>
              {AIcon && <AIcon className="size-3.5 mr-1.5" />}{action.label}
            </Button>
          )}
          {extra}
        </div>
      )}
    </div>
  );
}
