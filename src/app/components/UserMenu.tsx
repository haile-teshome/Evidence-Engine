import { useState } from "react";
import { useAuth } from "../lib/auth";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "./ui/dialog";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator } from "./ui/dropdown-menu";
import { UserPlus, Check, RotateCcw } from "lucide-react";
import { toast } from "sonner";

// Local reviewer-profile switcher. No accounts or passwords — profiles exist so
// multiple people screening on this machine (or LAN) get their own decisions in
// dual-review projects. Everything is stored locally by the backend.
export function UserMenu() {
  const { user, reviewers, addReviewer, selectReviewer, signOut } = useAuth();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");

  async function onCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) { toast.error("Enter a name for the reviewer"); return; }
    setBusy(true);
    try {
      await addReviewer(name.trim(), email.trim());
      toast.success(`Reviewer "${name.trim()}" added`);
      setName(""); setEmail(""); setOpen(false);
    } catch (err: any) {
      toast.error(err.message || "Could not add reviewer");
    } finally {
      setBusy(false);
    }
  }

  const current = user || { id: "local", name: "You", email: "" };
  const initial = (current.name || current.email || "?").charAt(0).toUpperCase();

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className="flex items-center gap-2 rounded-full hover:bg-muted px-2 py-1 transition-colors" title="Switch reviewer profile">
            <div className="size-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-medium text-sm">{initial}</div>
            <span className="text-sm hidden sm:inline">{current.name || current.email || "You"}</span>
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-64">
          <DropdownMenuLabel>
            <div className="font-medium">Reviewer profiles</div>
            <div className="text-xs text-muted-foreground">Local to this computer</div>
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          {reviewers.map(r => (
            <DropdownMenuItem key={r.id} onClick={() => selectReviewer(r.id)}>
              <div className="size-5 rounded-full bg-muted text-foreground/70 flex items-center justify-center text-[10px] mr-2">
                {(r.name || r.email || "?").charAt(0).toUpperCase()}
              </div>
              <span className="flex-1 truncate">{r.name || r.email || r.id}</span>
              {r.id === current.id && <Check className="size-4 text-primary" />}
            </DropdownMenuItem>
          ))}
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={() => setOpen(true)}><UserPlus className="size-4 mr-2" />Add reviewer…</DropdownMenuItem>
          {current.id !== "local" && (
            <DropdownMenuItem onClick={() => signOut()}><RotateCcw className="size-4 mr-2" />Switch to default</DropdownMenuItem>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Add a reviewer</DialogTitle>
            <DialogDescription>
              Create a local profile for another reviewer on this machine. Each reviewer's
              screening decisions are recorded separately for dual-review and conflict adjudication.
              No account or password needed.
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={onCreate} className="space-y-3 pt-1">
            <div><Label>Name</Label><Input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Jane Doe" autoFocus /></div>
            <div><Label>Email <span className="text-muted-foreground">(optional)</span></Label><Input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="jane@example.org" /></div>
            <Button className="w-full" disabled={busy} type="submit">{busy ? "Adding…" : "Add reviewer"}</Button>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}

export function SignedInOnly({ children, fallback }: { children: React.ReactNode; fallback?: React.ReactNode }) {
  const { user } = useAuth();
  if (!user) return <>{fallback}</>;
  return <>{children}</>;
}
