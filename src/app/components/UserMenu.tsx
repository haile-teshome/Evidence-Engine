import { useAuth } from "../lib/auth";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator } from "./ui/dropdown-menu";
import { LogOut } from "lucide-react";
import { toast } from "sonner";

// Account menu for the signed-in user (top-right). Shows who's logged in and a
// log-out action. Accounts are local to this machine.
export function UserMenu() {
  const { user, signOut } = useAuth();
  const current = user || { id: "local", name: "You", email: "" };
  const initial = (current.name || current.email || "?").charAt(0).toUpperCase();

  async function onSignOut() {
    await signOut();
    toast.success("Signed out");
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button className="flex items-center gap-2 rounded-full hover:bg-muted px-2 py-1 transition-colors" title={current.email || "Account"}>
          <div className="size-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-medium text-sm">{initial}</div>
          <span className="text-sm hidden sm:inline">{current.name || current.email || "You"}</span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64">
        <DropdownMenuLabel>
          <div className="font-medium truncate">{current.name || "You"}</div>
          {current.email && <div className="text-xs text-muted-foreground truncate">{current.email}</div>}
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={onSignOut}><LogOut className="size-4 mr-2" />Log out</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function SignedInOnly({ children, fallback }: { children: React.ReactNode; fallback?: React.ReactNode }) {
  const { user } = useAuth();
  if (!user) return <>{fallback}</>;
  return <>{children}</>;
}
