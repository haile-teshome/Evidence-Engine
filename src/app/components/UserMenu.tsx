import { useState } from "react";
import { useAuth, OAuthProvider } from "../lib/auth";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "./ui/dialog";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator } from "./ui/dropdown-menu";
import { LogIn, LogOut, Github } from "lucide-react";
import { toast } from "sonner";

function GoogleIcon() {
  return (
    <svg className="size-4" viewBox="0 0 48 48" aria-hidden>
      <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3c-1.6 4.5-5.9 8-11.3 8-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.9 1.2 8 3l5.7-5.7C34 6.1 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.2-.1-2.3-.4-3.5z"/>
      <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.7 16 19 13 24 13c3.1 0 5.9 1.2 8 3l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.4 6.3 14.7z"/>
      <path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 35 26.7 36 24 36c-5.4 0-9.7-3.5-11.3-8l-6.5 5C9.6 39.6 16.2 44 24 44z"/>
      <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.2-2.2 4.1-4 5.6l6.2 5.2C41 35 44 30 44 24c0-1.2-.1-2.3-.4-3.5z"/>
    </svg>
  );
}
function MicrosoftIcon() {
  return (
    <svg className="size-4" viewBox="0 0 23 23" aria-hidden>
      <path fill="#f3f3f3" d="M0 0h23v23H0z"/>
      <path fill="#f35325" d="M1 1h10v10H1z"/>
      <path fill="#81bc06" d="M12 1h10v10H12z"/>
      <path fill="#05a6f0" d="M1 12h10v10H1z"/>
      <path fill="#ffba08" d="M12 12h10v10H12z"/>
    </svg>
  );
}

export function UserMenu() {
  const { user, signIn, signUp, signInWithProvider, signOut } = useAuth();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");

  async function onSignIn(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    try { await signIn(email, password); setOpen(false); }
    catch (err: any) { toast.error(err.message || "Sign-in failed"); }
    finally { setBusy(false); }
  }
  async function onSignUp(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    try { await signUp(email, password, name); toast.success("Account created"); setOpen(false); }
    catch (err: any) { toast.error(err.message || "Sign-up failed"); }
    finally { setBusy(false); }
  }
  async function onSSO(provider: OAuthProvider) {
    setBusy(true);
    try { await signInWithProvider(provider); }
    catch (err: any) { toast.error(err.message || "SSO failed"); setBusy(false); }
  }

  if (user) {
    const initial = (user.name || user.email || "?").charAt(0).toUpperCase();
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className="flex items-center gap-2 rounded-full hover:bg-muted px-2 py-1 transition-colors">
            <div className="size-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-medium text-sm">{initial}</div>
            <span className="text-sm hidden sm:inline">{user.name || user.email}</span>
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-56">
          <DropdownMenuLabel>
            <div className="font-medium">{user.name || "Signed in"}</div>
            <div className="text-xs text-muted-foreground truncate">{user.email}</div>
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={() => signOut()}><LogOut className="size-4 mr-2" />Sign out</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    );
  }

  return (
    <>
      <Button size="sm" onClick={() => setOpen(true)}><LogIn className="size-4 mr-2" />Sign in</Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Sign in</DialogTitle>
            <DialogDescription>Sign in to save your research sessions across devices.</DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Button variant="outline" className="w-full" disabled={busy} onClick={() => onSSO("google")}>
              <GoogleIcon /><span className="ml-2">Continue with Google</span>
            </Button>
            <Button variant="outline" className="w-full" disabled={busy} onClick={() => onSSO("azure")}>
              <MicrosoftIcon /><span className="ml-2">Continue with Microsoft</span>
            </Button>
            <Button variant="outline" className="w-full" disabled={busy} onClick={() => onSSO("github")}>
              <Github className="size-4" /><span className="ml-2">Continue with GitHub</span>
            </Button>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <div className="h-px bg-border flex-1" /> or with email <div className="h-px bg-border flex-1" />
          </div>
          <Tabs defaultValue="signin">
            <TabsList className="grid grid-cols-2 w-full">
              <TabsTrigger value="signin">Sign in</TabsTrigger>
              <TabsTrigger value="signup">Create account</TabsTrigger>
            </TabsList>
            <TabsContent value="signin">
              <form onSubmit={onSignIn} className="space-y-3 pt-3">
                <div><Label>Email</Label><Input type="email" required value={email} onChange={e => setEmail(e.target.value)} /></div>
                <div><Label>Password</Label><Input type="password" required value={password} onChange={e => setPassword(e.target.value)} /></div>
                <Button className="w-full" disabled={busy} type="submit">{busy ? "Signing in..." : "Sign in"}</Button>
              </form>
            </TabsContent>
            <TabsContent value="signup">
              <form onSubmit={onSignUp} className="space-y-3 pt-3">
                <div><Label>Name</Label><Input value={name} onChange={e => setName(e.target.value)} placeholder="Your name" /></div>
                <div><Label>Email</Label><Input type="email" required value={email} onChange={e => setEmail(e.target.value)} /></div>
                <div><Label>Password</Label><Input type="password" required minLength={6} value={password} onChange={e => setPassword(e.target.value)} /></div>
                <Button className="w-full" disabled={busy} type="submit">{busy ? "Creating..." : "Create account"}</Button>
              </form>
            </TabsContent>
          </Tabs>
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
