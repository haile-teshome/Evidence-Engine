import { useEffect, useReducer, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "./ui/dialog";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { KeyRound, Eye, EyeOff, ShieldCheck, Lock, Check, Trash2, Loader2 } from "lucide-react";
import { toast } from "sonner";
import {
  type LlmProvider, LLM_PROVIDERS,
  getMode, setMode, subscribe, refreshKeychainStatus,
  keychainIsAvailable, keychainStatus, keychainSet, keychainDelete,
  hasEncrypted, isUnlocked, unlock, lock, saveEncrypted, clearEncrypted, unlockedKeys,
} from "../lib/keystore";

const META: Record<LlmProvider, { label: string; placeholder: string }> = {
  anthropic: { label: "Anthropic (Claude)", placeholder: "sk-ant-..." },
  openai: { label: "OpenAI (GPT)", placeholder: "sk-..." },
  google: { label: "Google (Gemini)", placeholder: "AIza..." },
};
const ORDER: LlmProvider[] = ["anthropic", "openai", "google"];

function useKeystore() {
  const [, force] = useReducer(x => x + 1, 0);
  useEffect(() => subscribe(force), []);
  return force;
}

export function ApiKeysDialog({ open, onOpenChange, highlight }: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  onSaved?: () => void;
  highlight?: LlmProvider | null;
}) {
  useKeystore();
  const mode = getMode();
  const kcAvailable = keychainIsAvailable();
  const kcStatus = keychainStatus();

  const [values, setValues] = useState<Record<LlmProvider, string>>({ openai: "", anthropic: "", google: "" });
  const [shown, setShown] = useState<Record<LlmProvider, boolean>>({ openai: false, anthropic: false, google: false });
  const [pass, setPass] = useState("");
  const [pass2, setPass2] = useState("");
  const [unlockPass, setUnlockPass] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    refreshKeychainStatus();
    const uk = unlockedKeys();
    setValues(uk ? { openai: "", anthropic: "", google: "", ...uk } : { openai: "", anthropic: "", google: "" });
    setPass(""); setPass2(""); setUnlockPass("");
  }, [open]);

  const encLocked = mode === "encrypted" && hasEncrypted() && !isUnlocked();

  const doKeychainSave = async (p: LlmProvider) => {
    if (!values[p]?.trim()) return;
    setBusy(true);
    try { await keychainSet(p, values[p].trim()); setValues(v => ({ ...v, [p]: "" })); toast.success(`${META[p].label} key saved to keychain`); }
    catch (e: any) { toast.error(e?.message || "Could not save to keychain"); }
    finally { setBusy(false); }
  };
  const doKeychainRemove = async (p: LlmProvider) => {
    setBusy(true);
    try { await keychainDelete(p); toast.success(`${META[p].label} key removed`); }
    finally { setBusy(false); }
  };

  const doUnlock = async () => {
    setBusy(true);
    try { await unlock(unlockPass); const uk = unlockedKeys(); setValues({ openai: "", anthropic: "", google: "", ...(uk || {}) }); toast.success("Keys unlocked for this session"); }
    catch (e: any) { toast.error(e?.message || "Incorrect passphrase"); }
    finally { setBusy(false); }
  };
  const doEncryptedSave = async () => {
    if (!pass) { toast.error("Enter a passphrase"); return; }
    if (!hasEncrypted() && pass !== pass2) { toast.error("Passphrases do not match"); return; }
    if (pass.length < 6) { toast.error("Use a passphrase of at least 6 characters"); return; }
    setBusy(true);
    try { await saveEncrypted(values, pass); toast.success("Keys encrypted and saved on this device"); onOpenChange(false); }
    catch (e: any) { toast.error(e?.message || "Could not save"); }
    finally { setBusy(false); }
  };

  const KeyField = ({ p, showRemove, onSave, onRemove, saved }: {
    p: LlmProvider; showRemove?: boolean; onSave?: () => void; onRemove?: () => void; saved?: boolean;
  }) => (
    <div className={`space-y-1.5 rounded-md p-2 -mx-2 ${highlight === p ? "bg-primary/5 ring-1 ring-primary/20" : ""}`}>
      <div className="flex items-center justify-between">
        <Label htmlFor={`key-${p}`} className="text-sm">{META[p].label}</Label>
        {saved && <span className="inline-flex items-center gap-1 text-[11px] text-emerald-600 dark:text-emerald-400"><Check className="size-3" />Saved</span>}
      </div>
      <div className="flex items-center gap-1.5">
        <div className="relative flex-1">
          <Input id={`key-${p}`} type={shown[p] ? "text" : "password"} autoComplete="off" spellCheck={false}
            placeholder={saved ? "•••••••• (replace)" : META[p].placeholder}
            value={values[p]} onChange={e => setValues(v => ({ ...v, [p]: e.target.value }))}
            className="pr-9 font-mono text-xs" />
          <button type="button" onClick={() => setShown(s => ({ ...s, [p]: !s[p] }))}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground" title={shown[p] ? "Hide" : "Show"}>
            {shown[p] ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
          </button>
        </div>
        {onSave && <Button size="sm" variant="outline" className="h-8 shrink-0" disabled={busy || !values[p]?.trim()} onClick={onSave}>Save</Button>}
        {showRemove && onRemove && <Button size="sm" variant="ghost" className="h-8 px-2 shrink-0 text-muted-foreground" disabled={busy} onClick={onRemove} title="Remove"><Trash2 className="size-4" /></Button>}
      </div>
    </div>
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2"><KeyRound className="size-4" />Cloud model API keys</DialogTitle>
          <DialogDescription>Add a key for any provider whose models you want to use. Local models (Ollama, LEADS) need no key.</DialogDescription>
        </DialogHeader>

        {/* Storage mode */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Store keys in:</span>
          <div className="inline-flex rounded-md border p-0.5 text-xs">
            {kcAvailable && (
              <button type="button" onClick={() => setMode("keychain")}
                className={`px-2.5 h-7 rounded font-medium transition-colors ${mode === "keychain" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}>
                Device keychain
              </button>
            )}
            <button type="button" onClick={() => setMode("encrypted")}
              className={`px-2.5 h-7 rounded font-medium transition-colors ${mode === "encrypted" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}>
              Encrypted (passphrase)
            </button>
          </div>
        </div>

        {/* Keychain mode */}
        {mode === "keychain" && (
          <div className="space-y-3">
            {ORDER.map(p => (
              <KeyField key={p} p={p} saved={kcStatus[p]} showRemove={kcStatus[p]}
                onSave={() => doKeychainSave(p)} onRemove={() => doKeychainRemove(p)} />
            ))}
          </div>
        )}

        {/* Encrypted mode — locked (needs passphrase) */}
        {mode === "encrypted" && encLocked && (
          <div className="space-y-3">
            <div className="flex items-start gap-2 rounded-md border bg-muted/40 p-2.5 text-xs text-muted-foreground">
              <Lock className="size-4 shrink-0 mt-0.5" />
              <span>Your keys are encrypted on this device. Enter your passphrase to unlock them for this session.</span>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="unlock-pass" className="text-sm">Passphrase</Label>
              <Input id="unlock-pass" type="password" value={unlockPass} onChange={e => setUnlockPass(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter") doUnlock(); }} placeholder="Enter passphrase" />
            </div>
            <div className="flex justify-between">
              <Button variant="ghost" className="text-muted-foreground" disabled={busy} onClick={() => { clearEncrypted(); toast.success("Encrypted keys cleared"); }}>Forget keys</Button>
              <Button disabled={busy || !unlockPass} onClick={doUnlock}>{busy ? <Loader2 className="size-4 mr-2 animate-spin" /> : null}Unlock</Button>
            </div>
          </div>
        )}

        {/* Encrypted mode — unlocked or first-time setup */}
        {mode === "encrypted" && !encLocked && (
          <div className="space-y-3">
            {ORDER.map(p => <KeyField key={p} p={p} />)}
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1.5">
                <Label htmlFor="enc-pass" className="text-sm">Passphrase</Label>
                <Input id="enc-pass" type="password" value={pass} onChange={e => setPass(e.target.value)} placeholder={hasEncrypted() ? "Passphrase" : "Choose a passphrase"} />
              </div>
              {!hasEncrypted() && (
                <div className="space-y-1.5">
                  <Label htmlFor="enc-pass2" className="text-sm">Confirm</Label>
                  <Input id="enc-pass2" type="password" value={pass2} onChange={e => setPass2(e.target.value)} placeholder="Repeat passphrase" />
                </div>
              )}
            </div>
            <div className="flex justify-between">
              {isUnlocked() ? <Button variant="ghost" className="text-muted-foreground" onClick={lock}>Lock</Button> : <span />}
              <Button disabled={busy} onClick={doEncryptedSave}>{busy ? <Loader2 className="size-4 mr-2 animate-spin" /> : null}Encrypt &amp; save</Button>
            </div>
          </div>
        )}

        <div className="flex items-start gap-2 rounded-md bg-muted/50 border p-2.5 text-[11px] text-muted-foreground">
          <ShieldCheck className="size-4 shrink-0 mt-0.5 text-emerald-600 dark:text-emerald-400" />
          <span>
            {mode === "keychain"
              ? "Keys are stored in your operating system's keychain through the local app and read directly when a model runs. They are never sent to us or saved to the project."
              : "Keys are encrypted on this device with your passphrase (AES-GCM) and stored in this browser only. You unlock them once per session; they are never sent to us or saved to the project."}
          </span>
        </div>
      </DialogContent>
    </Dialog>
  );
}
