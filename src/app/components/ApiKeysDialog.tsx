import { useEffect, useReducer, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "./ui/dialog";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { KeyRound, Eye, EyeOff, ShieldCheck, Lock, Check, Trash2, Loader2, Database } from "lucide-react";
import { toast } from "sonner";
import {
  type LlmProvider, LLM_PROVIDERS,
  getMode, setMode, subscribe, refreshKeychainStatus,
  keychainIsAvailable, keychainStatus, keychainSet, keychainDelete,
  hasEncrypted, isUnlocked, unlock, lock, saveEncrypted, clearEncrypted, unlockedKeys,
} from "../lib/keystore";
import { getDbKey, setDbKey, hasDbKey, getContactEmail, setContactEmail, isValidContactEmail, type DbSource } from "../lib/dbKeys";

const META: Record<LlmProvider, { label: string; placeholder: string }> = {
  anthropic: { label: "Anthropic (Claude)", placeholder: "sk-ant-..." },
  openai: { label: "OpenAI (GPT)", placeholder: "sk-..." },
  google: { label: "Google (Gemini)", placeholder: "AIza..." },
};
const ORDER: LlmProvider[] = ["anthropic", "openai", "google"];

// Data-source keys the UI offers. These correspond to sources whose backend
// service reads get_cred(): CORE and Semantic Scholar today (Scopus once a Scopus
// source is added, since the header + store already support it).
const DB_ITEMS: { source: DbSource; label: string; placeholder: string; help: string; tier: "free" | "subscription" }[] = [
  { source: "core", label: "CORE", placeholder: "CORE API key", help: "Free: core.ac.uk/services/api", tier: "free" },
  { source: "semantic_scholar", label: "Semantic Scholar", placeholder: "Semantic Scholar API key", help: "Free: semanticscholar.org/product/api", tier: "free" },
  { source: "ncbi", label: "PubMed (NCBI)", placeholder: "NCBI API key", help: "Free, raises PubMed rate limit: ncbi.nlm.nih.gov/account", tier: "free" },
  { source: "springer", label: "Springer Nature", placeholder: "Springer API key", help: "Free: dev.springernature.com", tier: "free" },
  { source: "ieee", label: "IEEE Xplore", placeholder: "IEEE API key", help: "Free: developer.ieee.org", tier: "free" },
  { source: "scopus", label: "Scopus (Elsevier)", placeholder: "Elsevier API key", help: "Institutional: dev.elsevier.com", tier: "subscription" },
  { source: "wos", label: "Web of Science", placeholder: "Web of Science key", help: "Institutional: developer.clarivate.com", tier: "subscription" },
];

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
  const force = useKeystore();
  const mode = getMode();
  const kcAvailable = keychainIsAvailable();
  const kcStatus = keychainStatus();

  const [values, setValues] = useState<Record<LlmProvider, string>>({ openai: "", anthropic: "", google: "" });
  const [shown, setShown] = useState<Record<LlmProvider, boolean>>({ openai: false, anthropic: false, google: false });
  const [dbValues, setDbValues] = useState<Record<string, string>>({ core: "", semantic_scholar: "" });
  const [dbShown, setDbShown] = useState<Record<string, boolean>>({});
  const [tab, setTab] = useState<"models" | "databases">("models");
  const [pass, setPass] = useState("");
  const [pass2, setPass2] = useState("");
  const [unlockPass, setUnlockPass] = useState("");
  const [busy, setBusy] = useState(false);
  const [email, setEmail] = useState(getContactEmail());

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

  // A render function (not a nested component) so typing in the field doesn't
  // remount it and lose focus.
  const renderDbField = (item: (typeof DB_ITEMS)[number]) => {
    const { source, label, placeholder, help } = item;
    const saved = hasDbKey(source);
    return (
      <div key={source} className="space-y-1">
        <div className="flex items-center justify-between">
          <Label className="text-sm">{label}</Label>
          {saved && <span className="inline-flex items-center gap-1 text-[11px] text-emerald-600 dark:text-emerald-400"><Check className="size-3" />Saved</span>}
        </div>
        <div className="flex items-center gap-1.5">
          <div className="relative flex-1">
            <Input type={dbShown[source] ? "text" : "password"} autoComplete="off" spellCheck={false}
              placeholder={saved ? "•••••••• (replace)" : placeholder}
              value={dbValues[source] || ""} onChange={e => setDbValues(v => ({ ...v, [source]: e.target.value }))}
              className="pr-9 font-mono text-xs" />
            <button type="button" onClick={() => setDbShown(s => ({ ...s, [source]: !s[source] }))}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground" title={dbShown[source] ? "Hide" : "Show"}>
              {dbShown[source] ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
            </button>
          </div>
          <Button size="sm" variant="outline" className="h-8 shrink-0" disabled={!dbValues[source]?.trim()}
            onClick={() => { setDbKey(source, dbValues[source]); setDbValues(v => ({ ...v, [source]: "" })); force(); toast.success(`${label} key saved`); }}>Save</Button>
          {saved && <Button size="sm" variant="ghost" className="h-8 px-2 shrink-0 text-muted-foreground"
            onClick={() => { setDbKey(source, ""); force(); toast.success(`${label} key removed`); }} title="Remove"><Trash2 className="size-4" /></Button>}
        </div>
        <p className="text-[10px] text-muted-foreground">{help}</p>
      </div>
    );
  };

  const TABS: { id: "models" | "databases"; label: string }[] = [
    { id: "models", label: "Models" },
    { id: "databases", label: "Databases" },
  ];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-3xl max-h-[92vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2"><KeyRound className="size-4" />API keys</DialogTitle>
          <DialogDescription>Keys for cloud models, and for databases that need one. Local models (Ollama, LEADS) and most databases need no key.</DialogDescription>
        </DialogHeader>

        <div className="flex gap-5 min-h-[34rem]">
          {/* Left tab rail */}
          <div className="w-36 shrink-0 space-y-1 border-r pr-2">
            {TABS.map(t => (
              <button key={t.id} type="button" onClick={() => setTab(t.id)}
                className={`w-full flex items-center gap-2 px-2.5 py-2 rounded-md text-sm text-left leading-tight transition-colors ${tab === t.id ? "bg-primary/10 text-foreground font-medium" : "text-muted-foreground hover:bg-muted"}`}>
                {t.id === "models" ? <KeyRound className="size-4 shrink-0" /> : <Database className="size-4 shrink-0" />}
                <span className="min-w-0">{t.label}</span>
              </button>
            ))}
          </div>

          {/* Right content panel sizes to its content (no inner scroll); the whole
              dialog scrolls instead, and only on very short viewports. */}
          <div className="flex-1 min-w-0 pr-1">
            {tab === "models" && (
              <div className="space-y-3">
                {mode === "keychain" && (
                  <div className="space-y-3">
                    {ORDER.map(p => (
                      <KeyField key={p} p={p} saved={kcStatus[p]} showRemove={kcStatus[p]}
                        onSave={() => doKeychainSave(p)} onRemove={() => doKeychainRemove(p)} />
                    ))}
                  </div>
                )}

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

                <div className="rounded-md bg-muted/50 border p-2.5 text-[11px] text-muted-foreground space-y-1.5">
                  <div className="flex items-start gap-2">
                    <ShieldCheck className="size-4 shrink-0 mt-0.5 text-emerald-600 dark:text-emerald-400" />
                    <span>
                      {mode === "keychain"
                        ? "Keys are kept in your device keychain and read directly when a model runs. They never leave this device or get saved to the project."
                        : "Keys are encrypted on this device with your passphrase (AES-GCM) and stored in this browser only, unlocked once per session. They never leave this device or get saved to the project."}
                    </span>
                  </div>
                  {kcAvailable && (
                    <button type="button" onClick={() => setMode(mode === "keychain" ? "encrypted" : "keychain")}
                      className="ml-6 text-primary/80 hover:text-primary underline underline-offset-2">
                      {mode === "keychain" ? "Use an encrypted passphrase instead" : "Use the device keychain instead"}
                    </button>
                  )}
                </div>
              </div>
            )}

            {tab === "databases" && (
              <div className="space-y-5">
                <section className="space-y-2">
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Contact email</span>
                      <span className="text-[10px] rounded-full bg-sky-500/10 text-sky-600 dark:text-sky-400 px-1.5 py-0.5">recommended</span>
                    </div>
                    <p className="text-[11px] text-muted-foreground">
                      Not an account. Unpaywall and NCBI ask who is calling: Unpaywall refuses requests without a
                      real address, and NCBI raises your rate limit when you give one. Adding yours finds more full texts.
                    </p>
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="contact-email" className="text-sm">Your email</Label>
                    <Input
                      id="contact-email" type="email" value={email} placeholder="you@university.edu"
                      onChange={e => { setEmail(e.target.value); setContactEmail(e.target.value); }}
                    />
                    {email.trim() && !isValidContactEmail(email) && (
                      <p className="text-[11px] text-amber-600 dark:text-amber-400">
                        These APIs reject placeholder addresses. Use a real one you can receive mail at.
                      </p>
                    )}
                    {email.trim() && isValidContactEmail(email) && (
                      <p className="text-[11px] text-emerald-600 dark:text-emerald-400 inline-flex items-center gap-1">
                        <Check className="size-3" />Sent only to Unpaywall, NCBI, OpenAlex and Crossref, with each request.
                      </p>
                    )}
                  </div>
                </section>

                <section className="space-y-3 border-t pt-4">
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Free</span>
                      <span className="text-[10px] rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 px-1.5 py-0.5">quick signup</span>
                    </div>
                    <p className="text-[11px] text-muted-foreground">CORE needs a key; the others work without one but are faster or more reliable with it.</p>
                  </div>
                  {DB_ITEMS.filter(i => i.tier === "free").map(renderDbField)}
                </section>

                <section className="space-y-3 border-t pt-4">
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Subscription</span>
                      <span className="text-[10px] rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400 px-1.5 py-0.5">institutional</span>
                    </div>
                    <p className="text-[11px] text-muted-foreground">Works within your institution's access (on-campus IP or an entitlement token).</p>
                  </div>
                  {DB_ITEMS.filter(i => i.tier === "subscription").map(renderDbField)}
                </section>

                <p className="text-[10px] text-muted-foreground">Keys are stored on this device and sent only to that database.</p>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
