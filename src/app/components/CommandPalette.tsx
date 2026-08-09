import { useEffect, useState } from "react";
import { useStore } from "../lib/store";
import { CommandDialog, CommandInput, CommandList, CommandEmpty, CommandGroup, CommandItem } from "./ui/command";
import { Home, BarChart3, ShieldCheck, FileSearch, FileDown, FlaskConical, Network, Table2, ScanText, GitBranch, Sigma, Users, PenLine, Moon, Sun } from "lucide-react";
import type { PageId } from "../lib/store";

const NAV: { id: PageId; label: string; icon: any }[] = [
  { id: "home", label: "Research Strategy", icon: Home },
  { id: "simulation", label: "Search Planning", icon: BarChart3 },
  { id: "projects", label: "Projects", icon: Users },
  { id: "abstract", label: "Abstract Screening", icon: FileSearch },
  { id: "acquisition", label: "Full-Text Acquisition", icon: FileDown },
  { id: "fulltext", label: "Full-Text Evidence", icon: FlaskConical },
  { id: "snowball", label: "Citation Snowballing", icon: Network },
  { id: "extraction", label: "Table Extraction", icon: Table2 },
  { id: "textextraction", label: "Text Extraction", icon: ScanText },
  { id: "quality", label: "Quality Assessment", icon: ShieldCheck },
  { id: "prisma", label: "Diagramming", icon: GitBranch },
  { id: "meta", label: "Meta-analysis", icon: Sigma },
  { id: "writing", label: "Writing Assistant", icon: PenLine },
];

// Command palette (Cmd/Ctrl-K): jump to any stage or run a quick action.
export function CommandPalette() {
  const s = useStore();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen(o => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const go = (id: PageId) => { s.setPage(id); setOpen(false); };
  const toggleTheme = () => {
    const root = document.documentElement;
    const dark = root.classList.contains("dark");
    root.classList.toggle("dark", !dark);
    setOpen(false);
  };

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput placeholder="Jump to a stage or run an action…" />
      <CommandList>
        <CommandEmpty>No matches.</CommandEmpty>
        <CommandGroup heading="Go to">
          {NAV.map(n => (
            <CommandItem key={n.id} value={n.label} onSelect={() => go(n.id)}>
              <n.icon className="size-4" />{n.label}
            </CommandItem>
          ))}
        </CommandGroup>
        <CommandGroup heading="Actions">
          <CommandItem value="toggle theme dark light" onSelect={toggleTheme}>
            <Moon className="size-4" /><Sun className="size-4" />Toggle light / dark theme
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
