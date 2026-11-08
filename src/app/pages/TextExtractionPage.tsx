import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "motion/react";
import { useStore, useStageModel, TextExtractionResult, TextEvidenceItem } from "../lib/store";
import { AIService } from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { ImportStudies } from "../components/ImportStudies";
import { readTextOnServer } from "../lib/pdfHandoff";
import { quoteInContext, quotePosition } from "../lib/quoteLocate";
import { PdfQuotePage } from "../components/PdfQuotePage";
import { hasPdfBlob } from "../lib/pdfBlobs";
import { guessTitle } from "../lib/pdfImport";
import { apiFetch } from "../lib/backendClient";
import { Alert, AlertDescription } from "../components/ui/alert";
import { EmptyState } from "../components/EmptyState";
import { Button } from "../components/ui/button";
import { Textarea } from "../components/ui/textarea";
import { Input } from "../components/ui/input";
import { Badge } from "../components/ui/badge";
import { Checkbox } from "../components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "../components/ui/dialog";
import { Popover, PopoverClose, PopoverContent, PopoverTrigger } from "../components/ui/popover";
import {
  ScanText, Sparkles, Search, AlertTriangle, Download,
  MapPin, Quote as QuoteIcon, FileSpreadsheet, Maximize2, ChevronDown, ChevronRight,
  FileText, ListChecks, X, Check, Table2, Loader2, FormInput, Upload, RotateCcw, GripVertical, Undo2,
} from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";
import { ControlPane, InlineStat, PaneDivider } from "../components/ControlPane";
import { getPdfBlob } from "../lib/pdfBlobs";
import ExcelJS from "exceljs";

const PRESETS = [
  "What was the primary outcome and its effect size?",
  "What was the sample size and participant demographics?",
  "List all reported adverse events with frequencies.",
  "Extract dose, frequency, and duration of the intervention.",
  "What were the inclusion and exclusion criteria for participants?",
];

// The form outlives the page: this stage is one of ten, and the reviewer walks
// between them. Leaving for Acquisition and coming back unmounts this page, so
// anything the reviewer built has to be written down rather than held in state.

/** Keep a piece of the form for the next visit. Failing to is never fatal. */
function remember(key: string, value: unknown) {
  try {
    const json = JSON.stringify(value);
    // A few megabytes is all there is, and a huge results table is not worth
    // losing the form definition over.
    if (json.length > 3_000_000) localStorage.removeItem(key);
    else localStorage.setItem(key, json);
  } catch { /* private window, or full: it works, it just forgets */ }
}

/** Read one back, or null if it is absent or no longer the right shape. */
function recall<T>(key: string, ok: (v: any) => boolean): T | null {
  try {
    const v = JSON.parse(localStorage.getItem(key) || "null");
    if (ok(v)) return v as T;
  } catch { /* ignore */ }
  return null;
}

// A structured extraction form: named fields the AI fills for each chosen
// article, producing a study-characteristics table.
type FormField = { id: string; label: string; type: string; options: string[]; description?: string };
type InferredField = { label: string; type: string; options: string[]; description: string;
                       include: boolean };
// A tab is a named group of fields plus the studies it covers.
type SheetRole = "extract" | "dictionary" | "other";
const ROLE_LABEL: Record<SheetRole, string> = {
  extract: "Data to extract", dictionary: "Data dictionary", other: "Skip this tab",
};
type InferredTab = { name: string; fields: InferredField[]; role: SheetRole;
                     // Paper ids this tab covers. null means every study.
                     studies?: string[] | null;
                     // The sheet it was read from, kept because the name is the
                     // reviewer's to change and the file's is not.
                     sheet?: string };

/** The workbook sheet behind a tab, which a rename must not lose. */
const sheetOf = (t: { name: string; sheet?: string }) => t.sheet || t.name;
// What survives onto the form: which fields belong to which tab, and who it is for.
type FormTabSpec = { name: string; fieldIds: string[]; studies?: string[] | null };
const slug = (x: string) => (x || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "") || "field";
// Below this, a "full text" is a running head and a page number. Asking a model
// to find a sample size in it produces a row of blanks, which reads as a paper
// that reports nothing rather than as a document nobody managed to read.
const THIN_TEXT = 400;

// Fields per request. An imported workbook can carry two hundred, and asking
// for all of them at once does not make a hard question: it makes a long
// answer, which the model cannot finish, and a half-finished answer parses to
// nothing at all, so every cell comes back blank.
//
// Measured on a 41,000-character paper with the default local model, asking
// for values whose answers are known: 16 fields gave 12 of 12 right in 15s;
// 20 fields returned no usable JSON, twice. Twelve leaves a margin.
const FIELDS_PER_CALL = 12;

/** One row of extracted values for one study. */
/** Why a cell is empty. "reported" means it holds a value. */
export type CellStatus = "reported" | "not_reported" | "unread";

/**
 * What a review writes in a cell the paper does not report.
 *
 * A dash, which is the convention in a published review table, rather than the
 * words: the cell is read down a column beside a hundred others, and the
 * distinction from a cell nothing read is carried by the hover text and the
 * source dialog instead of by two words in every row.
 */
export const NOT_REPORTED = "\u2014";

export type FormRow = {
  paper_id: string;
  title: string;
  values: Record<string, any>;
  status?: Record<string, "suggested" | "accepted" | "rejected">;
  // An empty cell used to mean two things: the paper does not report it, and
  // nothing ever read it. The first belongs in the review; the second is work
  // still to do, and a reviewer cannot tell them apart from a blank.
  cell?: Record<string, CellStatus>;
};

/**
 * The workbook a tabbed form exports as: one sheet per tab, one row per study,
 * that tab's fields as its columns.
 *
 * Not one sheet per study holding every field. That was what came out before,
 * and it is neither the form the reviewer designed nor something anyone can
 * read down a column.
 */
export function tabSheets(
  tabs: FormTabSpec[],
  fields: FormField[],
  rows: FormRow[],
): { name: string; head: string[]; body: string[][] }[] {
  const byId = new Map(fields.map(f => [f.id, f]));
  const cell = (r: FormRow, id: string) => {
    if (r.status?.[id] === "rejected") return "";
    const value = String(r.values?.[id] ?? "").trim();
    if (value) return value;
    // A blank column in an exported workbook reads as an extraction nobody
    // finished. "Not reported" is a finding, and the reviewer who checks this
    // against the paper needs to see which of the two it was.
    return r.cell?.[id] === "not_reported" ? NOT_REPORTED : "";
  };

  return tabs.map((tab, ti) => {
    const cols = tab.fieldIds.map(id => byId.get(id)).filter(Boolean) as FormField[];
    // Only the studies this tab covers: a tab for one study design holds that
    // design's studies, which is what its study list said.
    const mine = rows.filter(r => !tab.studies || tab.studies.includes(r.paper_id));
    return {
      name: tab.name || `Tab ${ti + 1}`,
      head: ["Article", ...cols.map(c => c.label)],
      body: mine.map(r => [r.title || "Untitled", ...cols.map(c => cell(r, c.id))]),
    };
  }).filter(sheet => sheet.head.length > 1);
}

/**
 * The fields to extract for one study: the fields of the tabs that cover it.
 *
 * A tab can be scoped to a list of studies, and the table already hides it for
 * the others. Extraction ignored that and asked every field of every paper,
 * which is how a preclinical sheet came back holding a clinical cohort's
 * baseline table, and how one paper's columns were filled with results its
 * discussion had cited from somebody else's work. Asking only what the sheet
 * covers cannot produce those values at all.
 *
 * A study that no tab claims is read against the whole form, because the
 * alternative is an empty row and no way to see why.
 */
export function fieldsForStudy<T extends { id: string }>(
  tabs: FormTabSpec[] | null | undefined,
  fields: T[],
  paperId: string,
): T[] {
  if (!tabs || !tabs.length) return fields;
  const covered = new Set(
    tabs.filter(t => !t.studies || t.studies.includes(paperId))
        .flatMap(t => t.fieldIds));
  const mine = fields.filter(f => covered.has(f.id));
  return mine.length ? mine : fields;
}

const DEFAULT_FORM_FIELDS: FormField[] = [
  { id: "design", label: "Study design", type: "text", options: [], description: "the study design, e.g. randomized controlled trial, cohort, case-control, or cross-sectional survey" },
  { id: "sample_size", label: "Sample size", type: "number", options: [], description: "the total number of participants analysed (N); check the abstract, methods, or a baseline/flow table" },
  { id: "population", label: "Population", type: "text", options: [], description: "who was studied: age range, condition, and setting" },
  { id: "intervention", label: "Intervention", type: "text", options: [], description: "the intervention or exposure studied; write 'none (observational)' if the study has no intervention" },
  { id: "outcome", label: "Primary outcome", type: "text", options: [], description: "the primary outcome measure or main variable assessed" },
  { id: "result", label: "Main result / effect size", type: "text", options: [], description: "the main quantitative finding WITH its value: a correlation coefficient (r), beta, odds/risk/hazard ratio, or mean difference, including its confidence interval or p-value if reported" },
];

// Section → badge palette. Falls back to slate for unrecognised labels.
const SECTION_STYLE: Record<string, string> = {
  Abstract:     "bg-violet-100 text-violet-800 border-violet-200",
  Background:   "bg-blue-100 text-blue-800 border-blue-200",
  Introduction: "bg-blue-100 text-blue-800 border-blue-200",
  Methods:      "bg-sky-100 text-sky-800 border-sky-200",
  Results:      "bg-emerald-100 text-emerald-800 border-emerald-200",
  Discussion:   "bg-amber-100 text-amber-800 border-amber-200",
  Conclusion:   "bg-amber-100 text-amber-800 border-amber-200",
  Limitations:  "bg-orange-100 text-orange-800 border-orange-200",
  References:   "bg-slate-100 text-slate-700 border-slate-200",
};
// The dropdowns on a tab's bar are the same control with different lists, so
// they share a trigger and a row. Sizes are spelled out because a bare button
// otherwise takes the 16px base rule in theme.css.
/**
 * The picture that follows the cursor while a field is being dragged.
 *
 * Handing the live card to setDragImage gives a bare line of text: the card is
 * mostly inputs, and that is what the browser snapshots. A detached clone,
 * sized to the card and with each input's current value written in (cloneNode
 * copies the attribute, not what was typed), drags the whole bar instead.
 */
function dragGhost(card: HTMLElement, e: React.DragEvent) {
  const r = card.getBoundingClientRect();
  const ghost = card.cloneNode(true) as HTMLElement;
  const live = card.querySelectorAll("input");
  ghost.querySelectorAll("input").forEach((el, i) => {
    (el as HTMLInputElement).value = (live[i] as HTMLInputElement | undefined)?.value ?? "";
  });
  Object.assign(ghost.style, {
    position: "fixed", top: "0px", left: "-10000px", margin: "0",
    width: `${r.width}px`, pointerEvents: "none",
  });
  ghost.classList.add("shadow-lg");
  document.body.appendChild(ghost);
  e.dataTransfer.setDragImage(ghost, e.clientX - r.left, e.clientY - r.top);
  // The snapshot is taken during this event, so the clone's work is already
  // done by the time the stack unwinds.
  setTimeout(() => ghost.remove(), 0);
}

const PICKER_TRIGGER = "inline-flex items-center gap-1.5 h-7 px-2 rounded-md border "
                     + "bg-background text-xs hover:bg-muted";
const PICKER_ROW = "w-full text-left px-2 py-1 rounded-md text-xs font-normal hover:bg-muted";

// What to call the undo shortcut in a tooltip, on the machine reading it.
const UNDO_KEY = typeof navigator !== "undefined" && /Mac|iPhone|iPad/i.test(navigator.userAgent)
  ? "⌘Z" : "Ctrl+Z";

function sectionBadgeClass(section?: string): string {
  if (!section) return "bg-slate-100 text-slate-700 border-slate-200";
  return SECTION_STYLE[section] || "bg-slate-100 text-slate-700 border-slate-200";
}

// Compact count pill used in the header summary line.
function Pill({
  icon: Icon, children, tone = "default", title,
}: {
  icon: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
  tone?: "default" | "green" | "amber";
  title?: string;
}) {
  const cls = tone === "green" ? "bg-emerald-50 text-emerald-700 border-emerald-200"
    : tone === "amber" ? "bg-amber-50 text-amber-700 border-amber-200"
    : "bg-muted text-muted-foreground border-transparent";
  return (
    <span title={title} className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`}>
      <Icon className="size-3" />{children}
    </span>
  );
}

// An import in flight, held outside the component.
//
// Reading a workbook is a sequence of model calls that can run for a minute. If
// this lived in component state, leaving Text Extraction would unmount the page
// and the work would be thrown away (or worse, finish and write into a
// component nobody is looking at). Keeping it here means the reviewer can go and
// do something else and find the result waiting when they come back.
type ImportJob = {
  ctl: AbortController;
  progress: { done: number; total: number; name: string } | null;
  tabs: InferredTab[] | null;
  sheets: { name: string; rows: string[][] }[];
  note: string;
  finished: boolean;
};
let importJob: ImportJob | null = null;
const importWatchers = new Set<() => void>();
const notifyImport = () => importWatchers.forEach(f => f());

export function TextExtractionPage() {
  const s = useStore();
  const extractModel = useStageModel("text-extraction");
  const [query, setQuery] = useState(PRESETS[0]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [listQ, setListQ] = useState("");
  // The "Ask in natural language" panel collapses once results exist, so the
  // two-pane gets the screen. Starts open only when nothing's been run yet.
  const [askOpen, setAskOpen] = useState(s.textExtractions.length === 0);
  const task = s.tasks["text-extract"];
  const running = task?.status === "running";

  const acquired = useMemo(() => Object.values(s.fullTexts).filter(r => r.status === "found" && r.text), [s.fullTexts]);
  // Documents that arrived but carry no words: a scanned PDF imports cleanly
  // and has nothing for this stage to read. Without naming that, the page tells
  // the reviewer to go and acquire a full text they have already supplied.
  const textless = useMemo(
    () => Object.values(s.fullTexts).filter(r => r.status === "found" && !(r.text || "").trim()),
    [s.fullTexts]);
  const missing = useMemo(() => Object.values(s.fullTexts).filter(r => r.status === "missing"), [s.fullTexts]);

  // Structured extraction form: fields (persisted), which articles to run on,
  // and the resulting matrix.
  const [fields, setFields] = useState<FormField[]>(() =>
    recall<FormField[]>("ee:extract-form", v => Array.isArray(v) && v.length > 0) || DEFAULT_FORM_FIELDS);
  useEffect(() => remember("ee:extract-form", fields), [fields]);

  // Which half of the stage was open, so coming back does not land on the
  // other one with the form apparently gone.
  const [mode, setMode] = useState<"ask" | "form">(() =>
    recall<string>("ee:extract-mode", v => v === "ask" || v === "form") === "form" ? "form" : "ask");
  useEffect(() => remember("ee:extract-mode", mode), [mode]);
  const [formCollapsed, setFormCollapsed] = useState(false);
  const [resultsCollapsed, setResultsCollapsed] = useState(false);
  // Which cell's provenance is open. A value in a spreadsheet and a value a
  // model invented look identical; the sentence it was read from is the only
  // thing that tells them apart, so it has to be reachable from the cell.
  const [source, setSource] = useState<{ paperId: string; fieldId: string } | null>(null);
  // What the reviewer is typing in that dialog, separate from the stored value
  // so a half-typed correction is never what the table holds.
  const [draft, setDraft] = useState("");
  const [deselected, setDeselected] = useState<Set<string>>(() =>
    new Set(recall<string[]>("ee:extract-skip", Array.isArray) || []));
  useEffect(() => remember("ee:extract-skip", [...deselected]), [deselected]);
  const [formBusy, setFormBusy] = useState(false);
  const [formProgress, setFormProgress] = useState<{ done: number; total: number }>({ done: 0, total: 0 });
  const [formCurrent, setFormCurrent] = useState<string>("");
  const [formRows, setFormRows] = useState<{ paper_id: string; title: string; values: Record<string, any>; meta: Record<string, { source_quote: string; confidence: string; needs_review: boolean;
                        /** Why the app flagged it, when it has a reason to give. */ review_reason?: string;
                        // Set when the reviewer typed the value themselves, with
                        // what the model had said: an extraction table that cannot
                        // show which cells a human corrected is not an audit trail.
                        edited?: boolean; original?: string }>; status: Record<string, "suggested" | "accepted" | "rejected">; cell?: Record<string, CellStatus> }[]>(
    () => recall<any[]>("ee:extract-rows", Array.isArray) || []);
  // Not while a run is going: that would serialise the whole table once per
  // article. The run ends, formBusy clears, and this writes the finished set.
  useEffect(() => { if (!formBusy) remember("ee:extract-rows", formRows); }, [formRows, formBusy]);
  // Per-field accept/reject for AI suggestions (reviewer-in-the-loop).
  const setFieldStatus = (pid: string, fid: string, st: "accepted" | "rejected") =>
    setFormRows(rows => rows.map(r => r.paper_id === pid ? { ...r, status: { ...r.status, [fid]: st } } : r));

  /** The reviewer's own value for a cell. Theirs is the one that counts, so it
   *  is kept as accepted and needs no further review; what the model said is
   *  kept beside it, once, so the correction is visible and reversible. */
  const setFieldValue = (pid: string, fid: string, value: string) =>
    setFormRows(rows => rows.map(r => {
      if (r.paper_id !== pid) return r;
      const was = r.meta?.[fid];
      return {
        ...r,
        values: { ...r.values, [fid]: value },
        meta: {
          ...(r.meta || {}),
          [fid]: {
            source_quote: was?.source_quote || "",
            confidence: was?.confidence || "none",
            needs_review: false,
            edited: true,
            original: was?.original ?? String(r.values[fid] ?? ""),
          },
        },
        status: { ...(r.status || {}), [fid]: "accepted" as const },
      };
    }));

  /** Put the model's value back, and stop calling the cell edited. */
  const revertField = (pid: string, fid: string) =>
    setFormRows(rows => rows.map(r => {
      if (r.paper_id !== pid) return r;
      const was = r.meta?.[fid];
      if (!was?.edited) return r;
      const { edited, original, ...rest } = was;
      return {
        ...r,
        values: { ...r.values, [fid]: original ?? "" },
        meta: { ...(r.meta || {}), [fid]: { ...rest, needs_review: true } },
        status: { ...(r.status || {}), [fid]: "suggested" as const },
      };
    }));

  // The cell being typed into, if any.
  const [editing, setEditing] = useState<{ paperId: string; fieldId: string; text: string } | null>(null);
  const formFileRef = useRef<HTMLInputElement>(null);

  // The stage gates. Held as a value and returned below, after every hook,
  // rather than returned from here: the store restores itself from IndexedDB,
  // so this page can render before the review it reads arrives. Returning
  // early would run fewer hooks on that first pass than on the next one, which
  // React refuses ("Rendered more hooks than during the previous render"),
  // taking the whole stage down at the moment the data lands.
  const gate = !s.results
    ? <EmptyState icon={ScanText} title="Screening not complete yet"
        description="Complete Abstract Screening first; included studies unlock natural-language text extraction here."
        action={{ label: "Go to Abstract Screening", onClick: () => s.setPage("abstract"), icon: Search }}
        extra={<ImportStudies stage="text-extraction" />} />
    : acquired.length === 0
      ? (textless.length > 0
        ? <EmptyState icon={ScanText} title="Nothing could be read from those documents"
            description={`${textless.length} of your file${textless.length === 1 ? " is" : "s are"} scanned or image-only. `
              + "They were read on import, including a pass over the page images, and no text came back. "
              + "Every other stage still has them."}
            extra={<ImportStudies stage="text-extraction" label="Add studies with their PDFs" />} />
        : <EmptyState icon={ScanText} title="No full texts yet"
            description="Fetch the full text for your included studies on Full-Text Acquisition, then extract values here."
            action={{ label: "Go to Full-Text Acquisition", onClick: () => s.setPage("acquisition"), icon: Search }}
            extra={<ImportStudies stage="text-extraction" label="Add studies with their PDFs" />} />)
      : null;

  async function run() {
    if (!query.trim()) { toast.error("Enter a question or instruction first."); return; }
    const { abort } = s.startTask("text-extract", [{ id: "te", label: "Extracting from text", status: "running" }]);
    s.updateTask("text-extract", { progress: { done: 0, total: acquired.length } });
    const signal = abort.signal;
    // 8 concurrent requests saturates SGLang's continuous-batching scheduler;
    // on Ollama (single-threaded) it still helps by hiding HTTP round-trip overhead.
    const BATCH = 8;
    try {
      const out: TextExtractionResult[] = [];
      let done = 0;
      for (let i = 0; i < acquired.length; i += BATCH) {
        if (signal.aborted) break;
        const batch = acquired.slice(i, i + BATCH);
        s.updateTask("text-extract", {
          progress: { done, total: acquired.length, label: batch[0].title.slice(0, 80) },
          detail: `Batch ${Math.floor(i / BATCH) + 1} / ${Math.ceil(acquired.length / BATCH)}`,
        });
        const settled = await Promise.allSettled(
          batch.map(r => AIService.extractFromText(r.text || "", query, signal)),
        );
        settled.forEach((result, j) => {
          const r = batch[j];
          if (result.status === "fulfilled") {
            const ext = result.value;
            out.push({
              paper_id: r.paper_id,
              title: r.title,
              query,
              answer: ext.answer,
              summary: ext.summary,
              evidence: ext.evidence || [],
              spans: ext.spans || [],
              values: ext.values || [],
            });
          } else if (!signal.aborted) {
            console.error(`text-extract ${i + j + 1} failed:`, result.reason?.message);
          }
        });
        done += batch.length;
        s.updateTask("text-extract", { progress: { done, total: acquired.length } });
      }
      if (signal.aborted) {
        // Save partial progress: merge whatever finished this run into the
        // existing answers for THIS question (by paper). Answers to a previous,
        // different question are not mixed in.
        const byId = new Map((s.textExtractions ?? []).filter(r => r.query === query).map(r => [r.paper_id, r]));
        out.forEach(r => byId.set(r.paper_id, r));
        const merged = [...byId.values()];
        s.setTextExtractions(merged);
        s.updateTask("text-extract", { status: "canceled" });
        toast.info(`Canceled: ${out.length} extracted this run, ${merged.length} kept in total.`);
        return;
      }
      s.setTextExtractions(out);
      s.updateTask("text-extract", { status: "done" });
      toast.success(`Extracted from ${out.length} articles`);
      setAskOpen(false);   // collapse the question panel to free up the screen
    } catch (e: any) {
      s.updateTask("text-extract", { status: "error", detail: e?.message });
    }
  }

  function exportJson() {
    const blob = new Blob([JSON.stringify({ query, results: s.textExtractions }, null, 2)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "text_extractions.json"; a.click(); URL.revokeObjectURL(a.href);
  }

  // ── Structured extraction form ─────────────────────────────────────────────
  // ── Build the form from a sheet the reviewer already uses ─────────────────
  // Taking the headers literally (what "Import form" does) gives untyped fields
  // with no description, and the description is what tells the extractor where
  // in a paper to look. So the values are read too and the whole thing is
  // inferred. The values are examples only: nothing here is written in as a
  // result, because the reviewer extracted those, not this app.
  const [inferBusy, setInferBusy] = useState(false);
  // Real progress, not a spinner: one model call per sheet, so the reviewer
  // can see a five-tab workbook moving rather than guessing whether it hung.
  const [inferProgress, setInferProgress] = useState<{ done: number; total: number; name: string } | null>(null);
  // Reading a workbook is a sequence of model calls, so it can be long enough
  // to regret starting. Cancelling aborts the request in flight and stops the
  // loop before the next one.
  const inferAbort = useRef<AbortController | null>(null);
  // Dragging a field: its index while in flight, and the index it would land
  // on. Arrow buttons moved one place per click, which is useless for putting
  // column 40 next to column 3.
  const [dragField, setDragField] = useState<number | null>(null);
  // The form's own list drags by field id: it shows one tab at a time, so a
  // position in it is not a position in the form.
  const [dragForm, setDragForm] = useState<string | null>(null);
  const [formRenaming, setFormRenaming] = useState<{ ti: number; text: string } | null>(null);
  const formRenameOpen = useRef(false);
  const [dragFormOver, setDragFormOver] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState<number | null>(null);
  // Designing a workbook is work in its own right: seventy fields sorted into
  // five tabs, renamed and reordered, before a single value is extracted.
  // Losing it to a reload would be losing all of that, so the designer is
  // written down like the form it produces. Closing the dialog clears it.
  const [inferredTabs, setInferredTabs] = useState<InferredTab[] | null>(() =>
    recall<InferredTab[]>("ee:extract-design", v => Array.isArray(v) && v.length > 0));
  useEffect(() => remember("ee:extract-design", inferredTabs), [inferredTabs]);
  // Open only when the reviewer opens it. The draft survives a reload so that
  // an afternoon of sorting fields into tabs is not lost, but a workbook
  // imported once must not reopen itself every time this stage is visited.
  const [designOpen, setDesignOpen] = useState(false);
  const [renaming, setRenaming] = useState<{ ti: number; text: string } | null>(null);
  const renameOpen = useRef(false);
  const [activeInferTab, setActiveInferTab] = useState(() =>
    recall<number>("ee:extract-design-tab", v => typeof v === "number" && v >= 0) ?? 0);
  useEffect(() => remember("ee:extract-design-tab", activeInferTab), [activeInferTab]);
  // Undo. Deleting a field is one small x in a list of seventy, so it has to
  // be reversible. Only structural edits are recorded: deleting, adding,
  // reordering, moving between tabs, and what a tab is for. Typing inside a
  // field is left to the browser's own text undo, which a keystroke-level
  // history here would fight with.
  const [history, setHistory] = useState<{ tabs: InferredTab[]; active: number; what: string }[]>([]);
  const [previewing, setPreviewing] = useState(false);
  const [previewShowing, setPreviewShowing] = useState<"file" | "output">("file");
  // The workbook as read, kept so changing a tab's role re-reads it as a
  // dictionary (or back) without another round trip to the model.
  const [rawSheets, setRawSheets] = useState<{ name: string; rows: string[][] }[]>(
    () => recall<{ name: string; rows: string[][] }[]>("ee:extract-sheets", Array.isArray) || []);
  useEffect(() => remember("ee:extract-sheets", rawSheets), [rawSheets]);
  // Set when the form came from a multi-tab workbook; null for a plain form.
  const [formTabs, setFormTabs] = useState<FormTabSpec[] | null>(() =>
    recall<FormTabSpec[]>("ee:extract-tabs", v => Array.isArray(v) && v.length > 0));
  useEffect(() => remember("ee:extract-tabs", formTabs), [formTabs]);

  // What was remembered was true of the review it was saved in. Rows for a
  // paper this review does not have, and tabs naming fields that no longer
  // exist, are dropped on the way in rather than shown as though they were
  // this form's own output.
  const pruned = useRef(false);
  useEffect(() => {
    // Wait for the review itself. The store restores from IndexedDB, so a
    // render can arrive before any full text does, and pruning against an
    // empty review would throw away everything that was remembered.
    if (pruned.current || !acquired.length) return;
    pruned.current = true;
    const papers = new Set(acquired.map(p => p.paper_id));
    setFormRows(rows => (rows.some(r => !papers.has(r.paper_id))
      ? rows.filter(r => papers.has(r.paper_id)) : rows));
    const live = new Set(fields.map(f => f.id));
    setFormTabs(ts => {
      if (!ts) return ts;
      const next = ts.map(t => ({ ...t, fieldIds: t.fieldIds.filter(id => live.has(id)) }));
      // If no tab names a live field, the fields win: a tabbed form whose tabs
      // are all empty would show an empty editor over a full form.
      return next.some(t => t.fieldIds.length) ? next : null;
    });
  }, [acquired, fields]);
  const [inferSource, setInferSource] = useState(() =>
    recall<string>("ee:extract-design-note", v => typeof v === "string") || "");
  useEffect(() => remember("ee:extract-design-note", inferSource), [inferSource]);
  // The uploaded workbook's own shape: each tab and its headers in their
  // original order. Kept so the export is the same workbook the reviewer gave
  // us, including columns they chose not to extract, which stay blank rather
  // than disappearing and shifting everything left when they paste it back.
  const [sheetTabs, setSheetTabs] = useState<{ name: string; headers: string[] }[] | null>(null);

  // Every worksheet, not just the first. A real extraction template is several
  // tabs, and reading only worksheets[0] silently dropped the rest.
  async function readSheets(file: File): Promise<{ name: string; rows: string[][] }[]> {
    const out: { name: string; rows: string[][] }[] = [];
    if (/\.(xlsx|xls|xlsm)$/i.test(file.name)) {
      const wb = new ExcelJS.Workbook();
      await wb.xlsx.load(await file.arrayBuffer());
      wb.worksheets.forEach(ws => {
        const rows: string[][] = [];
        ws.eachRow({ includeEmpty: false }, r => {
          const cells: string[] = [];
          r.eachCell({ includeEmpty: true }, c => cells.push(String(c.value ?? "").trim()));
          rows.push(cells);
        });
        if (rows.length) out.push({ name: ws.name || `Sheet ${out.length + 1}`, rows });
      });
    } else {
      const text = await file.text();
      const sep = (text.split(/\r?\n/)[0] || "").includes("\t") ? "\t" : ",";
      const rows: string[][] = [];
      for (const line of text.split(/\r?\n/)) {
        if (!line.trim()) continue;
        rows.push(line.split(sep).map(x => x.replace(/^"|"$/g, "").trim()));
      }
      if (rows.length) out.push({ name: file.name.replace(/\.[^.]+$/, ""), rows });
    }
    return out;
  }

  // Find the header row and read its labels.
  //
  // Two things that lost most of a real template's columns:
  //
  //  - Taking the FIRST row with a couple of filled cells. Templates open with a
  //    title, or a "Reviewer / Date" line, and that row won. The row with the
  //    MOST labels is the header; ties keep the earliest.
  //  - Dropping a column whose header cell was empty. A merged two-row header
  //    (a group spanning several columns, with the real names beneath) leaves
  //    every cell of the merge blank except the first, so most columns vanished.
  //    The group label is carried across the blanks and joined to the name.
  function headerSpec(rows: string[][]) {
    let headerIndex = 0, best = 0;
    for (let i = 0; i < Math.min(rows.length, 15); i++) {
      const n = rows[i].filter(c => c && c.length <= 60).length;
      if (n > best) { best = n; headerIndex = i; }
    }
    const top = rows[headerIndex] || [];
    const above = headerIndex > 0 ? rows[headerIndex - 1] : [];
    // A group row has fewer labels than the header row, AND reaches across the
    // table. The span test is what separates it from a "Reviewer: HT / Date:"
    // line, which also sits above the headers with a couple of cells but only
    // in the first columns; without it every label was prefixed with the
    // reviewer's name.
    const aboveFilled = above.filter(Boolean).length;
    let lastAbove = -1;
    above.forEach((c, i) => { if (c) lastAbove = i; });
    const useGroups = aboveFilled >= 1 && aboveFilled < best
      && lastAbove >= Math.floor((top.length - 1) / 2);

    let carry = "";
    const labels = top.map((cell, c) => {
      if (useGroups && above[c]) carry = above[c];
      const base = (cell || "").trim();
      if (!base) return "";
      return useGroups && carry ? `${carry} \u2013 ${base}` : base;
    });
    return { headerIndex, labels };
  }

  // A template usually carries a data dictionary: a tab listing each variable

  // with its definition, allowed values and units. It is not something to

  // extract from papers, it is the instructions for extracting everything else,

  // and treating it as another tab turned "Variable / Definition / Allowed

  // values" into three extraction fields.

  const DICTIONARY_SHEET = /^(data[\s_-]*)?(dictionary|codebook|definitions?|variable[\s_-]*(list|definitions?)|glossary|key)$/i;

  const normLabel = (x: string) => (x || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();


  type Definition = { definition: string; values: string[]; unit: string };

  // A tab's column is often the dictionary's name plus a unit or a qualifier:
  // the dictionary says "Median OS", the Outcomes tab says "Median OS (months)".
  // Exact first, then without the bracketed part, then an unambiguous prefix.
  function lookupDefinition(label: string, dict: Record<string, Definition>) {
    const exact = dict[normLabel(label)];
    if (exact) return exact;
    const bare = normLabel(label.replace(/\([^)]*\)/g, " "));
    if (dict[bare]) return dict[bare];
    // A grouped header reads "Outcomes \u2013 Median OS" while the dictionary
    // lists the bare name. That label shape is one this importer creates, so a
    // template with both grouped headers and a dictionary would otherwise match
    // nothing at all.
    const tail = normLabel(label.split("\u2013").pop() || "");
    if (tail && dict[tail]) return dict[tail];
    const hits = Object.keys(dict).filter(k => k && (bare.startsWith(k) || k.startsWith(bare)));
    return hits.length === 1 ? dict[hits[0]] : undefined;   // ambiguous: leave it alone
  }


  // Which column holds what, by its header. Dictionaries are written loosely.

  function dictionaryColumns(labels: string[]) {

    const find = (re: RegExp) => labels.findIndex(l => re.test(l || ""));

    return {

      name: find(/^(variable|field|item|name|column|element)/i),

      definition: find(/(definition|description|meaning|guidance|instruction|notes?)/i),

      values: find(/(allowed|permitted|valid|possible|options?|categor|codes?|response)/i),

      unit: find(/(unit|measure)/i),

    };

  }


  // Re-apply every dictionary tab over every extraction tab. Called on import
  // and whenever a role changes, so marking a tab as the dictionary takes
  // effect immediately rather than only on the next import.
  function applyDictionaries(tabs: InferredTab[], sheets: { name: string; rows: string[][] }[]) {
    let dict: Record<string, Definition> = {};
    for (const t of tabs.filter(t => t.role === "dictionary")) {
      const sheet = sheets.find(sh => sh.name === sheetOf(t));
      if (sheet) dict = { ...dict, ...readDictionary(sheet.rows) };
    }
    let used = 0;
    const next = tabs.map(t => (t.role !== "extract" ? t : {
      ...t,
      fields: t.fields.map(f => {
        const d = lookupDefinition(f.label, dict);
        if (!d) return f;
        used++;
        return {
          ...f,
          description: d.definition
            ? (d.unit ? `${d.definition} (in ${d.unit})` : d.definition)
            : f.description,
          ...(d.values.length > 1 ? { type: "select", options: d.values } : {}),
        };
      }),
    }));
    return { tabs: next, used };
  }

  function readDictionary(rows: string[][]): Record<string, Definition> {

    const { headerIndex, labels } = headerSpec(rows);

    const col = dictionaryColumns(labels);

    if (col.name < 0 || col.definition < 0) return {};    // not a dictionary after all

    const out: Record<string, Definition> = {};

    for (const r of rows.slice(headerIndex + 1)) {

      const key = normLabel(r[col.name] || "");

      if (!key) continue;

      out[key] = {

        definition: (r[col.definition] || "").trim(),

        values: col.values >= 0

          ? (r[col.values] || "").split(/[,;|/]| or /i).map(x => x.trim()).filter(Boolean)

          : [],

        unit: col.unit >= 0 ? (r[col.unit] || "").trim() : "",

      };

    }

    return out;

  }


  // Progress goes through the job so it survives leaving the page.
  function setJobProgress(p: { done: number; total: number; name: string } | null) {
    if (importJob) importJob.progress = p;
    setInferProgress(p);
    notifyImport();
  }

  // Re-attach to an import that is running, or finished, while this page was

  // not mounted. Without this, leaving Text Extraction mid-import and coming

  // back showed an idle page with the work invisibly lost.

  useEffect(() => {

    const sync = () => {

      const job = importJob;

      if (!job) { setInferBusy(false); setInferProgress(null); return; }

      setInferBusy(!job.finished);

      setInferProgress(job.progress);

      if (job.finished && job.tabs) {

        setInferredTabs(job.tabs);
        setDesignOpen(true);

        setRawSheets(job.sheets);

        setInferSource(job.note);

        setActiveInferTab(0);

        inferAbort.current = job.ctl;

        importJob = null;          // handed over; do not re-apply on every tick

      }

    };

    sync();

    importWatchers.add(sync);

    return () => { importWatchers.delete(sync); };

  }, []);


  async function inferFromSheet(file: File) {
    const ctl = new AbortController();
    inferAbort.current = ctl;
    importJob = { ctl, progress: null, tabs: null, sheets: [], note: "", finished: false };
    setInferBusy(true);
    try {
      const sheets = await readSheets(file);
      if (!sheets.length) { toast.error("That file has no readable sheets."); return; }
      setRawSheets(sheets);

      const payload = sheets.map(sh => {
        const { headerIndex: h, labels } = headerSpec(sh.rows);
        return {
          name: sh.name,
          columns: labels.map((header, c) => ({
            header,
            samples: sh.rows.slice(h + 1, h + 40).map(r => r[c] || "").filter(Boolean).slice(0, 8),
          })).filter(c => c.header),
        };
      }).filter(sh => sh.columns.length);
      if (!payload.length) { toast.error("No column headers found in that file."); return; }

      // One sheet per request, so each completion is visible progress.


      const perSheet: any[] = [];


      for (let k = 0; k < payload.length; k++) {


        setJobProgress({ done: k, total: payload.length + 1, name: payload[k].name });


        if (ctl.signal.aborted) return;



        const one = await apiFetch("/extraction/infer-form", {



          method: "POST", signal: ctl.signal,



          body: JSON.stringify({ sheets: [payload[k]], model: extractModel }),



        });


        perSheet.push(...(one?.tabs || []));


      }


      // Roles and rules need every tab, so they come last and count as one step.


      setJobProgress({ done: payload.length, total: payload.length + 1, name: "Working out the tabs" });


      if (ctl.signal.aborted) return;
        const meta = await apiFetch("/extraction/classify-tabs", {


        method: "POST", signal: ctl.signal,


        body: JSON.stringify({


          tabs: perSheet.map((t: any) => ({


            name: t.name,


            columns: (t.fields || []).map((f: any) => ({ header: f.label, samples: f.options || [] })),


          })),


          model: extractModel,


        }),


      });


      const out = {


        source: perSheet.length ? "model" : "empty",


        tabs: perSheet.map((t: any) => ({


          ...t,
          role: meta?.roles?.[t.name] || "extract",


        })),


      };
      const tabs: InferredTab[] = (out?.tabs || []).map((t: any) => ({
        name: String(t.name || "Tab"),
        sheet: String(t.name || "Tab"),
        fields: (t.fields || []).map((f: any) => ({
          label: String(f.label || "").trim(),
          type: ["number", "text", "select"].includes(f.type) ? f.type : "text",
          options: Array.isArray(f.options) ? f.options.map(String) : [],
          description: String(f.description || ""),
          include: f.include !== false,
        })).filter((f: InferredField) => f.label),
        role: (["extract", "dictionary", "other"].includes(t.role) ? t.role : "extract") as SheetRole,
      })).filter((t: InferredTab) => t.fields.length);

      // Your definitions beat anything inferred from the values: you wrote them.
      const applied = applyDictionaries(tabs, sheets);
      const fromDictionary = applied.used;
      if (!tabs.length) { toast.error("Could not read any variables from that file."); return; }

      setInferSource(
        fromDictionary
          ? `Definitions for ${fromDictionary} field${fromDictionary === 1 ? "" : "s"} came from your data dictionary.`
          : (out?.source === "model"
            ? "" : "No model available, so types were guessed from the values and no tab rules were set."));
      if (importJob) {

        importJob.tabs = applied.tabs;

        importJob.sheets = sheets;

        importJob.note = fromDictionary

          ? `Definitions for ${fromDictionary} field${fromDictionary === 1 ? "" : "s"} came from your data dictionary.`

          : "";

        importJob.finished = true;

      }

      setSheetTabs(payload.map(sh => ({ name: sh.name, headers: sh.columns.map(c => c.header) })));
      setInferredTabs(applied.tabs);
      setDesignOpen(true);
      setActiveInferTab(0);
    } catch (e: any) {
      // An abort is the reviewer's own doing, not a failure to report.
      if (ctl.signal.aborted || e?.name === "AbortError") toast.info("Import cancelled");
      else toast.error(e?.message || "Could not read that file");
    } finally {
      if (inferAbort.current === ctl) inferAbort.current = null;
      if (importJob?.ctl === ctl) {
        importJob.progress = null;
        if (!importJob.finished) importJob = null;    // cancelled or failed
      }
      setInferBusy(false);
      setInferProgress(null);
      notifyImport();
    }
  }

  const patchInferred = (i: number, p: Partial<InferredField>) =>
    setInferredTabs(tabs => (tabs || []).map((t, ti) => (ti !== activeInferTab ? t : {
      ...t, fields: t.fields.map((x, j) => (j === i ? { ...x, ...p } : x)),
    })));
  // Applying flattens the tabs into the one field list the rest of the page
  // works with, and keeps the tab structure beside it so the export can put
  // the results back into the same workbook shape.
  function applyInferred() {
    const tabs = inferredTabs || [];
    const next: FormField[] = [];
    const groups: FormTabSpec[] = [];
    tabs.forEach((t, ti) => {
      if (t.role !== "extract") return;   // a dictionary or a notes tab is not data
      const ids: string[] = [];
      t.fields.filter(f => f.include).forEach((f, k) => {
        const id = `${slug(f.label)}_${ti}_${k}`;
        next.push({ id, label: f.label, type: f.type, options: f.options, description: f.description });
        ids.push(id);
      });
      if (ids.length) groups.push({ name: t.name, fieldIds: ids, studies: t.studies ?? null });
    });
    if (!next.length) return;
    setFields(next);
    setFormTabs(groups.length > 1 ? groups : null);
    setFormRows([]);
    setInferredTabs(null);
    setPreviewing(false);
    toast.success(`Form set to ${next.length} field${next.length === 1 ? "" : "s"}`
      + (groups.length > 1 ? ` across ${groups.length} tabs` : ""));
  }

  // Changing a role re-reads the workbook, so marking a tab as the dictionary

  // starts describing the other tabs immediately rather than on re-import.

  // The results table mirrors the imported workbook: a strip of its tabs, each

  // showing its own columns and only the studies its rule covers. A rule is

  // read against what was actually extracted, so a study whose deciding field

  // is still empty is shown rather than hidden: dropping a row because a value

  // is missing would quietly shrink the table and look like a smaller review.

  const [resultsTab, setResultsTab] = useState(() =>
    recall<number>("ee:extract-tab", v => typeof v === "number" && v >= 0) ?? 0);
  useEffect(() => remember("ee:extract-tab", resultsTab), [resultsTab]);

  function tabMatches(spec: FormTabSpec | null, row: typeof formRows[number]) {
    // null means every study; otherwise the ones the reviewer picked.
    return !spec?.studies || spec.studies.includes(row.paper_id);
  }

  const shownTab = formTabs ? formTabs[Math.min(resultsTab, formTabs.length - 1)] : null;

  const shownFields = shownTab ? fields.filter(f => shownTab.fieldIds.includes(f.id)) : fields;

  const shownRows = formRows.filter(r => tabMatches(shownTab, r));


  // Designing the workbook: tabs and fields can be added, renamed, reordered,


  // moved between tabs and deleted, so the output is whatever the reviewer


  // wants rather than only whatever their file happened to contain.


  const emptyField = (): InferredField =>


    ({ label: "New field", type: "text", options: [], description: "", include: true });



  // Open the designer on the form as it stands, so the layout can be changed



  // without re-importing a file. Tabs come from the last import if there were



  // any, otherwise the whole form is a single tab.



  function openDesigner() {



    const groups = formTabs && formTabs.length



      ? formTabs



      : [{ name: "Extraction", fieldIds: fields.map(f => f.id) }];



    setDesignOpen(true);
    setInferredTabs(groups.map(g => ({



      name: g.name,
    studies: g.studies ?? null,
    role: "extract" as SheetRole,



      fields: g.fieldIds



        .map(id => fields.find(f => f.id === id))



        .filter((f): f is FormField => !!f)



        .map(f => ({ label: f.label, type: f.type, options: f.options,



                     description: f.description || "", include: true })),



    })).filter(t => t.fields.length));



    setActiveInferTab(0);



    setPreviewing(false);



    setInferSource("");



  }




  /** Make a structural change, keeping what it replaced so it can be undone. */
  function editTabs(what: string, fn: (tabs: InferredTab[]) => InferredTab[]) {
    const before = inferredTabs || [];
    setHistory(h => [...h.slice(-29), { tabs: before, active: activeInferTab, what }]);
    setInferredTabs(fn(before));
  }

  function undo() {
    const last = history[history.length - 1];
    if (!last) return;
    setHistory(h => h.slice(0, -1));
    setInferredTabs(last.tabs);
    setActiveInferTab(Math.min(last.active, Math.max(last.tabs.length - 1, 0)));
  }

  // Cmd-Z / Ctrl-Z while the designer is open. Inside a text box the keystroke
  // belongs to the box, so it is left alone there.
  useEffect(() => {
    if (!inferredTabs) return;
    function onKey(e: KeyboardEvent) {
      if (e.key.toLowerCase() !== "z" || e.shiftKey || !(e.metaKey || e.ctrlKey)) return;
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || el?.isContentEditable) return;
      e.preventDefault();
      undo();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [inferredTabs, history, activeInferTab]);

  // A session starts with a clean history: undoing into the shape of a
  // workbook that is no longer open would restore fields nobody can see.
  useEffect(() => { if (!inferredTabs) setHistory([]); }, [inferredTabs]);

  // Renaming a tab. The ref is there because the blur that follows Enter or
  // Escape would otherwise commit the same rename a second time.
  function startRename(ti: number, name: string) {
    renameOpen.current = true;
    setRenaming({ ti, text: name });
  }

  function endRename(commit: boolean) {
    if (!renameOpen.current) return;
    renameOpen.current = false;
    const r = renaming;
    setRenaming(null);
    if (!commit || !r) return;
    const name = r.text.trim();
    // A tab with no name is a tab nobody can aim at, so blank means cancel.
    if (!name || name === (inferredTabs || [])[r.ti]?.name) return;
    editTabs("rename tab", tabs => tabs.map((t, j) => (j === r.ti ? { ...t, name } : t)));
  }

  function addTab() {


    editTabs("add tab", tabs => {


      const next = [...(tabs || []), { name: `Tab ${(tabs || []).length + 1}`,
                                       role: "extract" as SheetRole, fields: [emptyField()] }];


      setActiveInferTab(next.length - 1);


      return next;


    });


  }



  function removeTab(ti: number) {


    editTabs("delete tab", tabs => (tabs || []).filter((_, j) => j !== ti));


    setActiveInferTab(x => Math.max(0, x > ti ? x - 1 : Math.min(x, ti)));


  }



  function moveTab(ti: number, by: number) {


    editTabs("move tab", tabs => {


      const next = [...(tabs || [])];


      const to = ti + by;


      if (to < 0 || to >= next.length) return next;


      [next[ti], next[to]] = [next[to], next[ti]];


      setActiveInferTab(to);


      return next;


    });


  }



  function addFieldTo(ti: number) {


    editTabs("add field", tabs => (tabs || []).map((t, j) =>


      (j === ti ? { ...t, fields: [...t.fields, emptyField()] } : t)));


  }



  function removeDesignField(ti: number, fi: number) {


    editTabs("delete field", tabs => (tabs || []).map((t, j) =>


      (j === ti ? { ...t, fields: t.fields.filter((_, k) => k !== fi) } : t)));


  }



  function dropField(ti: number, from: number, to: number) {



    if (from === to || to < 0) return;



    editTabs("move field", tabs => (tabs || []).map((t, j) => {



      if (j !== ti) return t;



      const fs = [...t.fields];



      const [moved] = fs.splice(from, 1);



      fs.splice(from < to ? to - 1 : to, 0, moved);



      return { ...t, fields: fs };



    }));



  }




  function moveField(ti: number, fi: number, by: number) {


    editTabs("move field", tabs => (tabs || []).map((t, j) => {


      if (j !== ti) return t;


      const fs = [...t.fields];


      const to = fi + by;


      if (to < 0 || to >= fs.length) return t;


      [fs[fi], fs[to]] = [fs[to], fs[fi]];


      return { ...t, fields: fs };


    }));


  }



  // Column order is what the exported sheet looks like, so a field has to be


  // movable to the tab it belongs on rather than only deletable from this one.


  function moveFieldToTab(ti: number, fi: number, to: number) {


    editTabs("move field to another tab", tabs => {


      const next = [...(tabs || [])];


      if (!next[ti] || !next[to] || ti === to) return next;


      const field = next[ti].fields[fi];


      next[ti] = { ...next[ti], fields: next[ti].fields.filter((_, k) => k !== fi) };


      next[to] = { ...next[to], fields: [...next[to].fields, field] };


      return next;


    });


  }



  function setTabRole(ti: number, role: SheetRole) {

    editTabs("change what this tab is", tabs => applyDictionaries(

      (tabs || []).map((t, j) => (j === ti ? { ...t, role } : t)), rawSheets).tabs);

  }


  const patchTab = (ti: number, p: Partial<InferredTab>) =>
    editTabs("change which studies", tabs => (tabs || []).map((t, j) => (j === ti ? { ...t, ...p } : t)));

  function addField() {
    const id = `f_${Date.now().toString(36)}`;
    setFields(f => [...f, { id, label: "New field", type: "text", options: [] }]);
    // A field belonging to no tab would be written to no sheet, so it joins the
    // tab being edited.
    setFormTabs(ts => ts && ts.map((t, j) =>
      (j === Math.min(resultsTab, ts.length - 1) ? { ...t, fieldIds: [...t.fieldIds, id] } : t)));
  }

  const patchField = (id: string, p: Partial<FormField>) =>
    setFields(f => f.map(x => (x.id === id ? { ...x, ...p } : x)));

  function removeField(id: string) {
    setFields(f => f.filter(x => x.id !== id));
    setFormTabs(ts => ts && ts.map(t => ({ ...t, fieldIds: t.fieldIds.filter(x => x !== id) })));
  }

  /** Reorder by id. Field order is column order in the exported sheet. */
  function moveFormField(fromId: string, toId: string) {
    setFields(fs => {
      const from = fs.findIndex(f => f.id === fromId);
      const to = fs.findIndex(f => f.id === toId);
      if (from < 0 || to < 0 || from === to) return fs;
      const next = [...fs];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return next;
    });
  }

  // What gets extracted. With tabs that is whatever the tabs cover, since an
  // article no tab asks for has nowhere to be written; without tabs it is the
  // studies picked on the form itself.
  const targets = acquired.filter(p => !deselected.has(p.paper_id)
    && (!formTabs || formTabs.some(t => !t.studies || t.studies.includes(p.paper_id))));

  const formStudies = formTabs && formTabs.length
    ? (shownTab?.studies ?? null)
    : (deselected.size ? acquired.filter(p => !deselected.has(p.paper_id)).map(p => p.paper_id) : null);

  function pickFormStudies(studies: string[] | null) {
    if (formTabs && formTabs.length) {
      setFormTabs(ts => ts && ts.map((t, j) =>
        (j === Math.min(resultsTab, ts.length - 1) ? { ...t, studies } : t)));
      return;
    }
    setDeselected(studies === null
      ? new Set()
      : new Set(acquired.filter(p => !studies.includes(p.paper_id)).map(p => p.paper_id)));
  }

  // Tabs on the live form: the same three verbs as the designer. A flat form
  // becomes its own first tab rather than losing its fields to a new empty one.
  function addFormTab() {
    const ts = formTabs
      || [{ name: "Extraction", fieldIds: fields.map(f => f.id), studies: null }];
    setFormTabs([...ts, { name: `Tab ${ts.length + 1}`, fieldIds: [], studies: null }]);
    setResultsTab(ts.length);
  }

  function startFormRename(ti: number, name: string) {
    formRenameOpen.current = true;
    setFormRenaming({ ti, text: name });
  }

  function endFormRename(commit: boolean) {
    if (!formRenameOpen.current) return;
    formRenameOpen.current = false;
    const r = formRenaming;
    setFormRenaming(null);
    if (!commit || !r) return;
    const name = r.text.trim();
    if (!name || name === (formTabs || [])[r.ti]?.name) return;
    setFormTabs(ts => ts && ts.map((t, j) => (j === r.ti ? { ...t, name } : t)));
  }

  function removeFormTab(ti: number) {
    const ts = formTabs || [];
    const gone = ts[ti];
    if (!gone) return;
    const was = { fields, tabs: formTabs, at: resultsTab };
    const rest = ts.filter((_, j) => j !== ti);
    const kept = new Set(rest.flatMap(t => t.fieldIds));
    setFields(f => f.filter(x => kept.has(x.id) || !gone.fieldIds.includes(x.id)));
    setFormTabs(rest.length ? rest : null);
    setResultsTab(x => Math.max(0, x > ti ? x - 1 : Math.min(x, ti)));
    // The live form has no undo stack and a tab takes its fields with it, so
    // the way back is offered here instead.
    toast(`Deleted ${gone.name}`, {
      action: {
        label: "Undo",
        onClick: () => { setFields(was.fields); setFormTabs(was.tabs); setResultsTab(was.at); },
      },
    });
  }

  async function runForm() {
    if (!fields.length) { toast.error("Add at least one field to the form."); return; }
    if (!targets.length) { toast.error("Select at least one article to extract from."); return; }
    setFormBusy(true);
    setFormProgress({ done: 0, total: targets.length });
    setFormCurrent(targets[0]?.title || "");

    // The commonest reason a run comes back empty is a document the browser
    // only half read. Read those on the server first, images included, rather
    // than extracting from a page header.
    const textFor: Record<string, string> = {};
    for (const p of targets) textFor[p.paper_id] = (p.text || "").trim();
    for (const p of targets.filter(x => textFor[x.paper_id].length < THIN_TEXT)) {
      setFormCurrent(`Reading ${p.title || "Untitled"}\u2026`);
      const { text } = await readTextOnServer(p.paper_id);
      if (text.trim().length > textFor[p.paper_id].length) {
        textFor[p.paper_id] = text.trim();
        s.setFullTexts(prev => (prev[p.paper_id]
          ? { ...prev, [p.paper_id]: { ...prev[p.paper_id], text } }
          : prev));
        // A document nobody could read is named after its file, and a filename
        // has no spaces in it where a paper's title always does. Now that it has
        // been read, it can be called what it calls itself.
        if (!/\s/.test((p.title || "").trim())) {
          const better = guessTitle("", text.slice(0, 3000).replace(/^#+\s*/gm, ""), p.title || "");
          if (better && better !== p.title) {
            s.setRawPapers(prev => (prev
              ? prev.map(x => (x.id === p.paper_id ? { ...x, title: better } : x)) : prev));
            s.setUniquePapers(prev => (prev
              ? prev.map(x => (x.id === p.paper_id ? { ...x, title: better } : x)) : prev));
            s.setFullTexts(prev => (prev[p.paper_id]
              ? { ...prev, [p.paper_id]: { ...prev[p.paper_id], title: better } } : prev));
            s.setResults(prev => (prev
              ? prev.map(r => (r.paper_id === p.paper_id ? { ...r, Title: better } : r)) : prev));
            s.setFullTextResults(prev => (prev
              ? prev.map(r => (r.paper_id === p.paper_id ? { ...r, Title: better } : r)) : prev));
          }
        }
      }
    }

    const rows: { paper_id: string; title: string; values: Record<string, any>; meta: Record<string, { source_quote: string; confidence: string; needs_review: boolean; review_reason?: string }>; status: Record<string, "suggested" | "accepted" | "rejected">; cell: Record<string, CellStatus> }[] = [];
    let done = 0;
    // An article that fails produces no row. Silently, that reads as a paper
    // which reports nothing, so each failure is kept and reported at the end.
    const failures: { title: string; why: string }[] = [];
    // Local (Ollama) serves one request at a time, so run in series — it also
    // makes the live label exact. Cloud / GPU-served models batch well, so fan
    // out to saturate them.
    const CONCURRENCY = /^(claude|gpt|gemini)/i.test(extractModel) ? 8 : 1;
    const spec = fields.map(f => ({ id: f.id, label: f.label, type: f.type, options: f.options, description: f.description }));
    const specFor = (paperId: string) => fieldsForStudy(formTabs, spec, paperId);
    try {
      for (let i = 0; i < targets.length; i += CONCURRENCY) {
        const batch = targets.slice(i, i + CONCURRENCY);
        await Promise.all(batch.map(async p => {
          setFormCurrent(p.title || "Untitled");   // in series this is exact; concurrent shows the batch
          try {
            // Feed extracted tables too — numeric results and effect sizes live
            // in tables, not the prose. Big lever for sample-size/effect fields.
            const tables = ((p as any).tables || [])
              .map((t: any) => [t.caption, ...(t.rows || []).map((r: any[]) => r.join(" | "))].join("\n"))
              .join("\n\n");
            const values: Record<string, any> = {};
            const meta: Record<string, { source_quote: string; confidence: string; needs_review: boolean; review_reason?: string }> = {};
            const status: Record<string, "suggested" | "accepted" | "rejected"> = {};
            const cell: Record<string, CellStatus> = {};
            let callError = "";
            // In groups, so the model is answering a question it can finish,
            // and so a form of two hundred fields reports progress rather than
            // appearing to hang for minutes.
            const mySpec = specFor(p.paper_id);
            for (let k = 0; k < mySpec.length; k += FIELDS_PER_CALL) {
              const group = mySpec.slice(k, k + FIELDS_PER_CALL);
              if (mySpec.length > FIELDS_PER_CALL) {
                setFormCurrent(`${p.title || "Untitled"} · fields ${k + 1}-${k + group.length} of ${mySpec.length}`);
              }
              const rich = await AIService.extractFieldsRich(
                textFor[p.paper_id] || p.text || "", group, p.title || "", undefined, tables);
              Object.assign(values, rich.values || {});
              for (const rf of rich.fields || []) {
                meta[rf.id] = { source_quote: rf.source_quote || "", confidence: rf.confidence || "none",
                                needs_review: !!rf.needs_review, review_reason: rf.review_reason || "" };
                cell[rf.id] = (rf.status as CellStatus) || "unread";
                // "Not reported" is an answer, so it is not held for review.
                // Everything else: pre-accept confident, unflagged suggestions
                // and flag the rest, including anything nothing managed to read.
                status[rf.id] = cell[rf.id] === "not_reported"
                  ? "accepted"
                  : (rf.needs_review || rf.confidence !== "high") ? "suggested" : "accepted";
              }
              if (rich.error && !callError) callError = rich.error;
            }
            for (const f of mySpec) {
              if (f.id in status) continue;
              // Absent from every reply: nothing read this field. It is work
              // remaining, not a settled blank, which is what calling it
              // accepted used to make it look like.
              cell[f.id] = String(values[f.id] ?? "").trim() ? "reported" : "unread";
              status[f.id] = "suggested";
            }
            const filled = Object.values(values).filter(v => String(v ?? "").trim()).length;
            if (callError && !filled) failures.push({ title: p.title || "Untitled", why: callError });
            else if (!filled) {
              failures.push({
                title: p.title || "Untitled",
                why: textFor[p.paper_id].length < THIN_TEXT
                  ? "almost nothing could be read from this document"
                  : "no field could be filled from this text",
              });
            }
            rows.push({ paper_id: p.paper_id, title: p.title || "Untitled", values, meta, status, cell });
            setFormRows([...rows]);
          } catch (e: any) {
            failures.push({ title: p.title || "Untitled", why: e?.message || "the request failed" });
          } finally {
            done += 1;
            setFormProgress({ done, total: targets.length });
          }
        }));
      }
      toast.success(`Extracted the form from ${rows.length} article${rows.length === 1 ? "" : "s"}`);
      if (failures.length) {
        toast.error(
          failures.length === 1
            ? `${failures[0].title.slice(0, 60)}: ${failures[0].why}`
            : `${failures.length} articles came back empty. First: ${failures[0].title.slice(0, 40)} (${failures[0].why})`,
          { duration: 10000 },
        );
      }
      if (rows.length) setFormCollapsed(true);
    } catch (e: any) {
      toast.error(e?.message || "Form extraction failed");
    } finally {
      setFormBusy(false);
    }
  }

  // The exported shape follows the workbook that was imported, when there was
  // one: same columns, same order, so the file can be pasted back into it.
  function exportShape() {
    const flatHeaders = (sheetTabs || []).flatMap(t => t.headers);
    if (!flatHeaders.length) {
      return {
        head: ["Article", ...fields.map(f => f.label)],
        row: (r: typeof formRows[number]) =>
          [r.title, ...fields.map(f => (r.status?.[f.id] === "rejected" ? "" : (r.values[f.id] ?? "")))],
      };
    }
    const byLabel = new Map(fields.map(f => [f.label.trim().toLowerCase(), f]));
    return {
      head: ["Article", ...flatHeaders],
      row: (r: typeof formRows[number]) => [
        r.title,
        ...flatHeaders.map(h => {
          const f = byLabel.get(h.trim().toLowerCase());
          if (!f) return "";                                  // not extracted: left blank
          return r.status?.[f.id] === "rejected" ? "" : (r.values[f.id] ?? "");
        }),
      ],
    };
  }

  async function exportFormXlsx() {
    if (!formRows.length) { toast.error("Run the form first."); return; }
    const thin = { style: "thin" as const, color: { argb: "FFE2E8F0" } };
    const border = { top: thin, left: thin, bottom: thin, right: thin };
    const wb = new ExcelJS.Workbook();
    wb.creator = "Evidence Engine";

    // Unique, Excel-safe sheet names (<=31 chars, no []:*?/\).
    const used = new Set<string>();
    const sheetName = (want: string, fallback: string) => {
      const base = (want || fallback).replace(/[[\]*?/\\:]/g, " ").replace(/\s+/g, " ")
        .trim().slice(0, 28) || fallback;
      let name = base, n = 2;
      while (used.has(name.toLowerCase())) name = `${base.slice(0, 25)} ${n++}`;
      used.add(name.toLowerCase());
      return name;
    };

    // A form with tabs is a workbook, and it comes out as the workbook it is:
    // one worksheet per tab, one row per study, that tab's fields as its
    // columns. The alternative, a worksheet per study holding every field, is
    // not the form the reviewer designed and cannot be read down a column.
    if (formTabs && formTabs.length) {
      const headFill = { type: "pattern" as const, pattern: "solid" as const,
                         fgColor: { argb: "FF1F2937" } };
      tabSheets(formTabs, fields, formRows).forEach((sheet, ti) => {
        const ws = wb.addWorksheet(sheetName(sheet.name, `Tab ${ti + 1}`),
                                   { views: [{ state: "frozen", xSplit: 1, ySplit: 1 }] });
        ws.columns = [{ width: 46 }, ...sheet.head.slice(1).map(() => ({ width: 28 }))];
        const head = ws.insertRow(1, sheet.head);
        head.height = 22;
        head.eachCell(c => {
          c.font = { bold: true, color: { argb: "FFFFFFFF" }, size: 11 };
          c.fill = headFill;
          c.alignment = { vertical: "middle", wrapText: true };
          c.border = border;
        });
        sheet.body.forEach((values, i) => {
          const row = ws.addRow(values);
          row.eachCell(c => {
            c.alignment = { vertical: "top", wrapText: true };
            c.font = { size: 10 };
            c.border = border;
          });
          if (i % 2) {
            row.eachCell(c => {
              c.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF8FAFC" } };
            });
          }
        });
        ws.autoFilter = { from: { row: 1, column: 1 }, to: { row: 1, column: sheet.head.length } };
      });

      const tabbed = await wb.xlsx.writeBuffer();
      const tabbedBlob = new Blob([tabbed],
        { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(tabbedBlob);
      link.download = "extraction-form.xlsx";
      link.click();
      URL.revokeObjectURL(link.href);
      toast.success(`Exported ${formTabs.length} tab${formTabs.length === 1 ? "" : "s"}`);
      return;
    }

    const names = formRows.map((r, i) => sheetName(r.title || "", `Article ${i + 1}`));

    // ── Summary sheet: articles × fields ──
    const sum = wb.addWorksheet("Summary", { views: [{ state: "frozen", xSplit: 1, ySplit: 1 }] });
    sum.columns = [{ key: "article", width: 46 }, ...fields.map(f => ({ key: f.id, width: 28 }))];
    const shape = exportShape();
    const head = sum.insertRow(1, shape.head);
    head.height = 22;
    head.eachCell(c => {
      c.font = { bold: true, color: { argb: "FFFFFFFF" }, size: 11 };
      c.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2937" } };
      c.alignment = { vertical: "middle", wrapText: true };
      c.border = border;
    });
    formRows.forEach((r, i) => {
      // Same shape as the header above, so a sheet-derived export lines up
      // column for column with the workbook it came from.
      const row = sum.addRow(shape.row(r).map((v, j) =>
        (j === 0 ? (r.title || "Untitled") : String(v ?? "").trim())));
      row.eachCell(c => { c.alignment = { vertical: "top", wrapText: true }; c.font = { size: 10 }; c.border = border; });
      const a = row.getCell(1);
      a.value = { text: r.title || "Untitled", hyperlink: `#'${names[i]}'!A1` };
      a.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };
      if (i % 2) row.eachCell(c => { c.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF8FAFC" } }; });
    });
    sum.autoFilter = { from: { row: 1, column: 1 }, to: { row: 1, column: fields.length + 1 } };

    // ── One sheet per article: Field | Value ──
    formRows.forEach((r, i) => {
      const ws = wb.addWorksheet(names[i], { views: [{ state: "frozen", ySplit: 3 }] });
      ws.columns = [{ width: 28 }, { width: 90 }];
      ws.mergeCells("A1:B1");
      const t = ws.getCell("A1");
      t.value = r.title || "Untitled";
      t.font = { size: 13, bold: true, color: { argb: "FFFFFFFF" } };
      t.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2937" } };
      t.alignment = { wrapText: true, vertical: "middle" };
      ws.getRow(1).height = 30;
      ws.mergeCells("A2:B2");
      const back = ws.getCell("A2");
      back.value = { text: "← Back to Summary", hyperlink: "#Summary!A1" };
      back.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };
      const hr = ws.getRow(3);
      hr.getCell(1).value = "Field"; hr.getCell(2).value = "Value";
      hr.eachCell(c => {
        c.font = { bold: true, color: { argb: "FFFFFFFF" }, size: 11 };
        c.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF7C3AED" } };
        c.border = border;
      });
      fields.forEach((f, j) => {
        const row = ws.addRow([f.label, (r.status?.[f.id] === "rejected" ? "" : String(r.values[f.id] ?? "").trim()) || "—"]);
        row.getCell(1).font = { bold: true, size: 10 };
        row.getCell(1).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF1F5F9" } };
        row.getCell(2).font = { size: 10 };
        row.eachCell(c => { c.alignment = { vertical: "top", wrapText: true }; c.border = border; });
        if (j % 2) row.getCell(2).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF8FAFC" } };
      });
    });

    const buf = await wb.xlsx.writeBuffer();
    const blob = new Blob([buf], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "extraction-form.xlsx"; a.click(); URL.revokeObjectURL(a.href);
    toast.success("Exported Excel workbook");
  }

  function exportFormCsv() {
    if (!formRows.length) { toast.error("Run the form first."); return; }
    const q = (x: any) => `"${String(x ?? "").replace(/"/g, '""')}"`;
    const shape = exportShape();
    const head = shape.head;
    const rows = formRows.map(shape.row);
    const csv = [head, ...rows].map(r => r.map(q).join(",")).join("\r\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "extraction-form.csv"; a.click(); URL.revokeObjectURL(a.href);
  }

  // Import an existing extraction form.
  //
  // A JSON file already carries the field spec (label, type, options), so it is
  // loaded exactly as written. A spreadsheet carries only headers and values, so
  // reading it literally produced untyped fields with no description, which
  // extract poorly; those go through inference and a preview instead.
  async function importForm(file: File) {
    const name0 = file.name.toLowerCase();
    if (!name0.endsWith(".json")) return inferFromSheet(file);
    setSheetTabs(null);   // a JSON form carries no workbook shape to preserve
    try {
      const name = name0;
      let next: FormField[] | null = null;
      if (name.endsWith(".json")) {
        const j = JSON.parse(await file.text());
        const arr: any[] = Array.isArray(j) ? j : (Array.isArray(j?.fields) ? j.fields : []);
        next = arr.map((x, i) => typeof x === "string"
          ? { id: `${slug(x)}_${i}`, label: x, type: "text", options: [] }
          : { id: `${slug(x.label || x.name || "field")}_${i}`, label: x.label || x.name || `Field ${i + 1}`, type: x.type || "text", options: Array.isArray(x.options) ? x.options : [] });
      } else {
        let labels: string[] = [];
        if (name.endsWith(".xlsx") || name.endsWith(".xls")) {
          const wb = new ExcelJS.Workbook();
          await wb.xlsx.load(await file.arrayBuffer());
          wb.worksheets[0]?.getRow(1).eachCell(c => { const v = String(c.value ?? "").trim(); if (v) labels.push(v); });
        } else {
          const text = await file.text();
          const firstLine = text.split(/\r?\n/).find(l => l.trim()) || "";
          const sep = firstLine.includes("\t") ? "\t" : ",";
          labels = firstLine.split(sep).map(x => x.replace(/^"|"$/g, "").trim()).filter(Boolean);
        }
        // Drop a leading row-label column ("Article"/"Study"/etc.) if present.
        labels = labels.filter(l => !/^(article|articles|title|study|studies|paper|papers|id|#)$/i.test(l));
        next = labels.map((l, i) => ({ id: `${slug(l)}_${i}`, label: l, type: "text", options: [] }));
      }
      if (!next || !next.length) { toast.error("No fields found in that file."); return; }
      setFields(next);
      setFormRows([]);
      toast.success(`Imported ${next.length} field${next.length === 1 ? "" : "s"} from ${file.name}`);
    } catch {
      toast.error("Could not read that form file. Use JSON, CSV, or an Excel file whose headers are the fields.");
    }
  }

  const totalEvidence = s.textExtractions.reduce(
    (a, r) => a + (r.evidence?.length ?? r.spans.length),
    0,
  );
  const totalValues = s.textExtractions.reduce((a, r) => a + r.values.length, 0);
  const noEvidence = s.textExtractions.filter(
    r => (r.evidence?.length ?? r.spans.length) === 0,
  ).length;

  const filtered = listQ.trim()
    ? s.textExtractions.filter(r => r.title.toLowerCase().includes(listQ.toLowerCase()))
    : s.textExtractions;
  const selected = s.textExtractions.find(r => r.paper_id === selectedId) ?? s.textExtractions[0] ?? null;

  if (gate) return gate;

  return (
    <div className="space-y-3">
      {/* Mode toggle: free-text question vs a structured extraction form. */}
      <div className="inline-flex rounded-lg border bg-muted/40 p-0.5 text-sm">
        <button onClick={() => setMode("ask")} className={`relative inline-flex items-center gap-1.5 rounded-md px-3 h-8 font-medium transition-colors ${mode === "ask" ? "text-foreground" : "text-muted-foreground hover:text-foreground"}`}>
          {mode === "ask" && <motion.span layoutId="textextract-mode" transition={{ type: "spring", stiffness: 440, damping: 36, mass: 0.7 }} className="absolute inset-0 rounded-md bg-card shadow-sm" />}
          <span className="relative z-10 inline-flex items-center gap-1.5"><ScanText className="size-3.5" />Ask</span>
        </button>
        <button onClick={() => setMode("form")} className={`relative inline-flex items-center gap-1.5 rounded-md px-3 h-8 font-medium transition-colors ${mode === "form" ? "text-foreground" : "text-muted-foreground hover:text-foreground"}`}>
          {mode === "form" && <motion.span layoutId="textextract-mode" transition={{ type: "spring", stiffness: 440, damping: 36, mass: 0.7 }} className="absolute inset-0 rounded-md bg-card shadow-sm" />}
          <span className="relative z-10 inline-flex items-center gap-1.5"><FormInput className="size-3.5" />Form</span>
        </button>
      </div>

      {mode === "ask" && (<>
      {/* ── Control pane: counts + run + export ──────────────────────────── */}
      <ControlPane
        stats={s.textExtractions.length > 0 ? <>
          <InlineStat icon={FileText} value={s.textExtractions.length} label="Articles" />
          <PaneDivider />
          <InlineStat icon={QuoteIcon} tone="success" value={totalEvidence} label="Quotes" />
          <InlineStat icon={ListChecks} tone="success" value={totalValues} label="Values" />
          {noEvidence > 0 && <InlineStat icon={AlertTriangle} tone="amber" value={noEvidence} label="No evidence" />}
          {missing.length > 0 && <InlineStat icon={AlertTriangle} tone="amber" value={missing.length} label="Skipped" />}
        </> : (
          <InlineStat icon={FileText} value={acquired.length} label="Ready to extract" />
        )}
        actions={<>
          <ImportStudies stage="text-extraction" />
          {running ? (
            <Button size="sm" variant="destructive" className="h-8 shadow-sm" onClick={() => s.cancelTask("text-extract")} title="Stop extraction">
              <X className="size-3.5 mr-1.5" />Cancel
            </Button>
          ) : (
            <Button size="sm" className="h-8 shadow-sm" onClick={run}>
              <Sparkles className="size-3.5 mr-1.5" />Extract
            </Button>
          )}
          {s.textExtractions.length > 0 && (
            <>
              <Button size="sm" variant="outline" className="h-8" onClick={() => downloadTextExtractionXlsx(s.textExtractions, query)}>
                <FileSpreadsheet className="size-3.5 mr-1.5" />Excel
              </Button>
              <Button size="sm" variant="outline" className="h-8" onClick={exportJson}>
                <Download className="size-3.5 mr-1.5" />JSON
              </Button>
            </>
          )}
        </>}
      />

      {/* Ask-in-natural-language editor. */}
      <Card className="p-3 space-y-2">
        <button
          onClick={() => setAskOpen(o => !o)}
          className="flex items-center gap-2 text-sm font-medium w-full text-left min-w-0 hover:text-primary"
          title={askOpen ? "Collapse question" : "Edit question"}
        >
          <ChevronDown className={`size-4 shrink-0 transition-transform ${askOpen ? "" : "-rotate-90"}`} />
          <ScanText className="size-4 text-primary shrink-0" />
          {askOpen ? "Ask in natural language" : <span className="truncate text-muted-foreground font-normal">{query}</span>}
        </button>
        {askOpen && (
          <>
            <Textarea value={query} onChange={e => setQuery(e.target.value)} rows={2}
              placeholder="e.g. What was the primary outcome and effect size?" />
            <div className="flex flex-wrap gap-1.5">
              {PRESETS.map((p, i) => (
                <Button key={i} size="sm" variant="outline" className="h-7 text-xs" onClick={() => setQuery(p)}>{p}</Button>
              ))}
            </div>
          </>
        )}
      </Card>

      {task && task.status === "running" && (
        <TaskProgressCard
          task={task}
          title="Text extraction"
          onCancel={() => s.cancelTask("text-extract")}
        />
      )}

      {s.textExtractions.length > 0 && (
        <>
          {/* ── Two-pane: paper list (left) + selected extraction (right) ─── */}
          <div className="flex gap-4 h-[calc(100vh-17rem)] min-h-[28rem]">
            {/* LEFT: searchable paper list */}
            <Card className="w-80 shrink-0 p-0 overflow-hidden flex flex-col">
              <div className="p-2 border-b">
                <div className="relative">
                  <Search className="size-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={listQ}
                    onChange={e => setListQ(e.target.value)}
                    placeholder={`Filter ${s.textExtractions.length} articles…`}
                    className="pl-7 h-8 text-sm"
                  />
                </div>
              </div>
              <div className="overflow-auto flex-1">
                {filtered.map((r) => {
                  const active = r.paper_id === selected?.paper_id;
                  const evCount = r.evidence?.length ?? r.spans.length;
                  return (
                    <button
                      key={r.paper_id}
                      onClick={() => setSelectedId(r.paper_id)}
                      className={`block w-full text-left px-3 py-2.5 border-b hover:bg-muted/50 transition-colors ${active ? "bg-primary/10 border-l-2 border-l-primary" : "border-l-2 border-l-transparent"}`}
                    >
                      <div className="flex items-center gap-1.5 mb-1">
                        <Badge variant={evCount > 0 ? "default" : "secondary"} className="text-[10px]">{evCount} quotes</Badge>
                        <Badge variant="outline" className="text-[10px]">{r.values.length} values</Badge>
                      </div>
                      <div className="text-sm leading-snug line-clamp-2 break-words">{(r.title || "").replace(/\s+/g, " ").trim() || "Untitled"}</div>
                    </button>
                  );
                })}
                {filtered.length === 0 && (
                  <div className="p-4 text-sm text-muted-foreground">No articles match “{listQ}”.</div>
                )}
              </div>
            </Card>

            {/* RIGHT: selected paper's extraction */}
            <Card className="flex-1 min-w-0 p-0 overflow-hidden flex flex-col">
              {!selected ? (
                <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
                  Select an article on the left.
                </div>
              ) : (
                <PaperExtractionDetail
                  key={selected.paper_id}
                  result={selected}
                  fullText={s.fullTexts[selected.paper_id]?.text || ""}
                  pdfUrl={getPdfBlob(selected.paper_id) || s.fullTexts[selected.paper_id]?.pdf_url || ""}
                />
              )}
            </Card>
          </div>
        </>
      )}
      </>)}

      {mode === "form" && (
        <>
          <input type="file" ref={formFileRef} accept=".json,.csv,.tsv,.txt,.xlsx,.xls,.xlsm" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) importForm(f); e.currentTarget.value = ""; }} />

          {/* Controls live in their own pane, above the form box. */}
          {!formCollapsed && !formBusy && (
            <ControlPane
              stats={<>
                <InlineStat icon={ListChecks} value={fields.length} label="Fields" />
                <PaneDivider />
                <InlineStat icon={FileText} value={targets.length} label="Selected" hint={`of ${acquired.length}`} tone="success" />
              </>}
              actions={<>
                <Button size="sm" variant="outline" className="h-8" disabled={inferBusy}
                        onClick={() => formFileRef.current?.click()}
                        title="Build the form from a spreadsheet you already use, or load a form exported as JSON">
                  {inferBusy
                    ? <Loader2 className="size-3.5 mr-1.5 animate-spin" />
                    : <Upload className="size-3.5 mr-1.5" />}
                  Import form
                </Button>
                <Button size="sm" variant="outline" className="h-8" onClick={openDesigner}
                        disabled={!fields.length} title="Change the tabs and columns of the output workbook">
                  <Table2 className="size-3.5 mr-1.5" />Layout
                </Button>
                {/* A workbook that was imported and then left half-designed is
                    still here; offer it rather than opening it unbidden. */}
                {inferredTabs && !designOpen && (
                  <Button size="sm" variant="outline" className="h-8" onClick={() => setDesignOpen(true)}
                          title="Carry on with the workbook you imported">
                    <Table2 className="size-3.5 mr-1.5" />Resume import
                  </Button>
                )}
                <Button size="sm" variant="outline" className="h-8"
                        onClick={() => {
                          setFields(DEFAULT_FORM_FIELDS);
                          setFormRows([]);
                          setSheetTabs(null);
                          setFormTabs(null);
                          // The imported workbook, and the half-finished design
                          // of it, are part of what "reset" has to undo.
                          setInferredTabs(null);
                          setDesignOpen(false);
                          setRawSheets([]);
                          setInferSource("");
                        }}
                        title="Restore the fields this stage ships with, and forget an imported workbook">
                  <RotateCcw className="size-3.5 mr-1.5" />Reset
                </Button>
                <span className="mx-0.5 h-6 w-px bg-border" aria-hidden="true" />
                <Button size="sm" className="h-8 shadow-sm" onClick={runForm} disabled={!targets.length || !fields.length}>
                  <Table2 className="size-4 mr-1.5" />Extract from {targets.length} article{targets.length === 1 ? "" : "s"}
                </Button>
              </>}
            />
          )}

          {/* What the workbook turned into, before it replaces anything. Shown as
              tabs because a real template is several: characteristics for every
              study, then tabs for one design only, then outcomes again. */}
          <Dialog open={!!inferredTabs && designOpen}
                  onOpenChange={o => { if (!o) { setInferredTabs(null); setPreviewing(false); } }}>
            <DialogContent className="sm:max-w-[min(1200px,94vw)] max-h-[92vh] overflow-hidden flex! flex-col
                                    [&>button.absolute]:opacity-100 [&>button.absolute]:size-6
                                    [&>button.absolute]:rounded [&>button.absolute]:bg-rose-600
                                    [&>button.absolute]:text-white [&>button.absolute]:flex
                                    [&>button.absolute]:items-center [&>button.absolute]:justify-center
                                    [&>button.absolute:hover]:bg-rose-700">
              <DialogHeader>
                <DialogTitle>{previewing ? "Notebook preview" : "Fields found in your workbook"}</DialogTitle>
              </DialogHeader>
              {inferSource && <div className="text-xs text-amber-700">{inferSource}</div>}

              {/* Tab strip: the workbook's own tabs, in its own order. */}
              <div className="flex items-center gap-1 flex-wrap border-b pb-2 -mx-1 px-1">
                {/* A tab is also a drop target: dragging a field onto it moves the
                    field there, which is what the move-to-tab dropdown used to do. */}
                {(inferredTabs || []).map((t, ti) => (
                  renaming?.ti === ti ? (
                    <input key={ti} autoFocus value={renaming.text}
                           onChange={e => setRenaming({ ti, text: e.target.value })}
                           onFocus={e => e.currentTarget.select()}
                           onBlur={() => endRename(true)}
                           // Escape must not reach the dialog, which would close on it
                           // and take the whole form with it.
                           onKeyDown={e => {
                             if (e.key !== "Enter" && e.key !== "Escape") return;
                             e.stopPropagation();
                             endRename(e.key === "Enter");
                           }}
                           className="h-[26px] w-40 px-2 rounded-md border bg-background
                                      text-xs font-normal" />
                  ) : (
                  <button key={ti} onClick={() => setActiveInferTab(ti)}
                          onDoubleClick={() => startRename(ti, t.name)}
                          title="Double-click to rename"
                          onDragOver={e => { if (dragField !== null && ti !== activeInferTab) e.preventDefault(); }}
                          onDrop={e => { e.preventDefault();
                            if (dragField !== null && ti !== activeInferTab) moveFieldToTab(activeInferTab, dragField, ti);
                            setDragField(null); setDragOver(null); }}
                          className={`px-2.5 py-1 rounded-md text-xs ${ti === activeInferTab
                            ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}>
                    <span className="whitespace-nowrap">{t.name}</span>
                    {/* A skipped tab says so, because otherwise it looks like any
                        other tab while contributing nothing. No count: that is in
                        the footer, for the tab you are actually looking at. */}
                    {t.role === "other" && <span className="opacity-60 ml-1.5">skipped</span>}
                  </button>
                  )
                ))}
                {/* A workbook rarely has a tab for everything the reviewer wants. */}
                <button onClick={addTab} title="Add a tab"
                        className="px-2.5 py-1 rounded-md text-xs text-muted-foreground
                                   hover:bg-muted hover:text-foreground">
                  + Tab
                </button>
                <button onClick={undo} disabled={!history.length}
                        title={history.length
                          ? `Undo ${history[history.length - 1].what} (${UNDO_KEY})`
                          : "Nothing to undo"}
                        className="ml-auto order-last inline-flex items-center gap-1 px-2 py-1 rounded-md
                                   text-xs text-muted-foreground hover:bg-muted hover:text-foreground
                                   disabled:opacity-40 disabled:hover:bg-transparent">
                  <Undo2 className="size-3.5" />Undo
                </button>
                <span className="mx-1 h-5 w-px bg-border" aria-hidden="true" />
                <button onClick={() => setPreviewing(p => !p)}
                        className={`px-2.5 py-1 rounded-md text-xs ${previewing
                          ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}>
                  Preview
                </button>
                {previewing && (
                  <div className="inline-flex items-center rounded-md border text-xs overflow-hidden ml-1">
                    {(["file", "output"] as const).map(k => (
                      <button key={k} onClick={() => setPreviewShowing(k)}
                              className={`px-2 py-1 text-xs ${previewShowing === k
                                ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/50"}`}>
                        {k === "file" ? "File" : "Output"}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="flex-1 min-h-0 overflow-y-auto -mx-1 px-1">

              {previewing ? <WorkbookPreview tab={(inferredTabs || [])[activeInferTab]} showing={previewShowing}
                               studies={acquired} sheets={rawSheets} /> : (
                <>
                {(() => { const tab = (inferredTabs || [])[activeInferTab]; return tab ? (
                  <>
                  {/* One bar: what the tab is, and who it is for. Two stacked dropdown rows
                      above every field list made the dialog feel heavy. The rule half only
                      appears for a tab that holds study data, since a dictionary has none. */}
                  <div className="flex items-center gap-2 flex-wrap text-xs bg-muted/40 rounded-md px-2 py-1.5">
                    <Popover>
                      <PopoverTrigger asChild>
                        <button className={PICKER_TRIGGER}>
                          {ROLE_LABEL[tab.role]}
                          <ChevronDown className="size-3.5 opacity-60" />
                        </button>
                      </PopoverTrigger>
                      <PopoverContent align="start" className="w-52 p-1">
                        {(Object.keys(ROLE_LABEL) as SheetRole[]).map(r => (
                          <PopoverClose asChild key={r}>
                            <button onClick={() => setTabRole(activeInferTab, r)}
                                    className={`${PICKER_ROW} ${tab.role === r ? "bg-muted font-medium" : ""}`}>
                              {ROLE_LABEL[r]}
                            </button>
                          </PopoverClose>
                        ))}
                      </PopoverContent>
                    </Popover>
                  
                    {tab.role === "extract" && (
                      <Popover>
                        <PopoverTrigger asChild>
                          <button className={PICKER_TRIGGER}>
                            Studies
                            <ChevronDown className="size-3.5 opacity-60" />
                          </button>
                        </PopoverTrigger>
                        <PopoverContent align="start" className="w-80 p-0">
                          <StudyPicker studies={acquired} chosen={tab.studies ?? null}
                                       onChange={patch => patchTab(activeInferTab, patch)} />
                        </PopoverContent>
                      </Popover>

                  
                    )}
                    {/* Tab order is sheet order in the exported workbook, so it has to be
                        changeable here rather than only in the file it came from. */}
                    <span className="ml-auto flex items-center gap-0.5 text-muted-foreground">
                      <button onClick={() => moveTab(activeInferTab, -1)} title="Move tab left"
                              className="px-1 hover:text-foreground disabled:opacity-30"
                              disabled={activeInferTab === 0}>&lsaquo;</button>
                      <button onClick={() => moveTab(activeInferTab, 1)} title="Move tab right"
                              className="px-1 hover:text-foreground disabled:opacity-30"
                              disabled={activeInferTab >= (inferredTabs || []).length - 1}>&rsaquo;</button>
                      <button onClick={() => removeTab(activeInferTab)} title="Delete tab"
                              className="px-1 hover:text-rose-600">&times;</button>
                    </span>
                  </div>
                  
                  {tab.role !== "extract" ? (
                    tab.role === "dictionary" ? (
                      <DictionarySheet rows={(rawSheets.find(sh => sh.name === sheetOf(tab))?.rows) || []} />
                    ) : (
                      <div className="text-xs text-muted-foreground px-2 py-3">
                        No fields are taken from this tab.
                      </div>
                    )
                  ) : (
                  <>

                  <div className="space-y-0.5">
                    {tab.fields.map((f, i2) => (
                      <div key={i2}
                           draggable
                           onDragStart={e => {
                             e.dataTransfer.setData("text/plain", String(i2));
                             e.dataTransfer.effectAllowed = "move";
                             dragGhost(e.currentTarget as HTMLElement, e);
                             setDragField(i2);
                           }}
                           onDragEnd={() => { setDragField(null); setDragOver(null); }}
                           onDragOver={e => { e.preventDefault(); setDragOver(i2); }}
                           onDrop={e => { e.preventDefault();
                             if (dragField !== null) dropField(activeInferTab, dragField, i2);
                             setDragField(null); setDragOver(null); }}
                           className={`group relative flex items-start gap-2 rounded-md border border-border bg-card px-2 py-2 transition-[opacity,box-shadow] duration-150 ${dragField === i2 ? "opacity-50 shadow-sm" : ""} ${dragOver === i2 && dragField !== null && dragField !== i2 ? "before:absolute before:left-0 before:right-0 before:-top-[3px] before:h-0.5 before:rounded-full before:bg-primary" : ""}`}>
                        {/* The grip is both the handle and the thing that separates one row
                            from the next: a hover tint alone is nothing to aim at in a list
                            of seventy fields. */}
                        <GripVertical className="size-4 mt-1.5 shrink-0 text-muted-foreground/40 cursor-grab active:cursor-grabbing" />
                        <Checkbox className="mt-2" checked={f.include}
                                  onCheckedChange={v => patchInferred(i2, { include: !!v })} />
                        <div className="min-w-0 flex-1 space-y-1.5">
                          <div className="flex items-center gap-1.5">
                            <Input draggable={false} value={f.label} className="h-8 text-xs font-medium" placeholder="Field name"
                                   onChange={e => patchInferred(i2, { label: e.target.value })} />
                            <Select value={f.type} onValueChange={v => patchInferred(i2, { type: v })}>
                              <SelectTrigger className="h-7 w-28 text-xs shrink-0"><SelectValue /></SelectTrigger>
                              <SelectContent>
                                <SelectItem value="text">text</SelectItem>
                                <SelectItem value="number">number</SelectItem>
                                <SelectItem value="select">select</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <Input draggable={false} value={f.description} className="h-8 text-xs"
                                 placeholder="What to look for, and where in the paper"
                                 onChange={e => patchInferred(i2, { description: e.target.value })} />
                          {f.type === "select" && (
                            <Input draggable={false} value={f.options.join(", ")} className="h-8 text-xs"
                                   placeholder="Allowed values, comma separated"
                                   onChange={e => patchInferred(i2, {
                                     options: e.target.value.split(",").map(x => x.trim()).filter(Boolean) })} />
                          )}
                        </div>
                        {/* Hidden until hover: four controls on every row turned the list
                            into a toolbar. */}
                        <button onClick={() => removeDesignField(activeInferTab, i2)} title="Delete field"

                                className="mt-1.5 px-1 text-muted-foreground hover:text-rose-600 opacity-0 group-hover:opacity-100">

                          &times;

                        </button>
                      </div>
                    ))}
                  </div>
                  <button onClick={() => addFieldTo(activeInferTab)}
                          className="text-xs text-muted-foreground hover:text-foreground px-2 py-1">
                    + Add field
                  </button>
                  </>
                  )}
                  </>
                ) : null; })()}
                </>
              )}

              </div>


              {/* The count sits in the footer rather than over the list: the ticks
                  are already visible there, and a total belongs with the action. */}
              <div className="flex items-center justify-between -mx-6 -mb-6 px-6 pb-6 pt-3 border-t">
                <span className="text-xs text-muted-foreground">
                  {(() => {
                    const t = (inferredTabs || [])[activeInferTab];
                    return t && t.role === "extract"
                      ? `${t.fields.filter(f => f.include).length} of ${t.fields.length} fields`
                      : "";
                  })()}
                </span>


                <Button size="sm" className="text-xs" onClick={() => applyInferred()}


                        disabled={!(inferredTabs || []).some(t => t.role === "extract" && t.fields.some(f => f.include))}>


                  Use this form


                </Button>


              </div>
            </DialogContent>
          </Dialog>

          {/* Reading a workbook is one model call per sheet, so the wait is real and

              worth showing honestly: which sheet, and how many are left. */}

          {inferProgress && (

            <Card className="p-3">

              <div className="flex items-center justify-between text-xs mb-1.5">

                <span className="flex items-center gap-2 min-w-0">
                  <Loader2 className="size-3.5 animate-spin shrink-0" />
                  <span className="truncate">{inferProgress.name}</span>
                </span>

                <span className="flex items-center gap-2 shrink-0 ml-3">
                <span className="text-muted-foreground">{inferProgress.done} of {inferProgress.total}</span>
                <button onClick={() => inferAbort.current?.abort()} title="Cancel import"
                        aria-label="Cancel import"
                        className="inline-flex items-center justify-center size-5 rounded
                                   bg-rose-600 text-white hover:bg-rose-700 leading-none">
                  &times;
                </button>
              </span>

              </div>
                        <div className="h-1.5 rounded-full bg-muted overflow-hidden">

                <div className="h-full bg-primary transition-[width] duration-300"

                     style={{ width: `${Math.round((inferProgress.done / Math.max(inferProgress.total, 1)) * 100)}%` }} />

              </div>

            </Card>

          )}


          {/* Collapsible form builder. Auto-collapses after a run so the results
              take the screen; expand the header to edit fields / selection. */}
          <Card className="p-3 space-y-2">
            <div className="flex items-center justify-between gap-2 flex-wrap">
              <button onClick={() => setFormCollapsed(c => !c)} className="flex items-center gap-2 text-sm font-medium min-w-0 hover:text-primary">
                {formCollapsed ? <ChevronRight className="size-4 shrink-0" /> : <ChevronDown className="size-4 shrink-0" />}
                <FormInput className="size-4 text-primary shrink-0" />Extraction form
              </button>
            </div>

            {formBusy && (
              <div className="space-y-1.5 pt-0.5">
                <div className="h-2 rounded-full bg-muted overflow-hidden">
                  <div className="h-full rounded-full bg-gradient-to-r from-primary/70 to-primary transition-[width] duration-500 ease-out" style={{ width: `${formProgress.total ? (formProgress.done / formProgress.total) * 100 : 0}%` }} />
                </div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground min-w-0">
                  <span className="inline-flex items-center gap-1.5 min-w-0 flex-1">
                    <Loader2 className="size-3 animate-spin shrink-0" />
                    <span className="truncate">{formCurrent ? <>Extracting <span className="text-foreground/70">{formCurrent}</span></> : "Extracting…"}</span>
                  </span>
                  <span className="tabular-nums shrink-0">{formProgress.done} / {formProgress.total}</span>
                </div>
              </div>
            )}

            {!formCollapsed && !formBusy && (
            <div className="space-y-3 pt-1">

            {/* The same strip as the workbook designer, because these are the same
                tabs: the one picked here is the one whose fields you edit and whose
                results show below. Double-click a tab to rename it. */}
            <div className="flex items-center gap-1 flex-wrap border-b pb-2">
              {(formTabs || []).map((t, ti) => (
                formRenaming?.ti === ti ? (
                  <input key={ti} autoFocus value={formRenaming.text}
                         onChange={e => setFormRenaming({ ti, text: e.target.value })}
                         onFocus={e => e.currentTarget.select()}
                         onBlur={() => endFormRename(true)}
                         onKeyDown={e => {
                           if (e.key !== "Enter" && e.key !== "Escape") return;
                           e.stopPropagation();
                           endFormRename(e.key === "Enter");
                         }}
                         className="h-[26px] w-40 px-2 rounded-md border bg-background
                                    text-xs font-normal" />
                ) : (
                <button key={ti} onClick={() => setResultsTab(ti)}
                        onDoubleClick={() => startFormRename(ti, t.name)}
                        title="Double-click to rename"
                        className={`px-2.5 py-1 rounded-md text-xs ${t === shownTab
                          ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}>
                  {t.name}
                </button>
                )
              ))}
              <button onClick={addFormTab} title="Add a tab"
                      className="px-2.5 py-1 rounded-md text-xs text-muted-foreground
                                 hover:bg-muted hover:text-foreground">
                + Tab
              </button>
              <span className="ml-auto flex items-center gap-1">
                <Popover>
                  <PopoverTrigger asChild>
                    <button className={PICKER_TRIGGER}>
                      Studies
                      <ChevronDown className="size-3.5 opacity-60" />
                    </button>
                  </PopoverTrigger>
                  <PopoverContent align="end" className="w-80 p-0">
                    <StudyPicker studies={acquired} chosen={formStudies}
                                 onChange={patch => pickFormStudies(patch.studies)} />
                  </PopoverContent>
                </Popover>
                {formTabs && formTabs.length > 0 && (
                  <button onClick={() => removeFormTab(Math.min(resultsTab, formTabs.length - 1))}
                          title="Delete this tab and its fields"
                          className="px-1 text-muted-foreground hover:text-rose-600">&times;</button>
                )}
              </span>
            </div>

            <div className="space-y-2">
                <div className="space-y-0.5">
                  {shownFields.map(f => (
                    <div key={f.id}
                         draggable
                         onDragStart={e => {
                           e.dataTransfer.setData("text/plain", f.id);
                           e.dataTransfer.effectAllowed = "move";
                           dragGhost(e.currentTarget as HTMLElement, e);
                           setDragForm(f.id);
                         }}
                         onDragEnd={() => { setDragForm(null); setDragFormOver(null); }}
                         onDragOver={e => { e.preventDefault(); setDragFormOver(f.id); }}
                         onDrop={e => { e.preventDefault();
                           if (dragForm) moveFormField(dragForm, f.id);
                           setDragForm(null); setDragFormOver(null); }}
                         className={`group relative flex items-start gap-2 rounded-md border border-border bg-card px-2 py-2 transition-[opacity,box-shadow] duration-150 ${dragForm === f.id ? "opacity-50 shadow-sm" : ""} ${dragFormOver === f.id && dragForm && dragForm !== f.id ? "before:absolute before:left-0 before:right-0 before:-top-[3px] before:h-0.5 before:rounded-full before:bg-primary" : ""}`}>
                      <GripVertical className="size-4 mt-1.5 shrink-0 text-muted-foreground/40 cursor-grab active:cursor-grabbing" />
                      <div className="min-w-0 flex-1 space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <Input draggable={false} value={f.label} className="h-8 text-xs font-medium"
                                 placeholder="Field name"
                                 onChange={e => patchField(f.id, { label: e.target.value })} />
                          <Select value={f.type} onValueChange={v => patchField(f.id, { type: v })}>
                            <SelectTrigger className="h-7 w-28 text-xs shrink-0"><SelectValue /></SelectTrigger>
                            <SelectContent>
                              <SelectItem value="text">text</SelectItem>
                              <SelectItem value="number">number</SelectItem>
                              <SelectItem value="category">category</SelectItem>
                              <SelectItem value="date">date</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        {/* The hint used to hide behind a button. It is the field's
                            whole instruction to the extractor, so it is always here. */}
                        <Input draggable={false} value={f.description || ""} className="h-8 text-xs"
                               placeholder="What to look for, and where in the paper"
                               onChange={e => patchField(f.id, { description: e.target.value })} />
                        {f.type === "category" && (
                          <Input draggable={false} value={(f.options || []).join(", ")} className="h-8 text-xs"
                                 placeholder="Allowed values, comma separated"
                                 onChange={e => patchField(f.id, {
                                   options: e.target.value.split(",").map(x => x.trim()).filter(Boolean) })} />
                        )}
                      </div>
                      <button onClick={() => removeField(f.id)} title="Delete field"
                              className="mt-1.5 px-1 text-muted-foreground hover:text-rose-600
                                         opacity-0 group-hover:opacity-100">
                        &times;
                      </button>
                    </div>
                  ))}
                </div>
                <button onClick={addField}
                        className="text-xs text-muted-foreground hover:text-foreground px-2 py-1">
                  + Add field
                </button>
            </div>
            </div>
            )}
          </Card>

          {/* Where a value came from. */}
          <Dialog open={!!source} onOpenChange={o => { if (!o) setSource(null); }}>
            <DialogContent className="sm:max-w-[min(900px,92vw)] max-h-[85vh] overflow-y-auto">
              {(() => {
                if (!source) return null;
                const row = formRows.find(r => r.paper_id === source.paperId);
                const field = fields.find(f => f.id === source.fieldId);
                if (!row || !field) return null;
                const meta = row.meta?.[field.id];
                const value = String(row.values[field.id] ?? "").trim();
                const quote = (meta?.source_quote || "").trim();
                const cellState = row.cell?.[field.id];
                const text = s.fullTexts[row.paper_id]?.text || "";
                const ctx = quote ? quoteInContext(text, quote) : null;
                const at = quote ? quotePosition(text, quote) : null;
                return (
                  <>
                    <DialogHeader>
                      <DialogTitle>{field.label}</DialogTitle>
                      <DialogDescription>{row.title}</DialogDescription>
                    </DialogHeader>

                    <div className="rounded-lg border bg-muted/30 px-3 py-2 space-y-2">
                      <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Value</div>
                      {/* Editable here, where the quote is: correcting a number
                          while looking at the sentence it was read from is the
                          whole job, and it should not need a second screen. */}
                      <textarea
                        value={draft}
                        onChange={e => setDraft(e.target.value)}
                        rows={Math.min(6, Math.max(1, draft.split("\n").length))}
                        placeholder="nothing was extracted"
                        className="w-full rounded border bg-background px-2 py-1.5 text-sm font-normal" />
                      <div className="flex items-center gap-2">
                        <Button size="sm" className="h-7 text-xs"
                                disabled={draft.trim() === String(value).trim()}
                                onClick={() => {
                                  setFieldValue(row.paper_id, field.id, draft.trim());
                                  setSource(null);
                                }}>
                          Save
                        </Button>
                        {meta?.edited && (
                          // What it restores belongs on the button, not in it.
                          <Button size="sm" variant="outline" className="h-7 text-xs"
                                  title={`Put back what the model had: ${String(meta.original || "nothing").slice(0, 80)}`}
                                  onClick={() => { revertField(row.paper_id, field.id); setSource(null); }}>
                            <RotateCcw className="size-3.5 mr-1.5" />Undo
                          </Button>
                        )}
                      </div>
                    </div>

                    {/* Why the app flagged a value it did fill. Without this the
                        reviewer sees a cell marked for review, a quote that is
                        plainly in the paper, and no way to tell what the app
                        thinks is wrong with it. */}
                    {meta?.review_reason && (
                      <div className="rounded-md border border-amber-300/70 bg-amber-50
                                      dark:border-amber-800/50 dark:bg-amber-950/30
                                      px-2.5 py-2 text-xs leading-snug
                                      text-amber-800 dark:text-amber-200">
                        Check this one: {meta.review_reason}.
                      </div>
                    )}

                    {/* An empty cell has three quite different meanings and the
                        reviewer has to be able to tell them apart. */}
                    {!quote && (
                      <div className="text-sm text-muted-foreground">
                        {/* The run says which of the two empty cases this is.
                            Rows extracted before it did fall back to whether
                            the field has any metadata, which is the same
                            distinction read indirectly. */}
                        {(cellState ? cellState === "unread" : !meta)
                          ? "This field was never answered: the group of fields it was asked in failed. Run the form again to fill it."
                          : value
                            ? "The model gave no sentence for this value, so there is nothing to check it against. Treat it as unverified."
                            : "The model read the article and found nothing for this field. That the paper does not report it is itself a finding, and it is recorded as a dash rather than left blank."}
                      </div>
                    )}

                    {quote && (
                      <>
                        {/* Nothing is said when the sentence checks out: the
                            highlighted page is the answer, and a line of
                            metadata above it is noise. The failure still has to
                            be said, because an unverified value looks exactly
                            like a verified one. */}
                        {!ctx?.found && (
                          <div className="flex items-center gap-2 text-xs text-rose-600">
                            <span className="inline-block size-1.5 rounded-full bg-rose-500" />
                            This sentence is not in the document. The value is unverified.
                          </div>
                        )}

                        {/* The page itself when the file is here, because a
                            sentence in its own column, under its own table, is
                            what a reviewer checks a number against. The text
                            passage stays underneath: it is always available,
                            and it is what the exported audit trail quotes. */}
                        {hasPdfBlob(row.paper_id) && (
                          <PdfQuotePage paperId={row.paper_id} quote={quote} />
                        )}

                        <div className="rounded-lg border p-3 text-sm leading-relaxed whitespace-pre-wrap break-words">
                          {ctx?.found ? (
                            <>
                              <span className="text-muted-foreground">{ctx.before}</span>
                              <mark className="bg-amber-200/70 dark:bg-amber-500/30 rounded px-0.5">
                                {ctx.match}
                              </mark>
                              <span className="text-muted-foreground">{ctx.after}</span>
                            </>
                          ) : (
                            <span className="italic">“{quote}”</span>
                          )}
                        </div>
                      </>
                    )}
                  </>
                );
              })()}
            </DialogContent>
          </Dialog>

          {/* Results matrix */}
          {formRows.length > 0 && (
            <Card className="p-0 overflow-hidden">
              <div className="flex items-center justify-between gap-2 px-4 py-2.5 border-b bg-gradient-to-b from-muted/40 to-transparent">
                <button onClick={() => setResultsCollapsed(c => !c)} className="flex items-center gap-2 text-sm font-medium min-w-0 hover:text-primary" title={resultsCollapsed ? "Expand results" : "Collapse results"}>
                  {resultsCollapsed ? <ChevronRight className="size-4 shrink-0" /> : <ChevronDown className="size-4 shrink-0" />}
                  <span>Results <span className="text-xs text-muted-foreground font-normal">· {formRows.length} article{formRows.length === 1 ? "" : "s"} × {fields.length} field{fields.length === 1 ? "" : "s"}{(() => { const nr = formRows.reduce((n, r) => n + Object.values(r.status || {}).filter(x => x === "suggested").length, 0); return nr ? ` · ${nr} to review` : ""; })()}</span></span>
                </button>
                <div className="flex items-center gap-2">
                  <Button size="sm" className="h-8 shadow-sm" onClick={exportFormXlsx}><FileSpreadsheet className="size-3.5 mr-1.5" />Excel</Button>
                  <Button size="sm" variant="outline" className="h-8" onClick={exportFormCsv}><Download className="size-3.5 mr-1.5" />CSV</Button>
                </div>
              </div>
              {!resultsCollapsed && formTabs && formTabs.length > 1 && (
                <div className="flex items-center gap-1 flex-wrap px-4 py-2 border-b bg-muted/20">
                  {formTabs.map((t, ti) => (
                    <button key={ti} onClick={() => setResultsTab(ti)}
                            className={`px-2.5 py-1 rounded-md text-xs ${ti === resultsTab
                              ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}>
                      {t.name}
                    </button>
                  ))}
                </div>
              )}
              {!resultsCollapsed && (
              <div className="overflow-auto max-h-[30rem]">
                <table className="w-full text-xs border-collapse">
                  <thead className="bg-muted sticky top-0 z-10 [&_th]:text-[11px] [&_th]:font-semibold [&_th]:uppercase [&_th]:tracking-wider [&_th]:text-muted-foreground">
                    <tr>
                      <th className="px-3 py-2 text-left sticky left-0 bg-muted border-b border-r min-w-[220px] max-w-[220px] z-30">Article</th>
                      {shownFields.map(f => <th key={f.id} className="px-3 py-2 text-left border-b whitespace-nowrap">{f.label}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {shownRows.map(r => (
                      <tr key={r.paper_id} className="border-b border-border/60 last:border-0 align-top hover:bg-muted transition-colors bg-card">
                        <td className="px-3 py-2 sticky left-0 z-10 bg-card border-r min-w-[220px] max-w-[220px] align-top">
                          <div className="font-medium leading-snug break-words">{r.title}</div>
                        </td>
                        {shownFields.map(f => {
                          const st = r.status?.[f.id] || "accepted";
                          const m = r.meta?.[f.id];
                          const val = String(r.values[f.id] ?? "").trim();
                          const conf = m?.confidence || "none";
                          const dot = conf === "high" ? "bg-emerald-500" : conf === "low" ? "bg-amber-500" : "bg-muted-foreground/30";
                          return (
                            <td key={f.id} className={`px-3 py-2 align-top whitespace-pre-wrap break-words min-w-[150px] ${st === "suggested" ? "bg-amber-50/60 dark:bg-amber-950/20" : ""}`}>
                              <div className="flex items-start gap-1.5">
                                <span className={`mt-1 size-1.5 rounded-full shrink-0 ${m?.edited ? "bg-sky-500" : dot}`}
                                      title={m?.edited
                                        ? `You edited this. The model had: ${m.original || "(nothing)"}`
                                        : r.cell?.[f.id] === "not_reported"
                                          ? "The model read the article and found no value for this field."
                                          : r.cell?.[f.id] === "unread"
                                            ? "Nothing read this field yet."
                                            : `AI confidence: ${conf}${m?.source_quote ? `\nSource: "${m.source_quote}"` : ""}`} />
                                <div className="flex-1 min-w-0">
                                  {editing && editing.paperId === r.paper_id && editing.fieldId === f.id ? (
                                    <textarea
                                      autoFocus
                                      value={editing.text}
                                      onChange={e => setEditing({ ...editing, text: e.target.value })}
                                      onBlur={() => {
                                        setFieldValue(r.paper_id, f.id, editing.text.trim());
                                        setEditing(null);
                                      }}
                                      onKeyDown={e => {
                                        if (e.key === "Escape") { e.stopPropagation(); setEditing(null); }
                                        // Enter commits; shift-Enter is a new line, because some
                                        // of these fields hold a sentence.
                                        if (e.key === "Enter" && !e.shiftKey) {
                                          e.preventDefault();
                                          (e.currentTarget as HTMLTextAreaElement).blur();
                                        }
                                      }}
                                      rows={Math.min(5, Math.max(1, editing.text.split("\n").length))}
                                      className="w-full rounded border bg-background px-1.5 py-1 text-xs
                                                 font-normal leading-normal" />
                                  ) : (
                                  <button type="button"
                                          onClick={() => { setDraft(val); setSource({ paperId: r.paper_id, fieldId: f.id }); }}
                                          onDoubleClick={() => setEditing({ paperId: r.paper_id, fieldId: f.id, text: val })}
                                          title="Click for the source, double-click to edit"
                                          // text-xs font-normal because the base layer gives a
                                          // bare button 16px medium, which would make every cell
                                          // in the table bigger than the table it sits in.
                                          className={`block w-full text-left text-xs font-normal leading-normal
                                                      text-foreground/90 hover:underline
                                                      decoration-dotted underline-offset-2
                                                      ${st === "rejected" ? "line-through opacity-50" : ""}`}>
                                    {/* Three states, not two, but a column of
                                        "Not reported" is unreadable next to a
                                        hundred others. Both empty cases show a
                                        dash; the one the model answered is the
                                        darker of the two, and the hover text
                                        and the source dialog say which it is. */}
                                    {val || (
                                      <span className={r.cell?.[f.id] === "not_reported"
                                        ? "text-muted-foreground"
                                        : "text-muted-foreground/40"}>{NOT_REPORTED}</span>
                                    )}
                                  </button>
                                  )}
                                  {/* Both verbs on every cell, whatever the model
                                      thought of it. A value it was confident about
                                      can still be wrong, and a reviewer who spots
                                      one should not have to work out why this cell
                                      offers no way to say so. */}
                                  <div className="mt-1 flex items-center gap-1.5">
                                    <button type="button"
                                            onClick={() => setFieldStatus(r.paper_id, f.id, "accepted")}
                                            title={st === "accepted" ? "Kept" : "Keep this value"}
                                            aria-pressed={st === "accepted"}
                                            // Plain green, no box: which cells still want a
                                            // decision is already said by the amber tint on the
                                            // cell and its "review" tag, and a second marker on
                                            // the settled ones is noise on every row.
                                            className="inline-flex items-center text-emerald-600 hover:text-emerald-700">
                                      <Check className="size-3.5" />
                                    </button>
                                    <button type="button"
                                            onClick={() => setFieldStatus(r.paper_id, f.id, "rejected")}
                                            title={st === "rejected" ? "Rejected" : "Reject this value"}
                                            aria-pressed={st === "rejected"}
                                            className="inline-flex items-center text-rose-600 hover:text-rose-700">
                                      <X className="size-3.5" />
                                    </button>
                                    {st === "suggested" && (
                                      <span className="text-[9px] uppercase tracking-wide text-amber-600 dark:text-amber-500">review</span>
                                    )}
                                  </div>
                                </div>
                              </div>
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function PaperExtractionDetail({ result, fullText, pdfUrl = "" }: { result: TextExtractionResult; fullText: string; pdfUrl?: string }) {
  // Evidence may live on the new `evidence[]` field OR fall back to the
  // legacy `spans[]`. Normalise into TextEvidenceItem so the renderer below
  // doesn't have to branch.
  const evidence: TextEvidenceItem[] = result.evidence?.length
    ? result.evidence
    : result.spans.map((sp, i) => ({
        quote: fullText.slice(sp.start, sp.end),
        section: sp.label,
        start: sp.start,
        end: sp.end,
        why: `Match ${i + 1}`,
      }));

  const fullTextRef = useRef<HTMLDivElement | null>(null);
  const maxTextRef = useRef<HTMLDivElement | null>(null);
  const [activeSpan, setActiveSpan] = useState<[number, number] | null>(null);
  const [maxOpen, setMaxOpen] = useState(false);
  // Default to the PDF when a paper is opened; "Locate" flips to the text view.
  const [viewMode, setViewMode] = useState<"pdf" | "text">(pdfUrl ? "pdf" : "text");

  // Scroll to a span inside whichever viewer is currently visible (inline, or
  // the maximized dialog).
  function scrollToSpan(ref: React.RefObject<HTMLDivElement | null>, start: number, end: number) {
    setActiveSpan([start, end]);
    // Defer scroll until React renders the highlight class on the new active span.
    requestAnimationFrame(() => {
      const el = ref.current?.querySelector<HTMLElement>(`[data-span-start="${start}"]`);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }
  function locate(start: number, end: number) {
    // Char-offset jumping only works in the extracted-text view, so switch to it.
    if (viewMode !== "text") setViewMode("text");
    scrollToSpan(maxOpen ? maxTextRef : fullTextRef, start, end);
  }

  return (
    <>
      <div className="border-b p-4">
        <div className="font-medium leading-snug">{result.title}</div>
        <div className="flex items-center gap-2 mt-2">
          <Badge variant={evidence.length > 0 ? "default" : "secondary"}>{evidence.length} quotes</Badge>
          <Badge variant="outline">{result.values.length} values</Badge>
        </div>
      </div>
      <div className="flex-1 overflow-auto">
        <div className="p-4 space-y-4">
            {/* ANSWER ------------------------------------------------------ */}
            {(result.answer || result.summary) && (
              <div className="bg-primary/5 border border-primary/30 rounded-md p-3">
                <div className="text-xs font-medium uppercase tracking-wide text-primary mb-1">Answer</div>
                <div className="text-sm leading-relaxed text-foreground/90">
                  {result.answer || result.summary}
                </div>
              </div>
            )}

            {evidence.length === 0 ? (
              <div className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded p-3">
                <AlertTriangle className="size-4 inline mr-1" />
                No relevant passages found for this query. The information may not be in this article.
              </div>
            ) : (
              <>
                <div className={result.values.length > 0 ? "grid lg:grid-cols-2 gap-4 items-start" : ""}>
                {/* EVIDENCE CARDS -------------------------------------------- */}
                <div>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
                    Evidence quotes ({evidence.length})
                  </div>
                  <div className="space-y-2">
                    {evidence.map((e, i) => (
                      <div key={i} className="border rounded-md p-3 bg-card hover:border-primary/40 transition-colors">
                        <div className="flex items-start justify-between gap-3 mb-2">
                          <div className="flex items-center gap-2 text-xs">
                            <span className="text-muted-foreground tabular-nums">[{i + 1}]</span>
                            {e.section && (
                              <Badge variant="outline" className={`text-[10px] font-medium border ${sectionBadgeClass(e.section)}`}>
                                {e.section}
                              </Badge>
                            )}
                            <span className="text-muted-foreground tabular-nums">char {e.start}-{e.end}</span>
                          </div>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => locate(e.start, e.end)}
                            className="h-6 px-2 text-xs"
                            title="Scroll to this passage in the full text below"
                          >
                            <MapPin className="size-3 mr-1" /> Locate
                          </Button>
                        </div>
                        <blockquote className="text-sm border-l-2 border-primary/40 pl-3 italic text-foreground/90 break-words">
                          <QuoteIcon className="size-3 inline mr-1 text-muted-foreground" />
                          {e.quote}
                        </blockquote>
                        {e.why && (
                          <div className="text-xs text-muted-foreground mt-2">{e.why}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* EXTRACTED VALUES ------------------------------------------ */}
                {result.values.length > 0 && (
                  <div>
                    <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
                      Extracted values ({result.values.length})
                    </div>
                    <div className="grid grid-cols-1 gap-2">
                      {result.values.map((v, i) => (
                        <div
                          key={i}
                          className={`border rounded p-2 text-sm bg-card transition-colors ${
                            v.start !== undefined ? "hover:border-primary/40 cursor-pointer" : ""
                          }`}
                          onClick={() => {
                            if (v.start !== undefined && v.end !== undefined) {
                              locate(v.start, v.end);
                            }
                          }}
                          title={v.start !== undefined ? "Click to locate in text" : undefined}
                        >
                          <div className="flex items-center gap-2 flex-wrap">
                            <Badge variant="outline" className="text-[10px]">{v.field}</Badge>
                            {v.section && (
                              <Badge variant="outline" className={`text-[10px] font-medium border ${sectionBadgeClass(v.section)}`}>
                                {v.section}
                              </Badge>
                            )}
                            <span className="font-mono text-sm">{v.value}</span>
                          </div>
                          {v.quote && (
                            <div className="text-xs italic text-muted-foreground mt-1">"…{v.quote}…"</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                </div>

                {/* SOURCE VIEWER: PDF (default) or extracted text ----------- */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="inline-flex rounded-md border bg-muted/40 p-0.5 text-xs">
                      <button
                        onClick={() => pdfUrl && setViewMode("pdf")}
                        disabled={!pdfUrl}
                        className={`px-2.5 py-1 rounded transition-colors ${viewMode === "pdf" ? "bg-background shadow-sm font-medium" : "text-muted-foreground"} ${!pdfUrl ? "opacity-40 cursor-not-allowed" : ""}`}
                        title={pdfUrl ? "View the source PDF" : "No PDF available for this article"}
                      >
                        PDF
                      </button>
                      <button
                        onClick={() => setViewMode("text")}
                        className={`px-2.5 py-1 rounded transition-colors ${viewMode === "text" ? "bg-background shadow-sm font-medium" : "text-muted-foreground"}`}
                        title="View the extracted text with jump-to-passage"
                      >
                        Extracted text
                      </button>
                    </div>
                    {viewMode === "text" && (
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={() => setMaxOpen(true)}
                        className="size-7 text-muted-foreground hover:text-foreground"
                        title="Maximize full text"
                      >
                        <Maximize2 className="size-4" />
                      </Button>
                    )}
                  </div>
                  {viewMode === "pdf" && pdfUrl ? (
                    <iframe src={pdfUrl} title="Source PDF" className="w-full h-96 rounded-md border bg-muted/20" />
                  ) : (
                    <div
                      ref={fullTextRef}
                      className="rounded-md border bg-muted/20 text-xs font-mono leading-relaxed max-h-96 overflow-auto"
                    >
                      <FullTextViewer text={fullText} evidence={evidence} activeSpan={activeSpan} />
                    </div>
                  )}
                </div>
              </>
            )}
        </div>
      </div>

      {/* Full-screen full-text viewer. */}
      <Dialog open={maxOpen} onOpenChange={setMaxOpen}>
        <DialogContent className="max-w-[99vw] w-[99vw] h-[97vh] sm:max-w-[99vw] flex flex-col p-3 gap-2">
          <DialogHeader className="pr-8">
            <DialogTitle className="text-base leading-snug">
              Full text · <span className="font-normal text-muted-foreground">{result.title}</span>
            </DialogTitle>
          </DialogHeader>
          {evidence.length > 0 && (
            <div className="flex flex-wrap gap-1.5 shrink-0">
              {evidence.map((e, i) => (
                <Button
                  key={i}
                  size="sm"
                  variant="outline"
                  className="h-6 px-2 text-xs"
                  onClick={() => scrollToSpan(maxTextRef, e.start, e.end)}
                  title={e.quote}
                >
                  <MapPin className="size-3 mr-1" />[{i + 1}]{e.section ? ` ${e.section}` : ""}
                </Button>
              ))}
            </div>
          )}
          <div
            ref={maxTextRef}
            className="flex-1 min-h-0 overflow-auto rounded-md border bg-muted/20 text-sm font-mono leading-relaxed"
          >
            <FullTextViewer text={fullText} evidence={evidence} activeSpan={activeSpan} />
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

/** Renders the full text with line numbers, highlighting every evidence span.
 *  Each span carries `data-span-start={start}` so the parent can scrollIntoView.
 *  Lines that contain at least one highlighted span are nudged a bit so the
 *  reader can see the line number in the gutter. */
function FullTextViewer({
  text, evidence, activeSpan,
}: {
  text: string;
  evidence: TextEvidenceItem[];
  activeSpan: [number, number] | null;
}) {
  if (!text) return <div className="p-3 text-muted-foreground">No full text available.</div>;

  // Sort spans by start offset for the cursor-based renderer.
  const spans = [...evidence].sort((a, b) => a.start - b.start);

  // Split into lines and remember each line's character range so we can
  // overlay highlights line-by-line.
  const lines: { text: string; start: number; end: number; idx: number }[] = [];
  let lineStart = 0;
  text.split(/(\n)/).forEach((part) => {
    if (part === "\n") {
      // Treat the newline as part of the preceding line for offset purposes.
      const last = lines[lines.length - 1];
      if (last) last.end += 1;
      lineStart += 1;
      return;
    }
    if (!part && lines.length > 0) return;
    lines.push({ text: part, start: lineStart, end: lineStart + part.length, idx: lines.length + 1 });
    lineStart += part.length;
  });

  return (
    <table className="w-full border-collapse">
      <tbody>
        {lines.map((ln) => {
          // Restrict spans that overlap this line.
          const lineSpans = spans.filter(s => s.start < ln.end && s.end > ln.start);
          return (
            <tr key={ln.idx} className="align-top">
              <td className="select-none text-right pr-2 pl-2 py-0.5 text-muted-foreground/60 bg-muted/30 border-r tabular-nums w-10">
                {ln.idx}
              </td>
              <td className="py-0.5 px-3 whitespace-pre-wrap break-words">
                {lineSpans.length === 0
                  ? ln.text
                  : renderLineWithHighlights(ln.text, ln.start, lineSpans, activeSpan)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function renderLineWithHighlights(
  lineText: string,
  lineStart: number,
  spans: TextEvidenceItem[],
  activeSpan: [number, number] | null,
): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  // Local offsets within this line, clipped to the line bounds.
  const sorted = spans
    .map(s => ({
      ...s,
      localStart: Math.max(0, s.start - lineStart),
      localEnd:   Math.max(0, s.end   - lineStart),
    }))
    .sort((a, b) => a.localStart - b.localStart);

  sorted.forEach((s, i) => {
    if (s.localStart > cursor) {
      parts.push(<span key={`pre${i}`}>{lineText.slice(cursor, s.localStart)}</span>);
    }
    const isActive = activeSpan && s.start === activeSpan[0] && s.end === activeSpan[1];
    parts.push(
      <mark
        key={`mark${i}`}
        data-span-start={s.start}
        className={
          isActive
            ? "bg-amber-300 ring-2 ring-amber-500 rounded px-0.5 dark:bg-amber-700/60"
            : "bg-yellow-200 dark:bg-yellow-900/50 rounded px-0.5"
        }
        title={s.section ? `${s.section}, chars ${s.start}-${s.end}` : `chars ${s.start}-${s.end}`}
      >
        {lineText.slice(s.localStart, Math.min(lineText.length, s.localEnd))}
      </mark>
    );
    cursor = Math.min(lineText.length, s.localEnd);
  });
  if (cursor < lineText.length) {
    parts.push(<span key="tail">{lineText.slice(cursor)}</span>);
  }
  return parts;
}


// ---- XLSX export ----------------------------------------------------------
//
// Produces a formatted workbook with four kinds of sheets:
//   1. "Summary"      : one row per paper with link to its per-paper sheet
//   2. "All values"   : flat list of every extracted value across papers
//   3. "All evidence" : flat list of every evidence quote across papers
//   4. one sheet per paper with Answer / Evidence / Values stacked
// Section badges in the workbook share the same colour palette as the on-
// screen badges so the export reads like a continuation of the UI.

const SECTION_FILL: Record<string, string> = {
  Abstract:     "FFEDE9FE",    // violet-100
  Background:   "FFDBEAFE",    // blue-100
  Introduction: "FFDBEAFE",
  Methods:      "FFE0F2FE",    // sky-100
  Results:      "FFDCFCE7",    // emerald-100
  Discussion:   "FFFEF3C7",    // amber-100
  Conclusion:   "FFFEF3C7",
  Limitations:  "FFFFEDD5",    // orange-100
  References:   "FFF1F5F9",    // slate-100
};
const SECTION_TEXT: Record<string, string> = {
  Abstract:     "FF5B21B6",
  Background:   "FF1E40AF",
  Introduction: "FF1E40AF",
  Methods:      "FF075985",
  Results:      "FF065F46",
  Discussion:   "FF92400E",
  Conclusion:   "FF92400E",
  Limitations:  "FF9A3412",
  References:   "FF475569",
};

function _styleSectionCell(cell: ExcelJS.Cell, section?: string) {
  if (!section) {
    cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF1F5F9" } };
    cell.font = { size: 10, color: { argb: "FF475569" } };
    return;
  }
  const fg = SECTION_FILL[section] || "FFF1F5F9";
  const tx = SECTION_TEXT[section] || "FF475569";
  cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: fg } };
  cell.font = { size: 10, bold: true, color: { argb: tx } };
  cell.alignment = { vertical: "middle", horizontal: "center", wrapText: false };
}

function _styleHeader(row: ExcelJS.Row) {
  row.height = 26;
  row.eachCell((cell) => {
    cell.font = { size: 11, bold: true, color: { argb: "FFFFFFFF" } };
    cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2937" } };
    cell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };
    cell.border = { bottom: { style: "thin", color: { argb: "FF374151" } } };
  });
}

function _safeSheetName(name: string, taken: Set<string>): string {
  let base = (name || "Paper").replace(/[:\\/?*[\]]/g, " ").trim().slice(0, 28) || "Paper";
  let candidate = base;
  let i = 2;
  while (taken.has(candidate.toLowerCase())) {
    const suffix = ` (${i})`;
    candidate = base.slice(0, 31 - suffix.length) + suffix;
    i += 1;
  }
  taken.add(candidate.toLowerCase());
  return candidate;
}

async function downloadTextExtractionXlsx(results: TextExtractionResult[], query: string) {
  if (!results || results.length === 0) {
    toast.error("Nothing to export. Run extraction first.");
    return;
  }

  const wb = new ExcelJS.Workbook();
  wb.creator = "Evidence Engine";
  wb.created = new Date();

  const taken = new Set<string>(["summary", "all values", "all evidence"]);
  const sheetForPaper = new Map<string, string>();

  // ---- 1. Summary sheet --------------------------------------------------
  const summary = wb.addWorksheet("Summary", {
    views: [{ state: "frozen", ySplit: 1 }],
  });
  summary.columns = [
    { header: "Paper",            key: "title",    width: 60 },
    { header: "Query",            key: "query",    width: 50 },
    { header: "Answer",           key: "answer",   width: 70 },
    { header: "Evidence quotes",  key: "n_ev",     width: 16 },
    { header: "Extracted values", key: "n_val",    width: 16 },
    { header: "Worksheet",        key: "link",     width: 24 },
  ];
  _styleHeader(summary.getRow(1));

  results.forEach((r) => {
    const sheetName = _safeSheetName(r.title, taken);
    sheetForPaper.set(r.paper_id, sheetName);
    const evCount = r.evidence?.length ?? r.spans.length;
    const row = summary.addRow({
      title:  r.title,
      query:  r.query,
      answer: r.answer || r.summary,
      n_ev:   evCount,
      n_val:  r.values.length,
      link:   sheetName,
    });
    row.eachCell((cell) => {
      cell.alignment = { vertical: "top", wrapText: true };
      cell.font = { size: 10 };
    });
    const linkCell = row.getCell("link");
    linkCell.value = { text: `Open → ${sheetName}`, hyperlink: `#'${sheetName}'!A1` } as any;
    linkCell.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };

    // Tint Evidence + Values counts.
    const evCell = row.getCell("n_ev");
    evCell.alignment = { vertical: "top", horizontal: "center" };
    evCell.fill = {
      type: "pattern", pattern: "solid",
      fgColor: { argb: evCount > 0 ? "FFDCFCE7" : "FFFEE2E2" },
    };
    evCell.font = {
      size: 10, bold: true,
      color: { argb: evCount > 0 ? "FF065F46" : "FF991B1B" },
    };
    const valCell = row.getCell("n_val");
    valCell.alignment = { vertical: "top", horizontal: "center" };
    valCell.fill = {
      type: "pattern", pattern: "solid",
      fgColor: { argb: r.values.length > 0 ? "FFDCFCE7" : "FFFEE2E2" },
    };
    valCell.font = {
      size: 10, bold: true,
      color: { argb: r.values.length > 0 ? "FF065F46" : "FF991B1B" },
    };

    row.height = 60;
  });
  summary.autoFilter = { from: { row: 1, column: 1 }, to: { row: 1, column: 6 } };

  // Query banner at the very top so reviewers can see what was asked.
  summary.insertRow(1, [`Question asked across all papers:  ${query}`]);
  summary.mergeCells(1, 1, 1, 6);
  const banner = summary.getCell(1, 1);
  banner.font = { size: 11, italic: true, color: { argb: "FF374151" } };
  banner.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF1F5F9" } };
  banner.alignment = { vertical: "middle", horizontal: "left", wrapText: true };
  summary.getRow(1).height = 28;
  // Re-style header row 2 after insertion.
  _styleHeader(summary.getRow(2));
  summary.views = [{ state: "frozen", ySplit: 2 }];
  summary.autoFilter = { from: { row: 2, column: 1 }, to: { row: 2, column: 6 } };

  // ---- 2. "All values": flat list ---------------------------------------
  const allValues = wb.addWorksheet("All values", {
    views: [{ state: "frozen", ySplit: 1 }],
  });
  allValues.columns = [
    { header: "Paper",   key: "paper",   width: 50 },
    { header: "Field",   key: "field",   width: 22 },
    { header: "Value",   key: "value",   width: 28 },
    { header: "Section", key: "section", width: 14 },
    { header: "Quote",   key: "quote",   width: 60 },
  ];
  _styleHeader(allValues.getRow(1));
  for (const r of results) {
    for (const v of r.values) {
      const row = allValues.addRow({
        paper:   r.title,
        field:   v.field,
        value:   v.value,
        section: v.section || "",
        quote:   v.quote || "",
      });
      row.eachCell((c) => {
        c.alignment = { vertical: "top", wrapText: true };
        c.font = { size: 10 };
      });
      row.getCell("value").font = { size: 10, name: "Menlo, Consolas, monospace" };
      _styleSectionCell(row.getCell("section"), v.section);
    }
  }
  allValues.autoFilter = { from: { row: 1, column: 1 }, to: { row: 1, column: 5 } };

  // ---- 3. "All evidence": flat list -------------------------------------
  const allEv = wb.addWorksheet("All evidence", {
    views: [{ state: "frozen", ySplit: 1 }],
  });
  allEv.columns = [
    { header: "Paper",         key: "paper",   width: 50 },
    { header: "Section",       key: "section", width: 14 },
    { header: "Char range",    key: "range",   width: 14 },
    { header: "Quote",         key: "quote",   width: 80 },
    { header: "Why it matters", key: "why",    width: 50 },
  ];
  _styleHeader(allEv.getRow(1));
  for (const r of results) {
    const ev = r.evidence?.length ? r.evidence : r.spans.map((sp, i) => ({
      quote: "(legacy span)",
      section: sp.label,
      start: sp.start,
      end: sp.end,
      why: `Match ${i + 1}`,
    } as TextEvidenceItem));
    for (const e of ev) {
      const row = allEv.addRow({
        paper:   r.title,
        section: e.section || "",
        range:   `${e.start}-${e.end}`,
        quote:   e.quote,
        why:     e.why || "",
      });
      row.eachCell((c) => {
        c.alignment = { vertical: "top", wrapText: true };
        c.font = { size: 10 };
      });
      _styleSectionCell(row.getCell("section"), e.section);
    }
  }
  allEv.autoFilter = { from: { row: 1, column: 1 }, to: { row: 1, column: 5 } };

  // ---- 4. Per-paper sheets ----------------------------------------------
  for (const r of results) {
    const sheetName = sheetForPaper.get(r.paper_id)!;
    const ws = wb.addWorksheet(sheetName, { views: [{ state: "frozen", ySplit: 1 }] });

    // Banner with paper title
    const titleRow = ws.addRow([r.title]);
    titleRow.height = 30;
    ws.mergeCells(titleRow.number, 1, titleRow.number, 5);
    const titleCell = titleRow.getCell(1);
    titleCell.font = { size: 13, bold: true, color: { argb: "FFFFFFFF" } };
    titleCell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2937" } };
    titleCell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };

    // Query
    const qRow = ws.addRow([`Question: ${r.query}`]);
    ws.mergeCells(qRow.number, 1, qRow.number, 5);
    const qCell = qRow.getCell(1);
    qCell.font = { size: 10, italic: true, color: { argb: "FF374151" } };
    qCell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };

    // Back-to-summary
    const backRow = ws.addRow(["← Back to Summary"]);
    ws.mergeCells(backRow.number, 1, backRow.number, 5);
    const back = backRow.getCell(1);
    back.value = { text: "← Back to Summary", hyperlink: "#Summary!A1" } as any;
    back.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };

    ws.addRow([]);

    // Answer panel
    const ansHeader = ws.addRow(["Answer"]);
    ws.mergeCells(ansHeader.number, 1, ansHeader.number, 5);
    const ah = ansHeader.getCell(1);
    ah.font = { size: 11, bold: true, color: { argb: "FFFFFFFF" } };
    ah.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF7C3AED" } };
    ah.alignment = { vertical: "middle", horizontal: "left" };
    const ansBody = ws.addRow([r.answer || r.summary || "(no answer)"]);
    ws.mergeCells(ansBody.number, 1, ansBody.number, 5);
    const ab = ansBody.getCell(1);
    ab.alignment = { vertical: "top", wrapText: true };
    ab.font = { size: 10 };
    ansBody.height = 60;

    ws.addRow([]);

    // Evidence section
    const ev = r.evidence?.length ? r.evidence : r.spans.map((sp, i) => ({
      quote: "(legacy span)",
      section: sp.label,
      start: sp.start,
      end: sp.end,
      why: `Match ${i + 1}`,
    } as TextEvidenceItem));

    if (ev.length > 0) {
      const evHeaderRow = ws.addRow(["#", "Section", "Char range", "Quote", "Why it matters"]);
      _styleHeader(evHeaderRow);
      ws.getColumn(1).width = 4;
      ws.getColumn(2).width = 14;
      ws.getColumn(3).width = 14;
      ws.getColumn(4).width = 70;
      ws.getColumn(5).width = 46;
      ev.forEach((e, i) => {
        const r = ws.addRow([i + 1, e.section || "", `${e.start}-${e.end}`, e.quote, e.why || ""]);
        r.eachCell((c) => {
          c.alignment = { vertical: "top", wrapText: true };
          c.font = { size: 10 };
        });
        r.getCell(1).alignment = { vertical: "top", horizontal: "center" };
        _styleSectionCell(r.getCell(2), e.section);
        r.height = 56;
      });
    } else {
      const noEv = ws.addRow(["No evidence quotes found."]);
      ws.mergeCells(noEv.number, 1, noEv.number, 5);
      noEv.getCell(1).font = { size: 10, italic: true, color: { argb: "FF6B7280" } };
    }

    ws.addRow([]);
    ws.addRow([]);

    // Values section
    if (r.values.length > 0) {
      const valHeaderRow = ws.addRow(["#", "Field", "Value", "Section", "Quote"]);
      _styleHeader(valHeaderRow);
      r.values.forEach((v, i) => {
        const row = ws.addRow([i + 1, v.field, v.value, v.section || "", v.quote || ""]);
        row.eachCell((c) => {
          c.alignment = { vertical: "top", wrapText: true };
          c.font = { size: 10 };
        });
        row.getCell(1).alignment = { vertical: "top", horizontal: "center" };
        row.getCell(3).font = { size: 10, name: "Menlo, Consolas, monospace" };
        _styleSectionCell(row.getCell(4), v.section);
        row.height = 36;
      });
    } else {
      const noVals = ws.addRow(["No extracted values."]);
      ws.mergeCells(noVals.number, 1, noVals.number, 5);
      noVals.getCell(1).font = { size: 10, italic: true, color: { argb: "FF6B7280" } };
    }
  }

  const buf = await wb.xlsx.writeBuffer();
  const blob = new Blob([buf], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  a.href = url;
  a.download = `text-extraction-${stamp}.xlsx`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  const evCount = results.reduce((acc, r) => acc + (r.evidence?.length ?? r.spans.length), 0);
  const valCount = results.reduce((acc, r) => acc + r.values.length, 0);
  toast.success(`Exported ${results.length} articles (${evCount} quotes, ${valCount} values) to XLSX.`);
}

// What the exported workbook will look like, before anything is extracted.
//
// The rules only become real once extraction has run, because a tab gated on
// "Study design is RCT" cannot know which studies qualify until that field has
// been read. So this shows the structure honestly: every tab, its columns in
// the workbook's own order, and who it is for, with the row count given as a
// certainty for all-studies tabs and as a condition for the rest.
// What a dictionary tab holds, as the file holds it. Its rows are already
// folded into the other tabs' descriptions, so this is a reference view: editing
// a definition here would not change the field it filled in, which lives on the
// tab that owns it.
function DictionarySheet({ rows }: { rows: string[][] }) {
  if (!rows.length) {
    return <div className="text-xs text-muted-foreground px-2 py-3">Nothing was read from this sheet.</div>;
  }
  const [head, ...body] = rows;
  return (
    <div className="border rounded-md overflow-x-auto">
      <table className="text-xs w-full border-collapse">
        <thead className="sticky top-0 z-10">
          <tr className="bg-muted">
            {head.map((c, i) => (
              <th key={i} className="text-left font-semibold px-2 py-1.5 border-b border-r
                                     align-bottom min-w-[10rem]">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, ri) => (
            <tr key={ri} className={ri % 2 ? "bg-muted/20" : ""}>
              {/* Keyed off the header so a short row stays in its columns. */}
              {head.map((_, ci) => (
                <td key={ci} className="px-2 py-1 border-b border-r align-top
                                        max-w-[24rem] break-words">{row[ci] || ""}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function WorkbookPreview({ tab, studies, sheets, showing }: {
  // The tab comes from the dialog's own strip: a second strip in here was the
  // same navigation twice, one row above the other.
  tab: InferredTab | undefined;
  studies: { paper_id: string; title: string }[];
  sheets: { name: string; rows: string[][] }[];
  showing: "file" | "output";
}) {
  if (!tab) return null;
  const cols = tab.fields.filter(f => f.include);
  const sheet = sheets.find(sh => sh.name === sheetOf(tab));

  // What was actually read out of the file, rather than what will be written
  // into it. This is the view that answers "did it see all my columns", which
  // an empty grid of the output shape cannot.
  const fileRows = sheet ? sheet.rows : [];

  return (
    <div className="space-y-2">
      <div className="border rounded-md overflow-x-auto">
        {showing === "output" && tab.role !== "extract" ? (

          <div className="p-4 text-xs text-muted-foreground">

            {tab.role === "dictionary"

              ? "This tab defines the fields on the other tabs, so it is not written to the output workbook."

              : "This tab is skipped, so it is not written to the output workbook."}

          </div>

        ) : showing === "file" ? (
          <table className="text-xs w-full border-collapse">
            <tbody>
              {fileRows.map((row, ri) => (
                <tr key={ri} className={ri % 2 ? "bg-muted/20" : ""}>
                  <td className="px-1.5 py-1 border-b border-r text-muted-foreground/60 text-right w-10 select-none">
                    {ri + 1}
                  </td>
                  {row.map((cell, ci) => (
                    <td key={ci} className="px-2 py-1 border-b border-r align-top min-w-[8rem] max-w-[20rem] break-words">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
              {!fileRows.length && (
                <tr><td className="px-2 py-3 text-muted-foreground">Nothing was read from this sheet.</td></tr>
              )}
            </tbody>
          </table>
        ) : (
          <table className="text-xs w-full border-collapse">
            <thead className="sticky top-0 z-10">
              <tr className="bg-muted">
                <th className="text-left font-semibold px-2 py-1.5 border-b border-r sticky left-0 bg-muted z-30 min-w-[16rem] align-bottom">
                  {/* Not the articles in the uploaded file: those are what the reviewer
                      extracted before. These are the studies in THIS review that have full
                      text, which is what will actually be filled in. */}
                  Article in this review
                </th>
                {cols.map((f, i) => (
                  <th key={i}
                      className="text-left font-semibold px-2 py-1.5 border-b border-r align-bottom
                                 min-w-[9rem] max-w-[18rem] break-words">
                    <span className="block leading-snug">{f.label}</span>
                    {f.type !== "text" && (
                      <span className="block font-normal text-muted-foreground">{f.type}</span>
                    )}
                  </th>
                ))}
                {!cols.length && (
                  <th className="text-left px-2 py-1.5 border-b text-muted-foreground font-normal">
                    No fields selected on this tab
                  </th>
                )}
              </tr>
            </thead>
            <tbody>

              {studies.filter(p => !tab.studies || tab.studies.includes(p.paper_id)).map((p, ri) => (
                <tr key={p.paper_id} className={ri % 2 ? "[&>td:not(:first-child)]:bg-muted/20" : ""}>
                  <td className="px-2 py-1 border-b border-r sticky left-0 z-10 bg-card
                                 min-w-[16rem] max-w-[22rem] align-top">
                    {/* bg-inherit here resolved to the row's translucent zebra tint, so
                        the column scrolling underneath showed through the titles. */}
                    <span className="block leading-snug break-words">{p.title || "Untitled"}</span>
                  </td>
                  {cols.map((_, i) => (
                    <td key={i} className="px-2 py-1 border-b border-r">&nbsp;</td>
                  ))}
                </tr>
              ))}
              {!studies.length && (
                <tr><td className="px-2 py-3 text-muted-foreground" colSpan={cols.length + 1}>
                  No articles with full text yet, so there is nothing to fill in.
                </td></tr>
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// Which studies a tab covers: all of them, or a hand-picked set.
function StudyPicker({ studies, chosen, onChange }: {
  studies: { paper_id: string; title: string }[];
  chosen: string[] | null;
  onChange: (patch: { studies: string[] | null }) => void;
}) {
  const [filter, setFilter] = useState("");
  const has = (id: string) => !chosen || chosen.includes(id);
  const shown = studies.filter(a =>
    !filter.trim() || (a.title || "").toLowerCase().includes(filter.trim().toLowerCase()));

  function toggle(id: string) {
    const base = chosen || studies.map(a => a.paper_id);
    const next = base.includes(id) ? base.filter(x => x !== id) : [...base, id];
    // Every study is the same thing as no choice at all, and stays that way as
    // studies are added later.
    onChange({ studies: next.length === studies.length ? null : next });
  }

  return (
    <div className="flex flex-col max-h-[min(28rem,70vh)] text-xs">
      <div className="shrink-0 p-2 border-b">
        <Input value={filter} onChange={e => setFilter(e.target.value)}
               placeholder="Find a study" className="h-7 text-xs" />
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto p-1">
        {shown.map(a => (
          <label key={a.paper_id} title={a.title || "Untitled"}
                 className="flex items-start gap-2 px-2 py-1.5 rounded-md text-xs font-normal
                            hover:bg-muted cursor-pointer">
            <Checkbox className="mt-0.5" checked={has(a.paper_id)}
                      onCheckedChange={() => toggle(a.paper_id)} />
            {/* Two lines, then an ellipsis: a title that runs to five lines leaves
                room for two studies in the whole list. The rest is on the tooltip. */}
            <span className="leading-snug line-clamp-2">{a.title || "Untitled"}</span>
          </label>
        ))}
        {!shown.length && (
          <div className="px-2 py-3 text-xs text-muted-foreground">
            {studies.length ? "Nothing matches that." : "No articles with full text yet."}
          </div>
        )}
      </div>
    </div>
  );
}
