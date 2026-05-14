import { apiFetch } from "./supabaseClient";

export type SessionMeta = { id: string; title: string; updated_at: string; created_at: string };
export type SessionData = {
  history: any[];
  pico: any;
  inclusion: string[];
  exclusion: string[];
  query: string;
  unifiedSearchQuery: string;
  perDbQueries: Record<string, string>;
  sources: string[];
  numPerSource: number;
  model: string;
  rawPapers: any[] | null;
  uniquePapers: any[] | null;
  duplicatesCount: number;
  qualityReports: any[] | null;
  excludedByQuality: string[];
  rerankThreshold?: number;
  rerankResults?: any | null;
  results: any[] | null;
  fullTextResults: any[] | null;
  snowballResults: any[] | null;
  snowballScreened: any[] | null;
  extractedPapers: any[] | null;
  prisma: any;
};

export async function listSessions(): Promise<SessionMeta[]> {
  const r = await apiFetch("/sessions");
  return r.sessions || [];
}
export async function loadSession(id: string): Promise<{ id: string; title: string; data: SessionData }> {
  const r = await apiFetch(`/sessions/${id}`);
  return r.session;
}
export async function saveSession(id: string, title: string, data: SessionData) {
  const r = await apiFetch(`/sessions/${id}`, { method: "PUT", body: JSON.stringify({ title, data }) });
  return r.session;
}
export async function deleteSession(id: string) {
  await apiFetch(`/sessions/${id}`, { method: "DELETE" });
}
