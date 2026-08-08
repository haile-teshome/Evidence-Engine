// Risk-of-bias traffic-light figure and matrix export. Pure and
// dependency-free: it produces a self-contained SVG (the publishable figure)
// and a CSV of the study x domain judgments. Original implementation; the
// colour tiers follow the standard low / some-concerns / high convention.

export type BiasRow = { label: string; judgments: string[]; overall: string };
export type BiasData = { domainNames: string[]; rows: BiasRow[] };

type Tier = { color: string; symbol: string };
const TIER: Record<string, Tier> = {
  "low": { color: "#16a34a", symbol: "+" },
  "some concerns": { color: "#f59e0b", symbol: "?" },
  "unclear": { color: "#f59e0b", symbol: "?" },
  "moderate": { color: "#f59e0b", symbol: "?" },
  "high": { color: "#dc2626", symbol: "−" },
  "serious": { color: "#dc2626", symbol: "−" },
  "critical": { color: "#7f1d1d", symbol: "×" },
  "no information": { color: "#94a3b8", symbol: "?" },
  "not applicable": { color: "#e2e8f0", symbol: "" },
};
function tier(j: string): Tier {
  return TIER[(j || "").toLowerCase()] || { color: "#94a3b8", symbol: "?" };
}

function esc(s: string): string {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function biasCsv(data: BiasData): string {
  const cols = ["Study", ...data.domainNames, "Overall"];
  const q = (s: string) => `"${String(s ?? "").replace(/"/g, '""')}"`;
  const lines = [cols.map(q).join(",")];
  for (const r of data.rows) lines.push([r.label, ...r.judgments, r.overall].map(q).join(","));
  return lines.join("\r\n");
}

export function biasPlotSvg(data: BiasData, title = "Risk of bias"): string {
  const cell = 30, labelW = 260, pad = 16;
  const nCol = data.domainNames.length + 1; // domains + Overall
  const gridTop = 96;
  const gridLeft = pad + labelW;
  const gridW = gridLeft + nCol * cell + pad;
  const legendW = pad + 4 * 190 + 24;                                  // 4 judgment keys
  const domNameW = pad + 30 + Math.max(0, ...data.domainNames.map((n, i) => (`D${i + 1}: ${n}`).length)) * 6.3;
  const width = Math.ceil(Math.max(gridW, legendW, domNameW));
  const gridBottom = gridTop + data.rows.length * cell;
  const domLegendTop = gridBottom + 26;
  const judgLegendTop = domLegendTop + 14 + data.domainNames.length * 14 + 12;
  const height = judgLegendTop + 24;
  const cx = (c: number) => gridLeft + c * cell + cell / 2;
  const cy = (r: number) => gridTop + r * cell + cell / 2;

  const S: string[] = [];
  S.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" font-family="ui-sans-serif, system-ui, sans-serif">`);
  S.push(`<rect width="${width}" height="${height}" fill="#ffffff"/>`);
  S.push(`<text x="${pad}" y="26" font-size="15" font-weight="700" fill="#0f172a">${esc(title)}</text>`);

  // Rotated column headers: D1..Dn, then Overall.
  const headers = [...data.domainNames.map((_, i) => `D${i + 1}`), "Overall"];
  headers.forEach((h, c) => {
    const x = cx(c), y = gridTop - 8;
    S.push(`<text x="${x}" y="${y}" font-size="11" fill="#334155" text-anchor="start" transform="rotate(-45 ${x} ${y})">${esc(h)}</text>`);
  });

  data.rows.forEach((r, ri) => {
    S.push(`<text x="${gridLeft - 10}" y="${cy(ri) + 4}" font-size="11" fill="#0f172a" text-anchor="end">${esc(r.label.slice(0, 46))}</text>`);
    [...r.judgments, r.overall].forEach((j, c) => {
      const t = tier(j);
      S.push(`<circle cx="${cx(c)}" cy="${cy(ri)}" r="11" fill="${t.color}"/>`);
      if (t.symbol) S.push(`<text x="${cx(c)}" y="${cy(ri) + 4}" font-size="12" fill="#ffffff" text-anchor="middle" font-weight="700">${t.symbol}</text>`);
    });
  });

  // Domain key (D1 = full name), stacked so long names never overflow.
  S.push(`<text x="${pad}" y="${domLegendTop}" font-size="11" font-weight="600" fill="#334155">Domains</text>`);
  data.domainNames.forEach((n, i) => {
    S.push(`<text x="${pad}" y="${domLegendTop + 14 + i * 14}" font-size="10.5" fill="#475569">D${i + 1}: ${esc(n)}</text>`);
  });

  // Judgment key.
  const leg: [string, string][] = [["#16a34a", "Low"], ["#f59e0b", "Some concerns / moderate"], ["#dc2626", "High / serious"], ["#94a3b8", "No information"]];
  leg.forEach(([color, label], i) => {
    const lx = pad + i * 190, ly = judgLegendTop;
    S.push(`<circle cx="${lx + 7}" cy="${ly - 4}" r="7" fill="${color}"/><text x="${lx + 20}" y="${ly}" font-size="10.5" fill="#475569">${esc(label)}</text>`);
  });

  S.push(`</svg>`);
  return S.join("");
}
