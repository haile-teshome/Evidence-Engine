# ============================================================================
# FILE: review_pack.py
# A reviewer's screening workload as one self-contained HTML file.
# ============================================================================
"""Why this exists: invite links need the app to be reachable at an address the reviewer can
open. A locally run instance has none, so the link is useless however it is delivered. A pack
removes the dependency entirely: the lead emails one file, the reviewer opens it in any browser
with nothing installed, screens offline, and exports a decisions file to send back. The export
is a project bundle, so the lead imports it with the Import bundle button that already exists.

The file embeds its data as JSON and carries no external references, so it works from a mail
attachment, a USB stick, or a Downloads folder with no network at all. Progress is kept in the
browser's localStorage, keyed by project and reviewer, so closing the tab does not lose work.
"""
from __future__ import annotations

import html
import json
from typing import Optional

BUNDLE_VERSION = 1
BUNDLE_KIND = "evidence-engine-project"


def _esc(v) -> str:
    return html.escape("" if v is None else str(v))


def build_pack(project: dict, papers: list, reviewer_email: Optional[str], reviewer_id: str,
               stage: str = "abstract", blinded: bool = True) -> str:
    """One HTML document: criteria, the papers to screen, and an export button."""
    pico = project.get("pico") or {}
    data = {
        "kind": BUNDLE_KIND,
        "bundle_version": BUNDLE_VERSION,
        "project_id": project.get("id"),
        "project_name": project.get("name") or "Systematic review",
        "stage": stage,
        "reviewer_user_id": reviewer_id,
        "reviewer_email": reviewer_email,
        "pico": pico,
        "inclusion": project.get("inclusion") or [],
        "exclusion": project.get("exclusion") or [],
        "papers": [
            {
                "paper_id": p.get("paper_id") or p.get("id") or p.get("pmid") or str(i),
                "title": p.get("title") or "(no title)",
                "abstract": p.get("abstract") or "",
                "journal": p.get("journal") or p.get("venue") or "",
                "year": p.get("year") or "",
                "doi": p.get("doi") or "",
                "pmid": p.get("pmid") or "",
            }
            for i, p in enumerate(papers or [])
        ],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    crit = lambda items, cls: "".join(f'<li class="{cls}">{_esc(c)}</li>' for c in items) or f'<li class="{cls} muted">none recorded</li>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(data['project_name'])} — screening pack</title>
<style>
 :root {{ --bg:#fbfbfa; --fg:#1b1b1a; --mut:#6b6b68; --line:#e3e3df; --inc:#1a7f45; --exc:#b4232a; --may:#b8730f; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#16171a; --fg:#ececeb; --mut:#9a9a97; --line:#2c2e33; }} }}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
 header{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:14px 20px;z-index:5}}
 h1{{font-size:17px;margin:0 0 2px}} .muted{{color:var(--mut)}} .wrap{{max-width:860px;margin:0 auto;padding:0 20px 80px}}
 .bar{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:13px}}
 .crit{{display:grid;gap:14px;grid-template-columns:1fr 1fr;margin:18px 0}}
 @media(max-width:640px){{.crit{{grid-template-columns:1fr}}}}
 .crit ul{{margin:4px 0 0;padding-left:18px}} .crit li{{font-size:13px;margin:2px 0}}
 .card{{border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0;background:transparent}}
 .card h3{{margin:0 0 4px;font-size:15px}} .meta{{font-size:12px;color:var(--mut);margin-bottom:8px}}
 .abs{{font-size:13.5px;white-space:pre-wrap}}
 .acts{{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}}
 button{{font:inherit;font-size:13px;padding:6px 12px;border:1px solid var(--line);border-radius:7px;
   background:transparent;color:var(--fg);cursor:pointer}}
 button:hover{{border-color:var(--mut)}}
 .on[data-v=include]{{background:var(--inc);border-color:var(--inc);color:#fff}}
 .on[data-v=exclude]{{background:var(--exc);border-color:var(--exc);color:#fff}}
 .on[data-v=maybe]{{background:var(--may);border-color:var(--may);color:#fff}}
 textarea{{width:100%;margin-top:8px;font:inherit;font-size:13px;padding:7px;border:1px solid var(--line);
   border-radius:7px;background:transparent;color:var(--fg);resize:vertical}}
 .done{{opacity:.55}} .prim{{background:var(--fg);color:var(--bg);border-color:var(--fg);font-weight:600}}
 footer{{position:fixed;bottom:0;left:0;right:0;background:var(--bg);border-top:1px solid var(--line);
   padding:10px 20px;display:flex;gap:12px;align-items:center;justify-content:space-between;font-size:13px}}
 .note{{font-size:12.5px;color:var(--mut);margin-top:6px}}
</style></head><body>
<header><div class="wrap" style="padding-bottom:0">
  <h1>{_esc(data['project_name'])}</h1>
  <div class="bar muted">
    <span>Screening: <strong>{_esc(stage)}</strong></span>
    <span>Reviewer: <strong>{_esc(reviewer_email or reviewer_id)}</strong></span>
    <span id="count"></span>
  </div>
</div></header>
<div class="wrap">
  <div class="crit">
    <div><strong style="font-size:13px">Include if</strong><ul>{crit(data['inclusion'], 'inc')}</ul></div>
    <div><strong style="font-size:13px">Exclude if</strong><ul>{crit(data['exclusion'], 'exc')}</ul></div>
  </div>
  <div class="card" style="font-size:13px">
    <strong>Population</strong> {_esc(pico.get('population')) or '<span class="muted">not set</span>'}<br>
    <strong>Intervention</strong> {_esc(pico.get('intervention')) or '<span class="muted">not set</span>'}<br>
    <strong>Comparator</strong> {_esc(pico.get('comparator')) or '<span class="muted">not set</span>'}<br>
    <strong>Outcome</strong> {_esc(pico.get('outcome')) or '<span class="muted">not set</span>'}
  </div>
  <div class="note">Your answers are saved in this browser as you go. When you have finished, press
    <strong>Download my decisions</strong> and email the file back. {"You cannot see other reviewers' answers: this is a blinded review." if blinded else ""}</div>
  <div id="list"></div>
</div>
<footer>
  <span id="status" class="muted"></span>
  <span style="display:flex;gap:8px">
    <button id="reset">Clear my answers</button>
    <button id="save" class="prim">Download my decisions</button>
  </span>
</footer>
<script id="pack" type="application/json">{payload}</script>
<script>
(function () {{
  var D = JSON.parse(document.getElementById("pack").textContent);
  var KEY = "eepack:" + D.project_id + ":" + (D.reviewer_email || D.reviewer_user_id) + ":" + D.stage;
  var state = {{}};
  try {{ state = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ state = {{}}; }}
  function persist() {{ try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch (e) {{}} }}
  function decided() {{ return D.papers.filter(function (p) {{ return state[p.paper_id] && state[p.paper_id].decision; }}).length; }}
  function paint() {{
    var n = decided(), t = D.papers.length;
    document.getElementById("count").textContent = n + " of " + t + " decided";
    document.getElementById("status").textContent = n === t
      ? "All " + t + " done. Download your decisions and email the file back."
      : (t - n) + " left";
  }}
  var list = document.getElementById("list");
  D.papers.forEach(function (p) {{
    var card = document.createElement("div");
    card.className = "card";
    var meta = [p.journal, p.year, p.pmid ? "PMID " + p.pmid : "", p.doi].filter(Boolean).join(" · ");
    card.innerHTML = '<h3></h3><div class="meta"></div><div class="abs"></div>'
      + '<div class="acts">'
      + '<button data-v="include">Include</button>'
      + '<button data-v="exclude">Exclude</button>'
      + '<button data-v="maybe">Unsure</button>'
      + '</div><textarea rows="2" placeholder="Reason (optional, helps the adjudicator)"></textarea>';
    card.querySelector("h3").textContent = p.title;
    card.querySelector(".meta").textContent = meta;
    card.querySelector(".abs").textContent = p.abstract || "(no abstract supplied)";
    var ta = card.querySelector("textarea");
    var btns = card.querySelectorAll("button[data-v]");
    function sync() {{
      var cur = state[p.paper_id] || {{}};
      btns.forEach(function (b) {{ b.classList.toggle("on", cur.decision === b.dataset.v); }});
      card.classList.toggle("done", !!cur.decision);
      if (document.activeElement !== ta) ta.value = cur.reason || "";
    }}
    btns.forEach(function (b) {{
      b.addEventListener("click", function () {{
        var cur = state[p.paper_id] || {{}};
        cur.decision = cur.decision === b.dataset.v ? "" : b.dataset.v;   // click again to undo
        cur.reason = ta.value; cur.decided_at = new Date().toISOString();
        state[p.paper_id] = cur; persist(); sync(); paint();
      }});
    }});
    ta.addEventListener("input", function () {{
      var cur = state[p.paper_id] || {{}}; cur.reason = ta.value; state[p.paper_id] = cur; persist();
    }});
    sync(); list.appendChild(card);
  }});
  paint();
  document.getElementById("reset").addEventListener("click", function () {{
    if (!confirm("Clear every answer in this pack?")) return;
    state = {{}}; persist(); location.reload();
  }});
  document.getElementById("save").addEventListener("click", function () {{
    var decisions = D.papers.filter(function (p) {{
      return state[p.paper_id] && state[p.paper_id].decision;
    }}).map(function (p) {{
      var c = state[p.paper_id];
      return {{
        paper_id: p.paper_id, stage: D.stage, reviewer_user_id: D.reviewer_user_id,
        decision: c.decision, reason: c.reason || "", decided_at: c.decided_at,
        per_pico_verdict: null, ai_decision: null, is_override: false
      }};
    }});
    if (!decisions.length) {{ alert("Nothing to send yet: decide on at least one article."); return; }}
    var bundle = {{
      kind: D.kind, bundle_version: D.bundle_version,
      generated_at: new Date().toISOString(),
      project: {{ id: D.project_id, name: D.project_name }},
      reviewer: {{ user_id: D.reviewer_user_id, email: D.reviewer_email }},
      decisions: decisions
    }};
    var name = (D.project_name || "review").replace(/[^a-z0-9]+/gi, "-").toLowerCase()
      + "-" + (D.reviewer_email || D.reviewer_user_id).replace(/[^a-z0-9]+/gi, "-") + "-decisions.json";
    var blob = new Blob([JSON.stringify(bundle, null, 2)], {{ type: "application/json" }});
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () {{ URL.revokeObjectURL(a.href); }}, 4000);
  }});
}})();
</script></body></html>"""
