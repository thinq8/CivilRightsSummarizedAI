#!/usr/bin/env python3
"""
case_review_tool.py — Reference-free LLM-as-Judge + human-review dashboard

Freestanding case summary review tool for the Civil Rights Litigation Clearinghouse.
Runs structural QA checks (via existing summary_qa.py) and a reference-free LLM judge
that extracts per-sentence citations back to source documents.  Outputs one self-
contained HTML file: a split-pane view with clickable citation superscripts, QA flag
details, a judge scorecard, and a per-case reviewer feedback form.

The key architectural difference from claude_judge.py:
  Old:  (reference_summary, prediction) → quality scores
  New:  (source_chunks, prediction)     → {per-sentence citations, quality scores}

No human-written reference summary is required.

Usage
-----
  # Full pipeline: QA + judge + citation extraction
  python tools/case_review_tool.py \\
      --eval  eval_claude_code.jsonl \\
      --sources First_Train/test.jsonl \\
      --output review/ \\
      --n 10

  # QA only (free, no API calls)
  python tools/case_review_tool.py \\
      --eval eval_claude_code.jsonl \\
      --output review/ --no-judge

  # Eval file alone (judge without source citations)
  python tools/case_review_tool.py \\
      --eval eval_claude_code.jsonl \\
      --output review/

Environment
-----------
  ANTHROPIC_API_KEY   required for LLM judge
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

# ── Import existing QA checker (graceful fallback) ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
try:
    from summary_qa import SummaryQAChecker
    _HAS_QA = True
except ImportError:
    _HAS_QA = False
    print("Warning: summary_qa.py not importable — structural QA disabled", file=sys.stderr)

# ── Judge prompts (reference-free) ────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are an expert legal editor for the Civil Rights Litigation Clearinghouse (CRLC).
You evaluate AI-generated case summaries by checking them directly against the source
documents — you do NOT have access to a human-written reference summary.

Required editorial elements for a Clearinghouse case summary:
  1. Opening sentence conveying the stakes of the case
  2. Filing date and full formal court name
  3. Whether class or individual action
  4. Counsel — for legal-services organizations, identified by name and org
  5. Causes of action with statutory or constitutional basis (e.g. 42 U.S.C. § 1983)
  6. Remedies sought
  7. Key procedural events and final outcome / disposition

Style rules: past tense throughout · "judgment" not "judgement" ·
acronyms spelled out on first use · encyclopedic prose (no bullet points).
"""

_JUDGE_USER = """\
SOURCE DOCUMENTS (excerpts from the case file):
{source_text}

─────────────────────────────────────────────────
AI-GENERATED SUMMARY:
{prediction}
─────────────────────────────────────────────────

Task A — Citation extraction:
  Split the summary into individual sentences. For each sentence specify which
  numbered source chunk(s) contain evidence supporting the claim.
  If no chunk supports a claim, set grounded=false and chunks=[].

Task B — Quality scoring (1–5):
  factual_grounding  5=every claim traceable to a chunk  1=many unsupported claims
  completeness       5=all 7 required elements present   3=misses 1-2  1=covers <half
  legal_accuracy     5=correct statute/court/parties/cause-of-action throughout
  outcome_accuracy   5=disposition correctly described   1=wrong or absent
  overall            5=publishable with minimal edits    1=complete rewrite needed

Respond with ONLY this JSON (no markdown fences, no other text):
{{
  "citations": [
    {{"s": 0, "sentence": "Exact first sentence.", "chunks": [1, 2], "grounded": true}},
    {{"s": 1, "sentence": "Second sentence.",      "chunks": [],     "grounded": false}}
  ],
  "scores": {{
    "factual_grounding": 4,
    "completeness": 3,
    "legal_accuracy": 4,
    "outcome_accuracy": 5,
    "overall": 4,
    "rationale": "One sentence summarising the main quality issue."
  }}
}}
"""

# ── Tier classification for source chunks ─────────────────────────────────────

_TIER1_RE = re.compile(r"opinion|order|judgment|consent.?decree|settlement", re.I)
_TIER4_RE = re.compile(r"docket|pacer|press.?release|correspondence|newsletter|letter", re.I)


def _doc_tier(doc_type: str) -> int:
    if _TIER1_RE.search(doc_type):
        return 1
    if _TIER4_RE.search(doc_type):
        return 4
    return 2  # complaints, motions, briefs


# ── Source chunk parsing ───────────────────────────────────────────────────────

def parse_source_chunks(prompt_text: str, max_chars: int = 14_000) -> list[dict]:
    """
    Split the raw concatenated prompt from test.jsonl into individual chunks.
    Tier-filters (drops Tier 4), sorts Tier 1 first, truncates to max_chars total.
    Returns list of {index, title, type, date, tier, text}.
    """
    parts = re.split(r"\nChunk \d+:\n", prompt_text)
    if len(parts) <= 1:
        return [{"index": 1, "title": "Source", "type": "Document",
                 "date": "", "tier": 2, "text": prompt_text[:4000]}]

    chunks = []
    for i, part in enumerate(parts[1:], 1):
        m = re.search(r"Title:\s*(.+)", part)
        title = m.group(1).strip() if m else f"Document {i}"

        m = re.search(r"Type:\s*(.+)", part)
        doc_type = m.group(1).strip() if m else "Document"

        m = re.search(r"Date:\s*(.+)", part)
        raw_date = m.group(1).strip() if m else ""
        date = raw_date.split(" ")[0] if raw_date else ""  # strip timestamp

        # Remove the [DOCUMENT] metadata header block, keep body text
        text = re.sub(r"\[DOCUMENT\].*?\n\n", "", part, flags=re.DOTALL).strip()
        tier = _doc_tier(doc_type)
        chunks.append({"index": i, "title": title, "type": doc_type,
                        "date": date, "tier": tier, "text": text})

    # Drop tier-4 (dockets, press releases); sort tier-1 first
    chunks = [c for c in chunks if c["tier"] < 4]
    chunks.sort(key=lambda c: c["tier"])

    # Truncate to max_chars total, keeping higher-tier chunks whole
    kept, total = [], 0
    for c in chunks:
        room = max_chars - total
        if room <= 300:
            break
        c["text"] = c["text"][:room]
        kept.append(c)
        total += len(c["text"])

    return kept


# ── Reference-free LLM judge ──────────────────────────────────────────────────

class ReferenceFreeLLMJudge:
    def __init__(self, model: str = "claude-sonnet-4-6", max_concurrent: int = 5):
        self.model = model
        self.sem = asyncio.Semaphore(max_concurrent)
        self.client = anthropic.AsyncAnthropic()

    def _format_sources(self, chunks: list[dict]) -> str:
        if not chunks:
            return "(No source documents provided — scoring factual grounding as N/A.)"
        parts = []
        for c in chunks:
            header = f"[CHUNK {c['index']} — {c['type']} · {c['date']}]"
            parts.append(f"{header}\n{c['text'][:2800]}")
        return "\n\n".join(parts)

    async def judge_one(self, idx: int, prediction: str, chunks: list[dict]) -> dict:
        async with self.sem:
            source_text = self._format_sources(chunks)
            user_msg = _JUDGE_USER.format(source_text=source_text, prediction=prediction)
            try:
                resp = await self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=_JUDGE_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                text = resp.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                result = json.loads(text)
                score = result.get("scores", {}).get("overall", "?")
                print(f"  [{idx:>3}] overall={score}/5  citations={len(result.get('citations', []))}")
                return result
            except Exception as exc:
                print(f"  [{idx:>3}] ERROR: {exc}", file=sys.stderr)
                return {"citations": [], "scores": None, "error": str(exc)}

    async def judge_batch(self, records: list[dict]) -> list[dict]:
        tasks = [
            self.judge_one(r["index"], r["prediction"], r.get("chunks", []))
            for r in records
        ]
        return await asyncio.gather(*tasks)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_eval_records(path: Path, n: int | None = None) -> list[dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                r = json.loads(line)
                r.setdefault("index", i)
                records.append(r)
    return records


def load_sources(path: Path, n: int | None = None) -> dict[int, dict]:
    """
    Returns {line_index: {chunks, case_id, case_name}} from test.jsonl.
    """
    result = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prompt = r.get("prompt", "")
            # Extract case name from prompt preamble
            m = re.search(r"\nCase:\s*(.+)\n", prompt)
            case_name = m.group(1).strip() if m else f"Case {r.get('case_id', i)}"
            result[i] = {
                "case_id": str(r.get("case_id", i)),
                "case_name": case_name,
                "chunks": parse_source_chunks(prompt),
            }
    return result


# ── HTML template ─────────────────────────────────────────────────────────────
# Placeholders: __CASES_JSON__  __N_CASES__  __GENERATED_AT__

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Case Summary Review · CRLC</title>
<style>
:root{
  --navy:#00274C;--gold:#FFCB05;--cream:#faf8f5;
  --text:#1f2937;--muted:#6b7280;--border:#d1d5db;--card:#fff;
  --green:#16a34a;--gbg:#dcfce7;
  --amber:#b45309;--abg:#fef3c7;
  --red:#dc2626;--rbg:#fee2e2;
  --blue:#2563eb;--bbg:#dbeafe;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:var(--cream);color:var(--text);line-height:1.5;}
header{background:var(--navy);color:#fff;padding:.9rem 1.75rem;
       border-bottom:4px solid var(--gold);
       display:flex;align-items:center;gap:1rem;flex-wrap:wrap;}
header h1{font-size:1.3rem;font-weight:700;}
header .sub{color:var(--gold);font-size:.75rem;margin-top:2px;}
.container{max-width:1420px;margin:0 auto;padding:1.25rem 1.5rem 3rem;}

/* Stats */
.stat-row{display:flex;gap:.75rem;margin-bottom:1.25rem;flex-wrap:wrap;}
.stat-pill{background:var(--card);border:1px solid var(--border);
  border-radius:8px;padding:.65rem 1.1rem;min-width:110px;}
.stat-pill .v{font-size:1.9rem;font-weight:700;}
.stat-pill .l{font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);margin-top:1px;}
.stat-pill.pass .v{color:var(--green);}
.stat-pill.review .v{color:var(--amber);}
.stat-pill.reject .v{color:var(--red);}
.stat-pill.score .v{color:var(--blue);}

/* Card */
.card{background:var(--card);border:1px solid var(--border);
  border-radius:10px;overflow:hidden;margin-bottom:1.25rem;}
.card-hdr{padding:.8rem 1.1rem;border-bottom:1px solid var(--border);
  font-weight:600;font-size:.9rem;display:flex;align-items:center;gap:.6rem;}
.card-hdr input{padding:.3rem .6rem;border:1px solid var(--border);
  border-radius:6px;font-size:.78rem;width:210px;margin-left:auto;}

/* Table */
table{width:100%;border-collapse:collapse;font-size:.84rem;}
th{padding:.55rem .85rem;text-align:left;background:#f9fafb;
  font-size:.69rem;text-transform:uppercase;letter-spacing:.04em;
  color:var(--muted);border-bottom:1px solid var(--border);}
td{padding:.6rem .85rem;border-bottom:1px solid #f3f4f6;vertical-align:middle;}
tr.case-row{cursor:pointer;}
tr.case-row:hover td{background:#f0f4ff;}
tr.detail-row td{padding:0;background:#f8fafc;}

/* Badges */
.badge{display:inline-block;padding:2px 8px;border-radius:999px;
  font-size:.69rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;}
.bp{background:var(--gbg);color:#166534;}
.br{background:var(--abg);color:#92400e;}
.bx{background:var(--rbg);color:#991b1b;}
.bi{background:var(--bbg);color:#1e40af;}

/* Score bar */
.sb-wrap{display:flex;align-items:center;gap:5px;}
.sb-track{flex:1;height:5px;background:#e5e7eb;border-radius:3px;}
.sb-fill{height:5px;border-radius:3px;}
.sb-lbl{font-size:.72rem;min-width:26px;text-align:right;font-weight:600;}

/* Detail panel */
.detail-inner{padding:1.25rem 1.5rem;border-top:1px solid var(--border);}
.det-grid{display:grid;grid-template-columns:1fr 360px;gap:1.1rem;align-items:start;}
@media(max-width:860px){.det-grid{grid-template-columns:1fr;}}
.panel-lbl{font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;
  color:var(--muted);margin-bottom:.45rem;font-weight:600;}

/* Summary text */
.sum-text{background:var(--card);border:1px solid var(--border);
  border-radius:8px;padding:.9rem;font-size:.87rem;line-height:1.8;
  max-height:500px;overflow-y:auto;}

/* Sentence citation markup */
.s-grounded{border-bottom:1.5px solid #86efac;}
.s-ungrounded{border-bottom:1.5px solid #fca5a5;background:#fff5f5;}
.s-hover{background:#fef9c3;}
sup.cref{font-size:.63em;color:var(--blue);cursor:pointer;
  font-weight:700;margin-left:1px;}
sup.cref:hover{text-decoration:underline;}

/* Sources panel */
.src-panel{background:var(--card);border:1px solid var(--border);
  border-radius:8px;max-height:500px;overflow-y:auto;}
.src-chunk{padding:.7rem .9rem;border-bottom:1px solid #f3f4f6;
  font-size:.77rem;line-height:1.6;transition:background .12s;}
.src-chunk:last-child{border-bottom:none;}
.src-chunk.hl{background:#fefce8;border-left:3px solid var(--gold);}
.src-hdr{font-weight:600;color:var(--navy);font-size:.7rem;margin-bottom:.25rem;}
.src-meta{color:var(--muted);font-size:.65rem;margin-bottom:.3rem;}

/* QA flags */
.flags-wrap{margin-top:1rem;}
.flag-list{list-style:none;display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.4rem;}
.flag-li{padding:3px 9px;border-radius:5px;font-size:.73rem;
  font-weight:600;border-left:3px solid;}
.fc{background:#fef2f2;border-color:var(--red);color:#991b1b;}
.fw{background:#fffbeb;border-color:#f59e0b;color:#92400e;}
.fi{background:#f9fafb;border-color:#9ca3af;color:#374151;}

/* Judge scorecard */
.scorecard{margin-top:1rem;background:var(--card);border:1px solid var(--border);
  border-radius:8px;padding:.9rem;}
.sc-title{font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;
  color:var(--muted);font-weight:600;margin-bottom:.65rem;}
.sc-row{display:flex;align-items:center;gap:.5rem;
  margin-bottom:.45rem;font-size:.8rem;}
.sc-dim{width:155px;flex-shrink:0;}
.rationale{font-size:.77rem;color:var(--muted);margin-top:.65rem;
  padding-top:.65rem;border-top:1px solid var(--border);font-style:italic;}

/* Feedback form */
.fb-form{margin-top:1rem;background:#f0f4ff;
  border:1px solid #bfdbfe;border-radius:8px;padding:.9rem;}
.fb-form h4{font-size:.78rem;font-weight:700;color:var(--navy);margin-bottom:.65rem;}
.fb-row{display:flex;gap:.6rem;align-items:center;
  margin-bottom:.5rem;flex-wrap:wrap;}
.fb-row label{font-size:.77rem;}
.fb-row select{padding:2px 5px;border:1px solid var(--border);
  border-radius:4px;font-size:.77rem;}
.fb-row textarea{width:100%;padding:.35rem .55rem;
  border:1px solid var(--border);border-radius:4px;
  font-size:.77rem;font-family:inherit;resize:vertical;min-height:55px;}
.btn{display:inline-block;padding:.35rem .85rem;border-radius:6px;
  font-size:.77rem;font-weight:600;cursor:pointer;border:none;}
.btn-primary{background:var(--navy);color:#fff;}
.btn-primary:hover{background:#003a6b;}

.arr{margin-left:auto;font-size:.72rem;color:var(--muted);transition:transform .15s;}
.open .arr{transform:rotate(180deg);}
</style>
</head>
<body>
<header>
  <div>
    <h1>Case Summary Review</h1>
    <div class="sub">Civil Rights Litigation Clearinghouse &middot; Reference-Free Judge + QA Triage</div>
  </div>
  <div style="margin-left:auto;font-size:.73rem;opacity:.8;text-align:right;">
    __N_CASES__ cases &middot; __GENERATED_AT__
  </div>
</header>

<div class="container">
  <div class="stat-row" id="stat-row"></div>

  <div class="card">
    <div class="card-hdr">
      All Cases
      <input type="text" id="fi" placeholder="Filter by name, status, flag…"
             oninput="filterTable(this.value)">
    </div>
    <table>
      <thead>
        <tr>
          <th>Status</th><th>Case</th><th>QA Flags</th>
          <th>Judge Score</th><th>Words</th><th></th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>

  <div style="font-size:.71rem;color:var(--muted);text-align:center;padding-bottom:2rem;">
    Green underline&nbsp;=&nbsp;grounded sentence &nbsp;&middot;&nbsp;
    Red underline&nbsp;=&nbsp;unsupported claim &nbsp;&middot;&nbsp;
    Click citation superscripts to highlight source chunk
  </div>
</div>

<script>
const CASES = __CASES_JSON__;

const DIMS=[
  ["factual_grounding","Factual Grounding"],
  ["completeness","Completeness"],
  ["legal_accuracy","Legal Accuracy"],
  ["outcome_accuracy","Outcome Accuracy"],
  ["overall","Overall Quality"],
];

function esc(s){
  return String(s||"")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

function badge(st){
  const m={PASS:"bp",REVIEW:"br",REJECT:"bx"}[st]||"bi";
  return `<span class="badge ${m}">${esc(st)}</span>`;
}

function scoreBar(v,max=5){
  if(!v||v<=0)return`<span style="color:var(--muted);font-size:.77rem">—</span>`;
  const pct=Math.round((v/max)*100);
  const col=v>=4?"#16a34a":v>=3?"#f59e0b":"#dc2626";
  return`<div class="sb-wrap">
    <div class="sb-track"><div class="sb-fill" style="width:${pct}%;background:${col}"></div></div>
    <span class="sb-lbl" style="color:${col}">${v}/5</span>
  </div>`;
}

function buildSummary(c){
  if(!c.citations||!c.citations.length){
    return`<div class="sum-text">${esc(c.prediction)}</div>`;
  }
  const parts=c.citations.map((cit,i)=>{
    const cls=cit.grounded?"s-grounded":"s-ungrounded";
    const cks=cit.chunks||[];
    const sup=cks.length
      ?`<sup class="cref" onclick="hlChunks(${c.index},[${cks}])"
           title="Chunk ${cks.join(",")}">[${cks.join(",")}]</sup>`:"";
    return`<span class="${cls}"
      onmouseenter="hov(${c.index},[${cks}])"
      onmouseleave="unhov(${c.index},[${cks}])"
    >${esc(cit.sentence)}${sup} </span>`;
  });
  return`<div class="sum-text">${parts.join("")}</div>`;
}

function buildSources(c){
  if(!c.chunks||!c.chunks.length){
    return`<div class="src-panel" style="padding:1rem;font-size:.8rem;color:var(--muted)">
      No source documents — run with --sources to enable citations.
    </div>`;
  }
  const rows=c.chunks.map(ch=>`
    <div class="src-chunk" id="ck-${c.index}-${ch.index}">
      <div class="src-hdr">Chunk ${ch.index} &mdash; ${esc(ch.title)}</div>
      <div class="src-meta">${esc(ch.type)} &middot; ${esc(ch.date)}</div>
      <div>${esc((ch.text||"").slice(0,900))}${(ch.text||"").length>900?"…":""}</div>
    </div>`).join("");
  return`<div class="src-panel">${rows}</div>`;
}

function buildFlags(c){
  if(!c.qa_flags||!c.qa_flags.length)return"";
  const items=c.qa_flags.map(f=>{
    const cls={critical:"fc",warning:"fw",info:"fi"}[f.severity]||"fi";
    const tip=f.evidence?` title="${esc(f.evidence)}"`:""
    return`<li class="flag-li ${cls}"${tip}>${esc(f.code)}</li>`;
  }).join("");
  return`<div class="flags-wrap">
    <div class="panel-lbl">QA Flags</div>
    <ul class="flag-list">${items}</ul>
  </div>`;
}

function buildScorecard(c){
  if(!c.judge_scores||!c.judge_scores.overall)return"";
  const s=c.judge_scores;
  const rows=DIMS.map(([k,l])=>
    `<div class="sc-row"><span class="sc-dim">${l}</span>${scoreBar(s[k])}</div>`
  ).join("");
  const rat=s.rationale
    ?`<div class="rationale">${esc(s.rationale)}</div>`:"";
  return`<div class="scorecard">
    <div class="sc-title">Judge Scorecard &mdash; Reference-Free</div>
    ${rows}${rat}
  </div>`;
}

function buildFeedback(c){
  const id=c.index;
  return`<div class="fb-form">
    <h4>Human Reviewer Feedback</h4>
    <div class="fb-row">
      <label>Rating:</label>
      <select id="fbr-${id}">
        <option value="">—</option>
        <option value="1">1 — Poor</option>
        <option value="2">2 — Fair</option>
        <option value="3">3 — Acceptable</option>
        <option value="4">4 — Good</option>
        <option value="5">5 — Excellent</option>
      </select>
    </div>
    <div class="fb-row">
      <label><input type="checkbox" id="fba-${id}"> Factually accurate</label>
      <label><input type="checkbox" id="fbc-${id}"> All 7 elements present</label>
      <label><input type="checkbox" id="fbp-${id}"> Would publish (minor edits)</label>
    </div>
    <div class="fb-row" style="display:block">
      <label style="display:block;margin-bottom:3px;font-size:.77rem">Notes / corrections:</label>
      <textarea id="fbn-${id}" placeholder="Describe errors, missing elements, style issues…"></textarea>
    </div>
    <button class="btn btn-primary" onclick="dlFeedback(${id})">&#8659; Download Review JSON</button>
  </div>`;
}

function dlFeedback(idx){
  const c=CASES.find(x=>x.index===idx)||{};
  const rating=document.getElementById(`fbr-${idx}`)?.value||"";
  const fb={
    case_id:c.case_id||idx,
    case_name:c.case_name||"",
    reviewer_rating:parseInt(rating)||null,
    is_accurate:!!document.getElementById(`fba-${idx}`)?.checked,
    is_complete:!!document.getElementById(`fbc-${idx}`)?.checked,
    would_publish:!!document.getElementById(`fbp-${idx}`)?.checked,
    notes:document.getElementById(`fbn-${idx}`)?.value||"",
    qa_status:c.qa_status||"",
    judge_overall:c.judge_scores?.overall||null,
    timestamp:new Date().toISOString(),
    tool:"case_review_tool v1.0",
  };
  const blob=new Blob([JSON.stringify(fb,null,2)],{type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob);
  a.download=`review_${String(c.case_name||idx).replace(/\s+/g,"_").slice(0,40)}.json`;
  a.click();
}

function hlChunks(ci,ids){
  document.querySelectorAll(`[id^="ck-${ci}-"]`).forEach(el=>el.classList.remove("hl"));
  ids.forEach(id=>{
    const el=document.getElementById(`ck-${ci}-${id}`);
    if(el){el.classList.add("hl");el.scrollIntoView({behavior:"smooth",block:"nearest"});}
  });
}
function hov(ci,ids){ids.forEach(id=>{const el=document.getElementById(`ck-${ci}-${id}`);if(el)el.classList.add("hl");});}
function unhov(ci,ids){ids.forEach(id=>{const el=document.getElementById(`ck-${ci}-${id}`);if(el)el.classList.remove("hl");});}

function renderStats(){
  const n=CASES.length;
  const nP=CASES.filter(c=>c.qa_status==="PASS").length;
  const nR=CASES.filter(c=>c.qa_status==="REVIEW").length;
  const nX=CASES.filter(c=>c.qa_status==="REJECT").length;
  const jc=CASES.filter(c=>c.judge_scores?.overall>0);
  const avg=jc.length?(jc.reduce((s,c)=>s+c.judge_scores.overall,0)/jc.length).toFixed(2):"—";
  const pct=x=>Math.round(100*x/n);
  document.getElementById("stat-row").innerHTML=`
    <div class="stat-pill"><div class="v">${n}</div><div class="l">Total</div></div>
    <div class="stat-pill pass"><div class="v">${nP}</div><div class="l">Pass (${pct(nP)}%)</div></div>
    <div class="stat-pill review"><div class="v">${nR}</div><div class="l">Review (${pct(nR)}%)</div></div>
    <div class="stat-pill reject"><div class="v">${nX}</div><div class="l">Reject (${pct(nX)}%)</div></div>
    <div class="stat-pill score"><div class="v">${avg}</div><div class="l">Avg Judge /5</div></div>
  `;
}

function renderTable(){
  let html="";
  for(const c of CASES){
    const crit=(c.qa_flags||[]).filter(f=>f.severity==="critical").length;
    const warn=(c.qa_flags||[]).filter(f=>f.severity==="warning").length;
    const flagHtml=[
      crit?`<span class="badge bx">${crit} crit</span>`:"",
      warn?`<span class="badge br">${warn} warn</span>`:"",
      (!crit&&!warn)?`<span class="badge bp">none</span>`:"",
    ].join(" ");
    const jsc=c.judge_scores?.overall||0;
    const ung=(c.citations||[]).filter(x=>!x.grounded).length;
    const srch=[c.case_name,c.case_id,c.qa_status,...(c.qa_flags||[]).map(f=>f.code)]
      .join(" ").toLowerCase();
    const trunc=c.truncated?`<br><span class="badge bi" style="font-size:.63rem">truncated src</span>`:"";

    html+=`
      <tr class="case-row" id="row-${c.index}" data-idx="${c.index}"
          data-search="${esc(srch)}" onclick="toggle(${c.index})">
        <td>${badge(c.qa_status||"—")}</td>
        <td><strong>${esc(c.case_name||c.case_id||"Case "+c.index)}</strong>${trunc}</td>
        <td>${flagHtml}</td>
        <td>${jsc?scoreBar(jsc):`<span style="color:var(--muted);font-size:.77rem">not scored</span>`}</td>
        <td style="color:var(--muted);font-size:.78rem">${c.prediction_len||"—"}</td>
        <td><span class="arr">&#9660;</span></td>
      </tr>
      <tr class="detail-row" id="det-${c.index}" style="display:none">
        <td colspan="6">
          <div class="detail-inner">
            <div class="det-grid">
              <div>
                <div class="panel-lbl">
                  Generated Summary
                  ${ung?`<span class="badge bx" style="margin-left:6px">${ung} ungrounded</span>`:""}
                </div>
                ${buildSummary(c)}
                ${buildFlags(c)}
                ${buildScorecard(c)}
                ${buildFeedback(c)}
              </div>
              <div>
                <div class="panel-lbl">Source Documents (${(c.chunks||[]).length} chunks)</div>
                ${buildSources(c)}
              </div>
            </div>
          </div>
        </td>
      </tr>`;
  }
  document.getElementById("tbody").innerHTML=html;
}

function toggle(idx){
  const row=document.getElementById(`row-${idx}`);
  const det=document.getElementById(`det-${idx}`);
  const open=det.style.display!=="none";
  det.style.display=open?"none":"";
  row.classList.toggle("open",!open);
}

function filterTable(q){
  q=q.toLowerCase().trim();
  document.querySelectorAll(".case-row").forEach(row=>{
    const show=!q||(row.getAttribute("data-search")||"").includes(q);
    row.style.display=show?"":"none";
    const idx=row.getAttribute("data-idx");
    const det=document.getElementById(`det-${idx}`);
    if(det&&!show)det.style.display="none";
  });
}

renderStats();
renderTable();
</script>
</body>
</html>
"""


# ── HTML rendering ─────────────────────────────────────────────────────────────

def render_html(cases_data: list[dict], output_path: Path) -> None:
    cases_json = json.dumps(cases_data, ensure_ascii=False, default=str)
    # Prevent </script> inside the embedded JSON from breaking the HTML page
    cases_json = cases_json.replace("</script>", "<\\/script>")

    n_cases = str(len(cases_data))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = _HTML \
        .replace("__CASES_JSON__", cases_json) \
        .replace("__N_CASES__", n_cases) \
        .replace("__GENERATED_AT__", generated_at)

    output_path.write_text(html, encoding="utf-8")
    size_kb = output_path.stat().st_size // 1024
    print(f"\nWrote {output_path}  ({size_kb} KB)")
    print(f"Open:  file://{output_path.resolve()}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    eval_path = Path(args.eval)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading eval records from {eval_path} …")
    records = load_eval_records(eval_path, n=args.n)
    print(f"  {len(records)} records")

    # Load source chunks if provided
    sources_map: dict[int, dict] = {}
    if args.sources:
        print(f"Loading source documents from {args.sources} …")
        sources_map = load_sources(Path(args.sources), n=args.n)
        print(f"  Parsed chunks for {len(sources_map)} cases")

    # QA checker
    qa_checker = SummaryQAChecker(enable_rouge=False) if _HAS_QA else None

    # Build enriched case records
    enriched: list[dict] = []
    for r in records:
        idx = r.get("index", 0)
        prediction = r.get("prediction", "")
        src = sources_map.get(idx, {})

        # QA structural checks
        qa_flags, qa_status = [], "—"
        if qa_checker:
            rep = qa_checker.check(prediction=prediction, identifier=str(idx))
            qa_status = rep.status
            qa_flags = [
                {"severity": f.severity, "category": f.category,
                 "code": f.code, "message": f.message,
                 "evidence": (f.evidence or "")[:100]}
                for f in rep.flags
            ]

        # Case name: from sources map if available, else first sentence of prediction
        case_name = src.get("case_name") or r.get("case_name", "")
        if not case_name:
            first = prediction.split(".")[0].strip()
            case_name = first[:80] if len(first) > 5 else f"Case {idx}"

        enriched.append({
            "index": idx,
            "case_id": src.get("case_id") or str(r.get("case_id", idx)),
            "case_name": case_name,
            "prediction": prediction,
            "prediction_len": r.get("prediction_len", len(prediction.split())),
            "truncated": r.get("truncated", False),
            "chunks": src.get("chunks", []),
            "qa_status": qa_status,
            "qa_flags": qa_flags,
            "citations": [],
            "judge_scores": None,
        })

    # LLM judge
    if not args.no_judge:
        print(f"\nRunning reference-free LLM judge ({args.model}) on {len(enriched)} cases …")
        judge = ReferenceFreeLLMJudge(model=args.model, max_concurrent=args.concurrency)
        judge_inputs = [
            {"index": e["index"], "prediction": e["prediction"], "chunks": e["chunks"]}
            for e in enriched
        ]
        results = await judge.judge_batch(judge_inputs)
        for e, res in zip(enriched, results):
            e["citations"] = res.get("citations", [])
            e["judge_scores"] = res.get("scores", {})
    else:
        print("Skipping LLM judge (--no-judge).")

    # Output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"case_review_{ts}.html"
    render_html(enriched, out_path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Reference-free LLM-as-judge + QA review dashboard for CRLC case summaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--eval", required=True,
                   help="Eval JSONL path (e.g. eval_claude_code.jsonl)")
    p.add_argument("--sources",
                   help="Source JSONL path (First_Train/test.jsonl) for citation extraction")
    p.add_argument("--output", default="review",
                   help="Output directory (default: review/)")
    p.add_argument("--n", type=int, default=None,
                   help="Process first N cases only")
    p.add_argument("--no-judge", action="store_true",
                   help="Skip LLM judge (structural QA only, free)")
    p.add_argument("--model", default="claude-sonnet-4-6",
                   help="Judge model (default: claude-sonnet-4-6)")
    p.add_argument("--concurrency", type=int, default=5,
                   help="Max concurrent API calls (default: 5)")
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
