"""Prepare improved training data with document-aware fragmentation strategies.

Transforms raw training JSONL (naive chunk concatenation) into optimized prompts
that fit within model context windows and prioritize important documents.

Strategies:
    priority_filter  — Tier-based document selection + token budget enforcement
    structured       — Hierarchical prompt with timeline, importance labels, end instructions
    extract_first    — Two-stage: extract structured facts per doc, then train on synthesis

Usage:
    # Strategy 1: Priority filter (fast, no API needed)
    python scripts/prepare_training_data.py \\
        --input data/training/train.jsonl \\
        --output data/training_v2/train.jsonl \\
        --strategy priority_filter \\
        --max-tokens 24000

    # Strategy 2: Structured with timeline + importance labels
    python scripts/prepare_training_data.py \\
        --input data/training/train.jsonl \\
        --output data/training_v2/train_structured.jsonl \\
        --strategy structured \\
        --max-tokens 24000

    # Strategy 3: Two-stage extraction with Claude API (best quality)
    python scripts/prepare_training_data.py \\
        --input data/training/train.jsonl \\
        --output data/training_v2/train_extracted.jsonl \\
        --strategy extract_first \\
        --max-tokens 24000 \\
        --extraction-backend claude \\
        --cache-dir data/extraction_cache

    # Strategy 3 fallback: Heuristic extraction (no API, lower quality)
    python scripts/prepare_training_data.py \\
        --input data/training/train.jsonl \\
        --output data/training_v2/train_extracted_heuristic.jsonl \\
        --strategy extract_first \\
        --max-tokens 24000

    # Dry run (show stats without writing):
    python scripts/prepare_training_data.py \\
        --input data/training/test.jsonl \\
        --strategy priority_filter \\
        --max-tokens 24000 \\
        --dry-run --sample 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Add scripts/ to path for doc_classifier import
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from doc_classifier import TIER_LABELS, classify_document, estimate_tokens, parse_chunks


# ── Extraction prompts (from court-document-summarizer.html v1.4) ─────────────
# These match the professor's production prompts used by human annotators.

EXTRACTION_PROMPTS = {
    "complaint": (
        "You are a legal expert extracting key details from a lawsuit's complaint.\n\n"
        "Important: A complaint contains allegations, not proven facts. Use language "
        "that reflects this (e.g., 'Plaintiffs allege that...').\n\n"
        "Base your extraction solely on the document provided.\n\n"
        "Extract the following:\n"
        "- Parties: plaintiffs and defendants with descriptions for individuals, "
        "names for organizations. Note putative class action status.\n"
        "- Underlying Facts: key facts of the dispute as alleged\n"
        "- Date Filed\n"
        "- Court: formal name (e.g., U.S. District Court for the District of Massachusetts)\n"
        "- Jurisdiction Basis: federal question, diversity, supplemental\n"
        "- Causes of Action and Legal Claims: with statutory basis\n"
        "- Remedies Sought: injunctive relief, damages, attorneys fees, etc.\n\n"
        "Format as a structured list with clear labels."
    ),
    "district_opinion": (
        "You are a legal expert extracting key details from a trial court opinion.\n\n"
        "Base your extraction solely on the document provided.\n\n"
        "Extract the following:\n"
        "- Case Name and Docket Number\n"
        "- Court and Judge\n"
        "- Date\n"
        "- Procedural Posture: what motion or issue is being decided\n"
        "- Facts Found: key facts as found or accepted by the court\n"
        "- Claims at Issue: legal claims being addressed\n"
        "- Holdings: rulings on each issue (granted, denied, etc.)\n"
        "- Key Reasoning: legal reasoning including key cases relied upon; "
        "how court addressed opposing arguments\n"
        "- Remedies Granted: if relief granted, describe what was ordered\n"
        "- Outcome: practical result for the case going forward\n\n"
        "Format as a structured list with clear labels."
    ),
    "appellate_opinion": (
        "You are a legal expert extracting key details from an appellate court opinion.\n\n"
        "Base your extraction solely on the document provided.\n\n"
        "Extract the following:\n"
        "- Case Name and Docket Number\n"
        "- Court and Panel/Judges: note author and any concurrences/dissents\n"
        "- Date\n"
        "- Published or Unpublished\n"
        "- Lower Court and Judge whose decision is reviewed\n"
        "- Procedural History: how the case arrived at the appellate court\n"
        "- Facts Found: key facts as found or accepted by the courts\n"
        "- Issues on Appeal: specific legal questions presented\n"
        "- Standard of Review: for each issue (de novo, abuse of discretion, etc.)\n"
        "- Holdings: rulings on each issue\n"
        "- Key Reasoning: including key cases and how court addressed opposing arguments\n"
        "- Disposition: affirmed, reversed, remanded, etc.\n"
        "- Concurrence/Dissent Summary: key points of disagreement, or 'None'\n\n"
        "Format as a structured list with clear labels."
    ),
    "motion_to_dismiss": (
        "You are a legal expert extracting key details from a motion to dismiss.\n\n"
        "Base your extraction solely on the document provided.\n\n"
        "Extract the following:\n"
        "- Moving Party: which defendant(s)\n"
        "- Date Filed\n"
        "- Procedural Posture: first motion, or follows amended complaint?\n"
        "- Claims Targeted: which claims or counts — all, or specific ones?\n"
        "- Grounds for Dismissal: Rule 12(b)(6), 12(b)(1), qualified immunity, "
        "statute of limitations, failure to exhaust, etc.\n"
        "- Key Arguments: main arguments for each ground\n\n"
        "Format as a structured list with clear labels."
    ),
    "motion_summary_judgment": (
        "You are a legal expert extracting key details from a motion for summary judgment.\n\n"
        "Base your extraction solely on the document provided.\n\n"
        "Extract the following:\n"
        "- Moving Party: plaintiff or defendant\n"
        "- Date Filed\n"
        "- Procedural Context: standalone or cross-motions? Full or partial?\n"
        "- Claims at Issue: which claims — all or specific ones?\n"
        "- Statement of Undisputed Facts: key material facts claimed undisputed\n"
        "- Legal Arguments: why summary judgment should be granted on each claim\n"
        "- Key Evidence Cited: depositions, documents, expert reports, etc.\n\n"
        "Format as a structured list with clear labels."
    ),
    "class_certification": (
        "You are a legal expert extracting key details from a class certification motion.\n\n"
        "Base your extraction solely on the document provided.\n\n"
        "Extract the following:\n"
        "- Moving Party\n"
        "- Date Filed\n"
        "- Proposed Class Definition: including subclasses\n"
        "- Estimated Class Size\n"
        "- Named Representatives\n"
        "- Type of Class Sought: Rule 23(b)(1), (b)(2), (b)(3)\n"
        "- Rule 23(a) Arguments: numerosity, commonality, typicality, adequacy\n"
        "- Rule 23(b) Arguments\n"
        "- Common Questions Identified\n"
        "- Key Evidence Cited\n\n"
        "Format as a structured list with clear labels."
    ),
    "settlement": (
        "You are extracting key details from a lawsuit settlement agreement.\n\n"
        "Base your extraction solely on the document provided. "
        "If any information is not present, state 'Not Specified'.\n\n"
        "Extract the following:\n"
        "- Type of Agreement: consent decree, stipulated dismissal, etc.\n"
        "- Date\n"
        "- Actions to be Taken by Defendants: who agreed to do what\n"
        "- Damages (Money): who is paying for what, including attorneys' fees\n"
        "- Implementation and Enforcement: monitors, oversight mechanisms\n"
        "- Duration: how long, what triggers close\n"
        "- Conditional Agreements: any conditions\n"
        "- Policy Adoptions: agreements to adopt policies with details\n"
        "- Cy Pres Provisions: if class action, any cy pres distribution\n\n"
        "Do NOT extract boilerplate, notice provisions, or claim-waiver language.\n"
        "Format as a structured list with clear labels."
    ),
    "motion": (
        "You are a legal expert extracting key details from a legal motion or brief.\n\n"
        "Base your extraction solely on the document provided.\n\n"
        "Extract the following:\n"
        "- Moving Party\n"
        "- Date Filed\n"
        "- Type of Motion\n"
        "- Claims or Issues Targeted\n"
        "- Key Legal Arguments with standards cited\n"
        "- Relief Requested\n\n"
        "Format as a structured list with clear labels."
    ),
    "default": (
        "You are a legal expert extracting key factual and legal information.\n\n"
        "Base your extraction solely on the document provided.\n\n"
        "Extract the following:\n"
        "- Document Type and Date\n"
        "- Parties Involved\n"
        "- Key Facts, Claims, or Arguments\n"
        "- Any Decisions, Orders, or Outcomes\n\n"
        "Format as a structured list with clear labels."
    ),
}


def get_extraction_prompt(title: str, doc_type: str) -> str:
    """Select the appropriate extraction prompt based on document type.

    Uses fine-grained categories matching the professor's court-document-summarizer.html
    to get document-type-specific extraction (opinions, motions, settlements, etc.).
    """
    combined = f"{title} {doc_type}".lower()

    # Settlement — check before opinion (consent decree is a settlement, not an opinion)
    if any(w in combined for w in ["settlement", "consent decree", "agreed judgment",
                                    "stipulated dismissal"]):
        return EXTRACTION_PROMPTS["settlement"]
    # Appellate opinions
    if any(w in combined for w in ["appellate", "usca", "per curiam", "circuit court",
                                    "court of appeals"]):
        return EXTRACTION_PROMPTS["appellate_opinion"]
    # Class certification
    if any(w in combined for w in ["class cert", "class action certification",
                                    "motion to certify"]):
        return EXTRACTION_PROMPTS["class_certification"]
    # Motion to dismiss
    if any(w in combined for w in ["motion to dismiss", "12(b)"]):
        return EXTRACTION_PROMPTS["motion_to_dismiss"]
    # Summary judgment
    if any(w in combined for w in ["summary judgment", "msj"]):
        return EXTRACTION_PROMPTS["motion_summary_judgment"]
    # District/trial court opinions, orders, judgments
    if any(w in combined for w in ["opinion", "order", "judgment", "memorandum opinion",
                                    "memorandum decision", "report and recommendation",
                                    "findings"]):
        return EXTRACTION_PROMPTS["district_opinion"]
    # Complaints
    if any(w in combined for w in ["complaint", "amended complaint"]):
        return EXTRACTION_PROMPTS["complaint"]
    # Generic motions and briefs
    if any(w in combined for w in ["motion", "brief", "memorandum"]):
        return EXTRACTION_PROMPTS["motion"]
    return EXTRACTION_PROMPTS["default"]


# ── Proportional budget allocation ────────────────────────────────────────────

def _allocate_proportional(docs: list[dict], budget: int) -> list[dict]:
    """Distribute token budget proportionally across documents when they overflow.

    Instead of giving the entire budget to the first document and truncating
    everything else, each document gets a share proportional to its original size.
    This ensures the model sees the opening + key holdings from EVERY document
    rather than reading one document in full and missing the rest entirely.

    Returns new list of (possibly truncated) document dicts.
    """
    total_tokens = sum(estimate_tokens(d["body"]) for d in docs)
    if total_tokens <= budget:
        return docs  # everything fits, no truncation needed

    result = []
    for doc in docs:
        doc_tokens = estimate_tokens(doc["body"])
        # Each doc gets a proportional share of the budget, with a minimum floor
        share = max(int(budget * (doc_tokens / total_tokens)), 200)
        share_chars = share * 4

        if len(doc["body"]) > share_chars:
            doc_copy = dict(doc)
            doc_copy["body"] = doc["body"][:share_chars] + "\n[...truncated]"
            result.append(doc_copy)
        else:
            result.append(doc)

    return result


# ── Strategy: Priority Filter ─────────────────────────────────────────────────

def strategy_priority_filter(prompt_text: str, max_tokens: int = 24000) -> tuple[str, dict]:
    """Select documents by priority tier, fitting within token budget.

    When Tier 1 documents alone exceed the budget, allocates space proportionally
    across all Tier 1 docs so the model sees key content from every important
    document rather than one giant truncated document.

    Returns (new_prompt, metadata_dict).
    """
    docs, case_name, preamble = parse_chunks(prompt_text)

    if not docs:
        return prompt_text, {"strategy": "priority_filter", "error": "no_docs_parsed"}

    # Group by tier
    by_tier = {1: [], 2: [], 3: [], 4: []}
    for doc in docs:
        by_tier[doc["tier"]].append(doc)

    # Budget allocation: reserve tokens for preamble + instructions
    preamble_tokens = estimate_tokens(preamble) + 200  # buffer for formatting
    available = max_tokens - preamble_tokens
    if available < 500:
        available = max_tokens  # fallback: use full budget

    selected = []
    tokens_used = 0
    excluded_titles = []

    # Check if Tier 1 alone overflows the budget
    tier1_total = sum(estimate_tokens(d["body"]) for d in by_tier[1])

    if tier1_total > available and len(by_tier[1]) > 1:
        # Proportional allocation: give every Tier 1 doc a fair share
        selected = _allocate_proportional(by_tier[1], available)
        tokens_used = sum(estimate_tokens(d["body"]) for d in selected)
        # All Tier 2/3 docs are excluded since Tier 1 fills the budget
        excluded_titles.extend(d["title"] for d in by_tier[2])
        excluded_titles.extend(d["title"] for d in by_tier[3])
    else:
        # Normal greedy selection: Tier 1 fits, try to add Tier 2 and 3
        for tier in [1, 2, 3]:  # Skip tier 4 entirely
            for doc in by_tier[tier]:
                doc_tokens = estimate_tokens(doc["body"])
                if tokens_used + doc_tokens <= available:
                    selected.append(doc)
                    tokens_used += doc_tokens
                elif tier == 1:
                    # Single large Tier 1 doc — truncate to fit
                    remaining = available - tokens_used
                    truncated_chars = remaining * 4
                    doc_copy = dict(doc)
                    doc_copy["body"] = doc["body"][:truncated_chars] + "\n[...truncated]"
                    selected.append(doc_copy)
                    tokens_used = available
                else:
                    excluded_titles.append(doc["title"])

    # Add all tier 4 to excluded
    excluded_titles.extend(d["title"] for d in by_tier[4])

    # Rebuild prompt with selected docs
    parts = [
        "Summarize the following legal case materials into a concise case summary. "
        "Focus on the dispute, major procedural posture, key rulings or events, "
        "and the current or final outcome when available.",
        f"\nCase: {case_name}\n",
    ]

    for i, doc in enumerate(selected, 1):
        header = f"\n[DOCUMENT]\nTitle: {doc['title']}"
        if doc["doc_type"]:
            header += f"\nType: {doc['doc_type']}"
        if doc["date"]:
            header += f"\nDate: {doc['date']}"
        parts.append(header + "\n\n" + doc["body"])

    new_prompt = "\n".join(parts)

    metadata = {
        "strategy": "priority_filter",
        "total_docs": len(docs),
        "selected_docs": len(selected),
        "excluded_docs": len(docs) - len(selected),
        "excluded_doc_types": excluded_titles,
        "prompt_tokens_est": estimate_tokens(new_prompt),
        "tier_counts": {f"tier_{t}": len(by_tier[t]) for t in [1, 2, 3, 4]},
    }

    return new_prompt, metadata


# ── Strategy: Structured ──────────────────────────────────────────────────────

def strategy_structured(prompt_text: str, max_tokens: int = 24000) -> tuple[str, dict]:
    """Reformat into hierarchical prompt with timeline, importance labels, and
    instructions at the end (closer to generation for better attention).

    Returns (new_prompt, metadata_dict).
    """
    docs, case_name, preamble = parse_chunks(prompt_text)

    if not docs:
        return prompt_text, {"strategy": "structured", "error": "no_docs_parsed"}

    # Sort by date (if available), then by tier
    def sort_key(d):
        # Parse date for sorting (YYYY-MM-DD prefix)
        date_str = d["date"][:10] if d["date"] else "9999-99-99"
        return (d["tier"], date_str)

    docs_sorted = sorted(docs, key=sort_key)

    # Build timeline from dated documents
    timeline_entries = []
    for doc in docs_sorted:
        if doc["date"] and doc["tier"] <= 3:
            date_short = doc["date"][:10]  # YYYY-MM-DD
            timeline_entries.append(f"- {date_short}: {doc['title'][:80]}")

    # Select documents within token budget
    budget = max_tokens - 500  # reserve for framing
    selected = []
    tokens_used = 0
    excluded = []

    # Check if Tier 1 alone overflows
    tier1_docs = [d for d in docs_sorted if d["tier"] == 1]
    tier1_total = sum(estimate_tokens(d["body"]) for d in tier1_docs)

    if tier1_total > budget and len(tier1_docs) > 1:
        # Proportional allocation across all Tier 1 docs
        selected = _allocate_proportional(tier1_docs, budget)
        tokens_used = sum(estimate_tokens(d["body"]) for d in selected)
        excluded.extend(d["title"] for d in docs_sorted if d["tier"] in (2, 3))
    else:
        # Normal greedy: Tier 1 fits, add Tier 2 and 3
        for tier in [1, 2, 3]:
            tier_docs_cur = [d for d in docs_sorted if d["tier"] == tier]
            for doc in tier_docs_cur:
                doc_tokens = estimate_tokens(doc["body"])
                if tokens_used + doc_tokens <= budget:
                    selected.append(doc)
                    tokens_used += doc_tokens
                elif tier == 1 and tokens_used < budget:
                    remaining = budget - tokens_used
                    doc_copy = dict(doc)
                    doc_copy["body"] = doc["body"][:remaining * 4] + "\n[...truncated]"
                    selected.append(doc_copy)
                    tokens_used = budget
                else:
                    excluded.append(doc["title"])

    excluded.extend(d["title"] for d in docs_sorted if d["tier"] == 4)

    # Build structured prompt
    parts = [f"## Case: {case_name}\n"]

    if timeline_entries:
        parts.append("### Case Timeline")
        parts.extend(timeline_entries[:15])  # cap timeline at 15 entries
        parts.append("")

    parts.append("### Key Documents (ordered by importance)\n")

    for doc in selected:
        label = TIER_LABELS[doc["tier"]]
        date_str = f", {doc['date'][:10]}" if doc["date"] else ""
        parts.append(f"#### [{label}] {doc['title'][:80]}{date_str}")
        parts.append(doc["body"])
        parts.append("")

    # Instructions at the end — closer to generation for better attention
    parts.append(
        "### Instructions\n"
        "Write a concise case summary in past tense covering: the dispute and stakes, "
        "filing date and court, whether class or individual action, type of counsel, "
        "legal claims and statutory basis, key procedural events, and outcome or "
        "current status."
    )

    new_prompt = "\n".join(parts)

    metadata = {
        "strategy": "structured",
        "total_docs": len(docs),
        "selected_docs": len(selected),
        "excluded_docs": len(excluded),
        "excluded_doc_types": excluded,
        "prompt_tokens_est": estimate_tokens(new_prompt),
        "timeline_entries": len(timeline_entries),
    }

    return new_prompt, metadata


# ── Extraction cache ──────────────────────────────────────────────────────────

def _cache_key(title: str, body: str) -> str:
    """Stable hash for caching extractions — based on title + first 2000 chars of body."""
    content = f"{title}|{body[:2000]}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _load_cache(cache_dir: Path | None) -> dict:
    if cache_dir is None:
        return {}
    cache_file = cache_dir / "extractions.jsonl"
    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    cache[entry["key"]] = entry["extraction"]
    return cache


def _save_cache_entry(cache_dir: Path | None, key: str, extraction: str) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "extractions.jsonl"
    with open(cache_file, "a") as f:
        f.write(json.dumps({"key": key, "extraction": extraction}, ensure_ascii=False) + "\n")


# ── Claude API extraction ────────────────────────────────────────────────────

def _extract_with_claude(
    doc: dict,
    client,
    model: str = "claude-sonnet-4-20250514",
    max_doc_chars: int = 400_000,  # ~100K tokens, safe for Claude's 200K context
) -> str:
    """Use Claude API to extract structured facts from a single legal document.

    Claude's large context window (200K) means we can send even very long documents
    without truncation in most cases.
    """
    extraction_prompt = get_extraction_prompt(doc["title"], doc["doc_type"])
    body = doc["body"][:max_doc_chars]

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=extraction_prompt,
        messages=[{"role": "user", "content": body}],
    )
    return response.content[0].text.strip()


# ── Strategy: Extract First ──────────────────────────────────────────────────

def strategy_extract_first(
    prompt_text: str,
    max_tokens: int = 24000,
    model_name: str | None = None,
    _model_cache: dict | None = None,
    extraction_backend: str = "heuristic",
    _claude_client=None,
    _extraction_cache: dict | None = None,
    _cache_dir: Path | None = None,
) -> tuple[str, dict]:
    """Two-stage: extract structured facts from each document, then build
    a condensed prompt for the synthesis task.

    Extraction backends:
        'claude'    — Use Claude API (best quality, requires ANTHROPIC_API_KEY)
        'model'     — Use a local HuggingFace model
        'heuristic' — Rule-based extraction (no API needed, lower quality)

    The Claude backend can compress a 50K-token opinion into ~2K tokens of
    structured facts, letting you fit ALL documents from a case into context.

    Returns (new_prompt, metadata_dict).
    """
    docs, case_name, preamble = parse_chunks(prompt_text)

    if not docs:
        return prompt_text, {"strategy": "extract_first", "error": "no_docs_parsed"}

    # Filter out tier 4 documents
    relevant_docs = [d for d in docs if d["tier"] <= 3]
    if not relevant_docs:
        relevant_docs = docs[:3]  # fallback: take first 3

    extractions = []
    extraction_method = extraction_backend
    cache = _extraction_cache or {}
    cache_hits = 0

    for doc in relevant_docs:
        # Check cache first
        key = _cache_key(doc["title"], doc["body"])
        if key in cache:
            extractions.append({
                "title": doc["title"],
                "date": doc["date"],
                "tier": doc["tier"],
                "extraction": cache[key],
            })
            cache_hits += 1
            continue

        # Extract based on backend
        if extraction_backend == "claude" and _claude_client is not None:
            try:
                extraction = _extract_with_claude(doc, _claude_client)
            except Exception as e:
                print(f"    Claude extraction failed for '{doc['title'][:40]}': {e}")
                extraction = _heuristic_extract(doc)
                extraction_method = "claude+heuristic_fallback"

        elif extraction_backend == "model" and _model_cache and "model" in _model_cache:
            import torch
            tokenizer = _model_cache["tokenizer"]
            model = _model_cache["model"]

            extraction_prompt = get_extraction_prompt(doc["title"], doc["doc_type"])
            max_doc_chars = 100000  # ~25K tokens
            body = doc["body"][:max_doc_chars]

            messages = [
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": body},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids

            if input_ids.shape[1] > 30000:
                extraction = _heuristic_extract(doc)
            else:
                input_ids = input_ids.to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids, max_new_tokens=512, temperature=0.0, do_sample=False
                    )
                extraction = tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
                ).strip()
        else:
            extraction = _heuristic_extract(doc)

        # Save to cache
        cache[key] = extraction
        _save_cache_entry(_cache_dir, key, extraction)

        extractions.append({
            "title": doc["title"],
            "date": doc["date"],
            "tier": doc["tier"],
            "extraction": extraction,
        })

    # Build condensed synthesis prompt
    parts = [f"## Case: {case_name}\n"]
    parts.append(
        "The following are extracted key facts from court documents in this case. "
        "Documents are ordered by importance (PRIMARY first).\n"
    )

    tokens_used = 0
    budget = max_tokens - 300
    docs_included = 0

    # Sort extractions by tier so PRIMARY docs come first
    extractions.sort(key=lambda x: x["tier"])

    for ext in extractions:
        ext_tokens = estimate_tokens(ext["extraction"])
        if tokens_used + ext_tokens > budget:
            # Try to fit a truncated version
            remaining = budget - tokens_used
            if remaining > 200:
                truncated = ext["extraction"][:remaining * 4] + "\n[...truncated]"
                date_str = f" ({ext['date'][:10]})" if ext["date"] else ""
                tier_label = TIER_LABELS[ext["tier"]]
                parts.append(f"### [{tier_label}] {ext['title'][:80]}{date_str}")
                parts.append(truncated)
                parts.append("")
                docs_included += 1
            break

        date_str = f" ({ext['date'][:10]})" if ext["date"] else ""
        tier_label = TIER_LABELS[ext["tier"]]
        parts.append(f"### [{tier_label}] {ext['title'][:80]}{date_str}")
        parts.append(ext["extraction"])
        parts.append("")
        tokens_used += ext_tokens
        docs_included += 1

    parts.append(
        "### Instructions\n"
        "Using ONLY the extracted information above, write a concise case summary "
        "in past tense covering: the dispute and stakes, filing date and court, "
        "whether class or individual action, type of counsel, legal claims and "
        "statutory basis, key procedural events, and outcome or current status."
    )

    new_prompt = "\n".join(parts)

    metadata = {
        "strategy": "extract_first",
        "extraction_method": extraction_method,
        "total_docs": len(docs),
        "relevant_docs": len(relevant_docs),
        "extracted_docs": docs_included,
        "cache_hits": cache_hits,
        "prompt_tokens_est": estimate_tokens(new_prompt),
    }

    return new_prompt, metadata


def _heuristic_extract(doc: dict) -> str:
    """Extract key content heuristically from a legal document.

    Improved extraction that looks for structural signals in legal documents:
    - Case caption / header (first ~2 paragraphs)
    - Holdings / conclusions (paragraphs containing ruling language)
    - Final paragraph (usually disposition / outcome)

    Falls back to first-2-last-1 for documents without clear structure.
    """
    body = doc["body"].strip()
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

    if len(paragraphs) <= 4:
        # Short doc — keep it all, cap at ~3000 tokens
        result = body
        if len(result) > 12000:
            result = result[:12000] + "\n[...truncated]"
        return result

    # Budget: ~1500 tokens per document (6000 chars) — allows ~16 docs in 24K
    max_chars = 6000

    # Always include: first 2 paragraphs (caption, intro, procedural posture)
    head = "\n\n".join(paragraphs[:2])

    # Look for key signal paragraphs in the middle
    holding_keywords = [
        "granted", "denied", "dismissed", "affirm", "revers", "remand",
        "ordered", "adjudged", "decreed", "we hold", "the court finds",
        "it is so ordered", "conclusion", "hereby", "accordingly",
        "for the foregoing reasons", "for these reasons",
        "judgment is entered", "summary judgment",
    ]

    # Score paragraphs by density of holding/conclusion keywords
    middle_paras = paragraphs[2:-1]
    scored = []
    for para in middle_paras:
        para_lower = para.lower()
        score = sum(1 for kw in holding_keywords if kw in para_lower)
        if score > 0:
            scored.append((score, para))

    # Take top 2 signal paragraphs from the middle
    scored.sort(key=lambda x: -x[0])
    middle_picks = [para for _, para in scored[:2]]

    # Last paragraph (usually disposition)
    tail = paragraphs[-1]

    # Assemble
    parts = [head]
    if middle_picks:
        parts.append("[...key holdings...]")
        parts.extend(middle_picks)
    parts.append("[...]")
    parts.append(tail)

    extracted = "\n\n".join(parts)

    if len(extracted) > max_chars:
        extracted = extracted[:max_chars] + "\n[...truncated]"

    return extracted


# ── Main ──────────────────────────────────────────────────────────────────────

STRATEGIES = {
    "priority_filter": strategy_priority_filter,
    "structured": strategy_structured,
    "extract_first": strategy_extract_first,
}


def process_record(rec: dict, strategy: str, max_tokens: int, **kwargs) -> dict:
    """Process a single training record with the chosen strategy."""
    strategy_fn = STRATEGIES[strategy]

    if strategy == "extract_first":
        new_prompt, meta = strategy_fn(rec["prompt"], max_tokens, **kwargs)
    else:
        new_prompt, meta = strategy_fn(rec["prompt"], max_tokens)

    return {
        "id": rec["id"],
        "case_id": rec["case_id"],
        "split": rec.get("split", ""),
        "prompt": new_prompt,
        "response": rec["response"],
        "completion": rec.get("completion", rec["response"]),
        "source_chunk_count": rec.get("source_chunk_count", 0),
        "used_chunk_count": meta.get("selected_docs", meta.get("extracted_docs",
                                     rec.get("used_chunk_count", 0))),
        **meta,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare improved training data with document-aware fragmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Quick test with stats:
  python scripts/prepare_training_data.py \\
      --input data/training/train.jsonl --strategy priority_filter --dry-run --sample 50

  # Full run with structured prompts:
  python scripts/prepare_training_data.py \\
      --input data/training/train.jsonl --output data/training_v2/train.jsonl \\
      --strategy structured --max-tokens 24000

  # Two-stage extraction with Claude API (best quality):
  python scripts/prepare_training_data.py \\
      --input data/training/train.jsonl --output data/training_v2/train_extracted.jsonl \\
      --strategy extract_first --extraction-backend claude \\
      --cache-dir data/extraction_cache --max-tokens 24000
        """,
    )
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", help="Output JSONL file path (omit for dry run)")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=list(STRATEGIES.keys()),
        help="Fragmentation strategy to apply",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=24000,
        help="Maximum token budget for prompts (default: 24000)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only first N records (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show statistics without writing output",
    )
    parser.add_argument(
        "--extraction-backend",
        choices=["heuristic", "claude", "model"],
        default="heuristic",
        help="Extraction backend for extract_first strategy: "
             "'claude' (best, requires ANTHROPIC_API_KEY), "
             "'model' (local HF model), "
             "'heuristic' (rule-based, no API needed, default)",
    )
    parser.add_argument(
        "--extraction-model",
        type=str,
        default=None,
        help="Model name for --extraction-backend model (e.g., Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache Claude/model extractions (avoids re-extracting on reruns)",
    )
    parser.add_argument(
        "--claude-model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model for extraction (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--claude-concurrency",
        type=int,
        default=5,
        help="Max concurrent Claude API calls (default: 5)",
    )
    args = parser.parse_args()

    # ── Set up extraction backend ──

    claude_client = None
    model_cache = {}
    extraction_cache = {}
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if args.strategy == "extract_first":
        # Load extraction cache
        if cache_dir:
            extraction_cache = _load_cache(cache_dir)
            print(f"Loaded {len(extraction_cache)} cached extractions from {cache_dir}")

        if args.extraction_backend == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("ERROR: --extraction-backend claude requires ANTHROPIC_API_KEY env var")
                sys.exit(1)
            try:
                import anthropic
                claude_client = anthropic.Anthropic(api_key=api_key)
                print(f"Claude extraction enabled (model: {args.claude_model})")
            except ImportError:
                print("ERROR: pip install anthropic is required for Claude extraction")
                sys.exit(1)

        elif args.extraction_backend == "model":
            if not args.extraction_model:
                print("ERROR: --extraction-backend model requires --extraction-model")
                sys.exit(1)
            print(f"Loading extraction model: {args.extraction_model}...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_cache["tokenizer"] = AutoTokenizer.from_pretrained(args.extraction_model)
            model_cache["model"] = AutoModelForCausalLM.from_pretrained(
                args.extraction_model, torch_dtype=torch.float16, device_map="auto"
            )
            model_cache["model"].eval()
            print("Model loaded.")

        else:
            print("Using heuristic extraction (no API calls)")

    # ── Process records ──

    input_path = Path(args.input)
    records_in = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                records_in.append(json.loads(line))
            if args.sample and len(records_in) >= args.sample:
                break

    print(f"Loaded {len(records_in)} records from {input_path}")
    print(f"Strategy: {args.strategy}, max tokens: {args.max_tokens}")

    results = []
    token_counts = []
    skipped = 0
    start_time = time.time()

    for i, rec in enumerate(records_in):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(records_in) - i) / rate if rate > 0 else 0
            print(f"  Processed {i}/{len(records_in)} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)...")

        try:
            kwargs = {}
            if args.strategy == "extract_first":
                kwargs["extraction_backend"] = args.extraction_backend
                kwargs["_claude_client"] = claude_client
                kwargs["model_name"] = args.extraction_model
                kwargs["_model_cache"] = model_cache if model_cache else None
                kwargs["_extraction_cache"] = extraction_cache
                kwargs["_cache_dir"] = cache_dir

            result = process_record(rec, args.strategy, args.max_tokens, **kwargs)

            # Skip records with empty responses
            if not result["response"].strip():
                skipped += 1
                continue

            results.append(result)
            token_counts.append(result.get("prompt_tokens_est", estimate_tokens(result["prompt"])))
        except Exception as e:
            print(f"  Error on record {rec.get('id', i)}: {e}")
            skipped += 1

    elapsed = time.time() - start_time

    # ── Print statistics ──

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(results)} processed, {skipped} skipped, {elapsed:.1f}s elapsed")
    print(f"{'='*60}")

    if token_counts:
        import statistics

        print(f"Prompt tokens (estimated):")
        print(f"  Min: {min(token_counts):,}")
        print(f"  Max: {max(token_counts):,}")
        print(f"  Mean: {statistics.mean(token_counts):,.0f}")
        print(f"  Median: {statistics.median(token_counts):,.0f}")

        fit = sum(1 for t in token_counts if t <= args.max_tokens)
        print(f"  Fit in {args.max_tokens} tokens: "
              f"{fit}/{len(token_counts)} ({100*fit/len(token_counts):.1f}%)")

        # Per-strategy stats
        if results and "extracted_docs" in results[0]:
            docs_included = [r.get("extracted_docs", 0) for r in results]
            docs_total = [r.get("relevant_docs", r.get("total_docs", 0)) for r in results]
            print(f"Documents included per record:")
            print(f"  Mean: {statistics.mean(docs_included):.1f} / {statistics.mean(docs_total):.1f}")
            print(f"  Median: {statistics.median(docs_included):.0f}")
            if any(r.get("cache_hits", 0) for r in results):
                total_hits = sum(r.get("cache_hits", 0) for r in results)
                print(f"  Cache hits: {total_hits}")
        elif results and "selected_docs" in results[0]:
            docs_selected = [r.get("selected_docs", 0) for r in results]
            docs_total = [r.get("total_docs", 0) for r in results]
            print(f"Documents selected per record:")
            print(f"  Mean: {statistics.mean(docs_selected):.1f} / {statistics.mean(docs_total):.1f}")
            print(f"  Median: {statistics.median(docs_selected):.0f}")

    if not args.dry_run and args.output:
        out_path = Path(args.output)
        os.makedirs(out_path.parent, exist_ok=True)
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(results)} records to {out_path}")
    elif args.dry_run:
        # Show samples
        if results:
            for idx in [0, min(len(results) - 1, len(results) // 2)]:
                sample = results[idx]
                print(f"\n--- Sample output (record {idx}) ---")
                print(f"Case: {sample['case_id']}")
                print(f"Strategy: {sample.get('strategy', 'unknown')}")
                print(f"Extraction method: {sample.get('extraction_method', 'n/a')}")
                print(f"Prompt tokens: {sample.get('prompt_tokens_est', '?')}")
                print(f"Docs selected/extracted: "
                      f"{sample.get('selected_docs', sample.get('extracted_docs', '?'))}")
                print(f"Prompt (first 800 chars):\n{sample['prompt'][:800]}")
    else:
        print("\nNo output path specified and not dry-run. Use --output or --dry-run.")


if __name__ == "__main__":
    main()
