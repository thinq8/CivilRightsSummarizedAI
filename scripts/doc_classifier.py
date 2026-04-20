"""Document type classification and priority tiering for legal case documents.

Classifies documents extracted from Clearinghouse case data into importance tiers
to support smart context budget allocation during training data preparation.

Usage:
    from scripts.doc_classifier import classify_document, TIER_LABELS

    tier = classify_document(title="Opinion", doc_type="Order/Opinion")
    # Returns: 1
"""

from __future__ import annotations

import re

# Tier definitions:
#   1 = MUST INCLUDE (core case narrative)
#   2 = INCLUDE IF SPACE (procedural context)
#   3 = INCLUDE SELECTIVELY (supporting detail)
#   4 = EXCLUDE by default (low signal)

TIER_LABELS = {
    1: "PRIMARY",
    2: "SUPPORTING",
    3: "CONTEXTUAL",
    4: "EXCLUDED",
}

# Patterns are matched case-insensitively against the document Title and Type fields.
# Order matters within a tier — first match wins.

_TIER_1_PATTERNS = [
    r"opinion",
    r"order",
    r"judgment",
    r"decree",
    r"consent\s+decree",
    r"settlement",
    r"appellate\s+decision",
    r"usca\s+opinion",
    r"per\s+curiam",
    r"memorandum\s+(opinion|decision)",
    r"report\s+and\s+recommendation",
    r"findings?\s+(of\s+fact|letter)",
]

_TIER_2_PATTERNS = [
    r"complaint",
    r"amended\s+complaint",
    r"class\s+action\s+complaint",
    r"motion\s+to\s+dismiss",
    r"motion\s+for\s+summary\s+judgment",
    r"class\s+cert",
    r"preliminary\s+injunction",
    r"temporary\s+restraining",
    r"brief\s+in\s+support",
    r"memorandum\s+in\s+(support|opposition)",
    r"motion\s+for\s+judgment",
    r"answer",
]

_TIER_3_PATTERNS = [
    r"reply",
    r"response",
    r"supplemental",
    r"amici?\s+curiae",
    r"brief\s+of\s+amici",
    r"memorandum\s+contra",
    r"objection",
    r"motion\s+for\s+leave",
    r"appeal",
    r"notice\s+of\s+appeal",
    r"motion",  # generic motion catch-all
    r"brief",  # generic brief catch-all
    r"memorandum",  # generic memo catch-all (after specific ones)
]

_TIER_4_PATTERNS = [
    r"docket",
    r"pacer",
    r"correspondence",
    r"letter",  # generic letters (findings letter caught in tier 1)
    r"newsletter",
    r"press\s+release",
    r"untitled",
    r"justice\s+department",  # press releases
]


def classify_document(title: str, doc_type: str = "") -> int:
    """Classify a document into a priority tier (1-4).

    Args:
        title: Document title from the chunk header (e.g., "Opinion", "Docket (PACER)")
        doc_type: Document type field if available (e.g., "Order/Opinion")

    Returns:
        Tier number (1=highest priority, 4=lowest/excluded)
    """
    combined = f"{title} {doc_type}".strip().lower()

    if not combined:
        return 4  # unknown → exclude

    # Check tiers in order (most important first)
    for tier, patterns in [
        (1, _TIER_1_PATTERNS),
        (4, _TIER_4_PATTERNS),  # Check tier 4 before 2/3 so "Docket" doesn't match "motion"
        (2, _TIER_2_PATTERNS),
        (3, _TIER_3_PATTERNS),
    ]:
        for pattern in patterns:
            if re.search(pattern, combined):
                return tier

    return 3  # unrecognized → include selectively


def parse_chunks(prompt_text: str) -> list[dict]:
    """Parse a training prompt into individual documents.

    Documents are separated by [DOCUMENT] markers within the prompt text.
    A single document may span multiple numbered chunks (chunks are just
    fixed-size splits at ~25KB boundaries, unrelated to document boundaries).

    Returns (documents, case_name, preamble) where documents is a list of dicts
    with keys: title, doc_type, date, body, tier, tier_label
    """
    # Extract preamble (everything before first [DOCUMENT])
    doc_splits = re.split(r"\[DOCUMENT\]", prompt_text)
    preamble = doc_splits[0].strip()

    # Remove "Chunk N:" markers from text — they're arbitrary split points
    def clean_chunk_markers(text):
        return re.sub(r"^Chunk\s+\d+:\s*\n?", "", text, flags=re.MULTILINE).strip()

    documents = []

    for fragment in doc_splits[1:]:  # skip preamble
        fragment = clean_chunk_markers(fragment).strip()
        if not fragment:
            continue

        # Parse metadata header
        title = ""
        doc_type = ""
        date = ""
        lines = fragment.split("\n")
        body_start = 0

        for j, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("Title:"):
                title = stripped[6:].strip().strip("[]")
            elif stripped.startswith("Type:"):
                doc_type = stripped[5:].strip()
            elif stripped.startswith("Date:"):
                date = stripped[5:].strip()
                body_start = j + 1
                break
            elif stripped and j > 0:
                # Non-metadata line after at least one header attempt
                body_start = j
                break

        body = clean_chunk_markers("\n".join(lines[body_start:])).strip()
        tier = classify_document(title, doc_type)

        documents.append({
            "title": title,
            "doc_type": doc_type,
            "date": date,
            "body": body,
            "tier": tier,
            "tier_label": TIER_LABELS[tier],
        })

    # Extract case name from preamble
    case_name = ""
    for line in preamble.split("\n"):
        if line.strip().startswith("Case:"):
            case_name = line.strip()[5:].strip()
            break

    return documents, case_name, preamble


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


if __name__ == "__main__":
    # Quick test with sample data
    import json
    import sys

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/training/test.jsonl"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Testing classifier on {n_samples} records from {test_path}\n")

    with open(test_path) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            rec = json.loads(line)
            docs, case_name, _ = parse_chunks(rec["prompt"])
            print(f"Case: {case_name} ({len(docs)} documents, {rec['source_chunk_count']} raw chunks)")
            for d in docs:
                tokens = estimate_tokens(d["body"])
                print(f"  Tier {d['tier']} [{d['tier_label']:11s}] {tokens:6d} tok  {d['title'][:60]}")
            print()
