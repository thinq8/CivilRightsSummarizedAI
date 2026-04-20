# Standalone Review Tools

These tools are partner-facing prototypes. They are designed to be easy to open and inspect, not to replace the Clearinghouse production website.

One-command local demo:

```bash
python tools/clearinghouse_api_proxy.py
```

Then open <http://127.0.0.1:8765/>.

## Tool Summary

| Tool | How to run | What it does |
|------|------------|--------------|
| Tool | Key requirement | What it does |
|------|-----------------|--------------|
| `summary_qa_standalone.html` | Browser only | Paste one summary and run structural QA checks offline |
| `case-summary-generator.html` | Browser; API token only for live loading | Draft a summary from case metadata, selected source documents, or API-loaded case data |
| `case-summary-evaluator.html` | Browser; `ANTHROPIC_API_KEY` only for optional Claude judging | Review a generated package, run local QA, optionally compare to a reference, and export feedback |
| `clearinghouse_api_proxy.py` | Python | Serves the browser tools on localhost and forwards only Clearinghouse API v2.1 requests |
| `case_review_tool.py` | Python; `ANTHROPIC_API_KEY` only for judge mode | Batch reference-free review with structural QA and optional Claude source-citation judging |
| `case_review_tool.html` | Browser only | Earlier single-page review dashboard retained for reference |

## Recommended Partner Workflow

1. Start the local proxy with `python tools/clearinghouse_api_proxy.py`.
2. Open <http://127.0.0.1:8765/>.
3. Generate a draft from metadata-only input for a thin/simple case, or metadata plus selected documents for a complex case.
4. Export the generator package.
5. Open the evaluator and import the package.
6. Run local QA, review source grounding, add human feedback, and export the review JSON.

This workflow matches the final recommendation in `REPORT.md`: use automation to create and triage drafts, but keep a human editor in the loop.

## QA Checker

`summary_qa_standalone.html` is a browser port of `scripts/summary_qa.py`.

It checks for:

- garbled dates and malformed years
- raw document artifacts and prompt leakage
- repetition loops and length collapse
- missing filing date, court, statute, remedy, or action type
- statute citation mismatches when a reference summary is provided

The browser tool has no server dependency and no API key requirement.

```bash
open tools/summary_qa_standalone.html
```

## Local API Proxy

Browsers block local HTML files from making authenticated cross-origin API calls directly. `clearinghouse_api_proxy.py` solves that for testing by serving the tools from `127.0.0.1` and forwarding authenticated requests to:

```text
https://clearinghouse.net/api/v2p1/
```

The proxy rejects other hosts and other API paths. It is for local testing only.

## Batch Review Tool

`case_review_tool.py` is useful when you already have generated summaries in JSONL form.

```bash
python tools/case_review_tool.py \
  --eval eval_claude_code.jsonl \
  --sources data/training/test.jsonl \
  --output review/
```

Use `--no-judge` for free structural QA only. Set `ANTHROPIC_API_KEY` when using Claude judging.

## Adding or Changing QA Flags

Keep the Python and browser versions aligned:

1. Add or update the check in `scripts/summary_qa.py`.
2. Add or update the matching JavaScript check in `tools/summary_qa_standalone.html`.
3. Update this README if the new flag changes how reviewers should interpret `PASS`, `REVIEW`, or `REJECT`.
