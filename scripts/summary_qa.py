#!/usr/bin/env python3
"""Civil Rights Case Summary — Quality Assurance & Human-Review Triage.

Runs a battery of robustness checks over case summaries (AI-generated or
human-written) and flags records that need human review. Built around the
failure modes we saw in our Qwen2.5-7B-LoRA evaluation plus the Clearinghouse
rubric for required editorial elements.

Usage
-----
  # Batch over our eval JSONL (has prediction + reference):
  python scripts/summary_qa.py \
      --input Polish/eval2/eval_ckpt3000.jsonl \
      --output-dir Polish/eval2/qa_report_ckpt3000/

  # Single-file text input (prediction only, no reference):
  python scripts/summary_qa.py \
      --input some_summary.txt \
      --output-dir qa_single/

  # Skip heavy metrics (structural checks only):
  python scripts/summary_qa.py --input eval.jsonl --no-rouge --output-dir qa/

The tool produces three artifacts in the output directory:

    qa_report.html    - self-contained dashboard (show this to a business user)
    qa_report.csv     - flat Excel-friendly view
    qa_report.jsonl   - structured per-record output for re-ingestion

Library usage
-------------
  from summary_qa import SummaryQAChecker
  checker = SummaryQAChecker()
  report = checker.check(prediction="...", reference="...")
  print(report.status, len(report.flags))
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────


Severity = Literal["critical", "warning", "info"]
Category = Literal["format", "content", "element", "metric", "judge"]


@dataclass
class Flag:
    severity: Severity
    category: Category
    code: str           # machine-readable (GARBLED_DATE, MISSING_FILING_DATE, …)
    message: str        # human-readable
    evidence: str = ""  # offending substring or metric value


@dataclass
class QAReport:
    identifier: str                  # case_id or index
    prediction: str
    reference: str | None
    word_count: int
    flags: list[Flag] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "info")

    @property
    def status(self) -> Literal["PASS", "REVIEW", "REJECT"]:
        if self.critical_count > 0:
            return "REJECT"
        if self.warning_count > 0:
            return "REVIEW"
        return "PASS"

    def to_dict(self) -> dict:
        return {
            "identifier": self.identifier,
            "status": self.status,
            "word_count": self.word_count,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "flags": [asdict(f) for f in self.flags],
            "metrics": self.metrics,
            "prediction": self.prediction,
            "reference": self.reference,
        }

    def top_flag_codes(self, n: int = 3) -> list[str]:
        ordered = sorted(
            self.flags,
            key=lambda f: {"critical": 0, "warning": 1, "info": 2}[f.severity],
        )
        return [f.code for f in ordered[:n]]


# ──────────────────────────────────────────────────────────────────────────────
# Checker
# ──────────────────────────────────────────────────────────────────────────────


# Required Clearinghouse elements (for regex-based heuristics)
MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|"
    r"July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

# Common failure-mode regexes
PLEADING_CAPTION = re.compile(
    r"(?mi)^\s*(?:Plaintiffs?,?|Defendants?,?)\s*[)\]].*$|^\s*v\.\s*[)\]]|"
    r"^\s*(?:UNITED STATES )?DISTRICT COURT\b|"
    r"^\s*Case No\.?\s*[:#]|^\s*Civil Action No\.?|"
    r"^\s*/s/|^\s*SO ORDERED\.?\s*$|^\s*IT IS SO ORDERED\.?\s*$"
)

LINE_NUMBER_BLOCK = re.compile(r"(?m)(^\s*\d{1,2}\s+\S.*(?:\n|$)){3,}")

META_PROMPT_PATTERNS = re.compile(
    r"(?i)\b(as an ai|i (?:will|am|can|'ll|'m) (?:now |provide|be )?(?:summariz|creat|generat|writ|provid)|"
    r"here(?:'s| is) (?:a |the |your )?(?:summary|case summary)|"
    r"based on the (?:provided|given|above) (?:documents?|text|case|information)|"
    r"the following is a summary|summary:\s*\n)"
)

BRITISH_JUDGMENT = re.compile(r"\bjudgement\b")

PRESENT_TENSE_VERBS = re.compile(
    r"\b(?:files|sues|alleges|argues|claims|seeks|brings|moves|asks|demands|challenges)\b"
)

# Required-element cues
FILING_DATE_CUES = re.compile(
    rf"\b(?:filed|commenced|brought|initiated)\b.*?(?:{MONTHS}|\d{{4}})|"
    rf"(?:on|in)\s+{MONTHS}\s+\d{{1,2}},?\s*\d{{4}}|"
    rf"{MONTHS}\s+\d{{1,2}},?\s*\d{{4}}",
    re.IGNORECASE,
)

COURT_CUES = re.compile(
    r"\b(?:U\.S\. District Court|United States District Court|Circuit Court|"
    r"Court of Appeals|Supreme Court|District of [A-Z][a-z]+|"
    r"Eastern District|Western District|Northern District|Southern District|"
    r"Middle District|[A-Z][a-zA-Z]+ County Court|"
    r"Court of Common Pleas|Chancery Court|Superior Court)\b"
)

STATUTE_CUES = re.compile(
    r"\b(?:42 U\.?S\.?C\.?|28 U\.?S\.?C\.?|Title [IVX]+|Title VII|"
    r"Americans with Disabilities Act|ADA\b|Section 1983|§\s*1983|"
    r"First Amendment|Fourth Amendment|Fifth Amendment|Eighth Amendment|"
    r"Fourteenth Amendment|Equal Protection|Due Process|"
    r"Voting Rights Act|Civil Rights Act|Rehabilitation Act|"
    r"Fair Housing Act|FHA\b|IDEA\b|FMLA\b|RFRA\b|IDEA\b|FLSA\b)\b"
)

REMEDY_CUES = re.compile(
    r"\b(?:damages|injunctive relief|declaratory (?:judgment|relief)|"
    r"compensatory|punitive|attorneys?'?\s*fees|restitution|"
    r"preliminary injunction|permanent injunction|"
    r"(?:sought|seeking|requested|requesting)\s+(?:relief|damages|an? (?:order|injunction)))\b",
    re.IGNORECASE,
)

OUTCOME_CUES = re.compile(
    r"\b(?:dismissed|granted|denied|settled|affirmed|reversed|remanded|"
    r"entered|approved|certified|judgment (?:for|against)|"
    r"court (?:held|ruled|found|ordered|entered)|class certification)\b",
    re.IGNORECASE,
)

CLASS_VS_INDIVIDUAL = re.compile(
    r"\b(?:class action|class of|putative class|individual capacity|pro se)\b",
    re.IGNORECASE,
)


class SummaryQAChecker:
    """Runs all QA checks and produces a QAReport per summary.

    Parameters
    ----------
    min_words, max_words : int
        Word-count bounds. Summaries outside are flagged.
    rouge_warn, rouge_crit : float
        ROUGE-Lsum thresholds for warning/critical.
    bertscore_warn, bertscore_crit : float
        BERTScore-F1 thresholds.
    enable_rouge, enable_bertscore : bool
        Whether to compute those metrics (require reference).
    """

    def __init__(
        self,
        min_words: int = 80,
        max_words: int = 2000,
        length_ratio_warn: tuple[float, float] = (0.4, 2.5),
        rouge_warn: float = 0.25,
        rouge_crit: float = 0.15,
        bertscore_warn: float = 0.0,
        bertscore_crit: float = -0.1,
        enable_rouge: bool = True,
        enable_bertscore: bool = False,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.length_ratio_warn = length_ratio_warn
        self.rouge_warn = rouge_warn
        self.rouge_crit = rouge_crit
        self.bertscore_warn = bertscore_warn
        self.bertscore_crit = bertscore_crit
        self.enable_rouge = enable_rouge
        self.enable_bertscore = enable_bertscore

        self._rouge_scorer = None
        if enable_rouge:
            try:
                from rouge_score import rouge_scorer
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True,
                )
            except ImportError:
                print("WARNING: rouge_score not installed; disabling ROUGE checks",
                      file=sys.stderr)
                self.enable_rouge = False

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        prediction: str,
        reference: str | None = None,
        identifier: str = "",
    ) -> QAReport:
        prediction = prediction or ""
        words = prediction.split()
        report = QAReport(
            identifier=identifier or "?",
            prediction=prediction,
            reference=reference,
            word_count=len(words),
        )

        # Mechanical checks (always run)
        self._check_length(report)
        self._check_truncation(report)
        self._check_garbled_years(report)
        self._check_raw_document_artifacts(report)
        self._check_repetition(report)
        self._check_meta_prompt_leakage(report)
        self._check_legal_style(report)
        self._check_suspicious_spelling(report)

        # Required-element checks (always run)
        self._check_required_elements(report)

        # Reference-dependent metric checks
        if reference:
            self._check_length_ratio(report)
            if self.enable_rouge and self._rouge_scorer is not None:
                self._check_rouge(report)
            if self.enable_bertscore:
                self._check_bertscore(report)

        return report

    # ── Mechanical checks ─────────────────────────────────────────────────────

    def _check_length(self, r: QAReport) -> None:
        wc = r.word_count
        if wc < 30:
            r.flags.append(Flag(
                "critical", "format", "EXTREMELY_SHORT",
                f"Summary is only {wc} words — far too short to cover required elements.",
                evidence=f"{wc} words",
            ))
        elif wc < self.min_words:
            r.flags.append(Flag(
                "warning", "format", "TOO_SHORT",
                f"Summary ({wc} words) is below the {self.min_words}-word minimum.",
                evidence=f"{wc} words",
            ))
        elif wc > self.max_words:
            r.flags.append(Flag(
                "warning", "format", "TOO_LONG",
                f"Summary ({wc} words) exceeds the {self.max_words}-word budget.",
                evidence=f"{wc} words",
            ))

    def _check_truncation(self, r: QAReport) -> None:
        text = r.prediction.rstrip()
        if not text:
            return
        last_char = text[-1]
        if last_char not in '.!?"\')]':
            # Suspicious ending
            last_line = text.splitlines()[-1] if text.splitlines() else text
            r.flags.append(Flag(
                "critical", "format", "UNTERMINATED",
                "Summary does not end with terminal punctuation — likely truncated "
                "mid-sentence or hit the max-token budget.",
                evidence=f"…{last_line[-60:]}",
            ))
            return
        # Ends with period but last sentence is suspiciously short or ellipsis-like
        if text.endswith("...") or text.endswith(".."):
            r.flags.append(Flag(
                "warning", "format", "ELLIPSIS_END",
                "Summary ends in an ellipsis — may indicate content was dropped.",
                evidence=text[-20:],
            ))

    def _check_garbled_years(self, r: QAReport) -> None:
        """Detect the overfitting-artifact dates we saw in ckpt-3690:
        '210', '2120', '2k00', '21020', '2oo4', '22016', etc.
        """
        text = r.prediction
        current_year = datetime.now().year

        # Pattern 1: mixed alphanumeric tokens that look like botched years
        # e.g. 2k00, 2oo4, 20o3
        for m in re.finditer(r"\b(?=\S*\d)(?=\S*[A-Za-z])[A-Za-z0-9]{3,6}\b", text):
            token = m.group()
            # Skip ordinals (1st, 2nd, 3rd, 4th ... 31st)
            if re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)", token, re.IGNORECASE):
                continue
            # Skip common legal abbreviations and docket refs
            if re.fullmatch(
                r"(?:Nos?|Count|Doc|ECF|ID|PL|DEF|No\d+|Fed|Cir|Cal|Ill|Mich|"
                r"Tex|Va|Wis|Pa|Ga|Md|Mo|Or|NC|NY|NJ|CA|DC|SD|ED|WD|MD|ND)\d+",
                token, re.IGNORECASE,
            ):
                continue
            # Skip percentages, counts, versions (10x, 3k, v2, p3)
            if re.fullmatch(r"\d+[xkKm]\d*|v\d+|p\d+|§\d+", token):
                continue
            # Must have at least one letter between digits or vice versa
            if re.search(r"\d[A-Za-z]|[A-Za-z]\d", token):
                r.flags.append(Flag(
                    "critical", "format", "GARBLED_DATE",
                    f"Alphanumeric token '{token}' looks like a garbled year "
                    f"(common overfitting artifact).",
                    evidence=token,
                ))

        # Pattern 2: 5+ digit numbers in year context
        for m in re.finditer(r"\b\d{5,}\b", text):
            token = m.group()
            start = max(0, m.start() - 40)
            ctx_before = text[start:m.start()].lower()
            # Likely year contexts
            if re.search(rf"\b(?:in|on|since|until|filed|year|dated|of|{MONTHS.lower()})\s*,?\s*$",
                         ctx_before):
                r.flags.append(Flag(
                    "critical", "format", "GARBLED_DATE",
                    f"{len(token)}-digit number '{token}' in year context — "
                    f"likely garbled year.",
                    evidence=token,
                ))

        # Pattern 3: 3-digit year-like numbers starting with 19/20 (e.g. "210", "201")
        # Only flag if they appear in a date context to avoid false positives on
        # case numbers, section citations, addresses.
        three_digit_in_context = re.compile(
            rf"(?:{MONTHS}\s+\d{{1,2}},?\s*|"
            rf"\b(?:in|on|since|until|filed|year|dated)\s+)(\d{{3}})\b",
            re.IGNORECASE,
        )
        for m in three_digit_in_context.finditer(text):
            token = m.group(1)
            if token.startswith(("19", "20", "21")):
                r.flags.append(Flag(
                    "critical", "format", "GARBLED_DATE",
                    f"3-digit 'year' {token!r} in date context — truncated "
                    f"year digit.",
                    evidence=m.group(),
                ))

        # Pattern 4: implausible 4-digit years (>current or <1700)
        for m in re.finditer(r"\b(\d{4})\b", text):
            year = int(m.group(1))
            if year < 1700 or year > current_year + 1:
                start = max(0, m.start() - 30)
                ctx_before = text[start:m.start()].lower()
                if re.search(rf"\b(?:in|on|since|until|filed|year|dated|of|{MONTHS.lower()})\s*,?\s*$",
                             ctx_before):
                    r.flags.append(Flag(
                        "warning", "format", "IMPLAUSIBLE_YEAR",
                        f"Year {year} is outside plausible range (1700-{current_year + 1}).",
                        evidence=m.group(),
                    ))

    def _check_raw_document_artifacts(self, r: QAReport) -> None:
        """Catch cases where the model regurgitates raw court-filing text
        instead of producing a summary.
        """
        text = r.prediction

        # Pleading-style captions (strong signal of raw filing text)
        pleading_matches = PLEADING_CAPTION.findall(text)
        if pleading_matches:
            r.flags.append(Flag(
                "critical", "format", "RAW_DOCUMENT_ARTIFACT",
                "Summary contains raw pleading or order text (caption, signature "
                "block, or court header) — model is regurgitating source "
                "documents instead of summarizing.",
                evidence=str(pleading_matches[0])[:80],
            ))

        # Consecutive line-numbered transcript block
        if LINE_NUMBER_BLOCK.search(text):
            r.flags.append(Flag(
                "critical", "format", "RAW_DOCUMENT_ARTIFACT",
                "Summary contains consecutive line-numbered lines — likely a "
                "raw transcript or brief block.",
                evidence="line-numbered block detected",
            ))

        # Excessive ALL-CAPS runs (>4 consecutive all-caps words)
        caps_run = re.search(r"(?:\b[A-Z]{2,}\b\s+){4,}", text)
        if caps_run:
            r.flags.append(Flag(
                "warning", "format", "EXCESSIVE_CAPS",
                "Extended run of ALL-CAPS words — may indicate a court header "
                "or shout block.",
                evidence=caps_run.group()[:80],
            ))

    def _check_repetition(self, r: QAReport) -> None:
        text = r.prediction
        # Sentence-level repetition
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        seen = Counter(sentences)
        for sent, count in seen.most_common():
            if count >= 2 and len(sent) > 40:
                r.flags.append(Flag(
                    "warning", "format", "DUPLICATE_SENTENCE",
                    f"Sentence appears {count} times.",
                    evidence=sent[:100],
                ))
                break  # only report the worst one

        # N-gram loop: any 8-gram appearing 3+ times
        tokens = text.split()
        if len(tokens) >= 24:
            grams = Counter(
                " ".join(tokens[i:i + 8]) for i in range(len(tokens) - 7)
            )
            for gram, count in grams.most_common(1):
                if count >= 3:
                    r.flags.append(Flag(
                        "critical", "format", "REPETITION_LOOP",
                        f"8-word phrase repeats {count} times — model is stuck "
                        f"in a loop.",
                        evidence=gram[:80],
                    ))

    def _check_meta_prompt_leakage(self, r: QAReport) -> None:
        m = META_PROMPT_PATTERNS.search(r.prediction)
        if m:
            r.flags.append(Flag(
                "critical", "format", "META_PROMPT_LEAK",
                "Summary contains meta-prompt or self-referential AI language.",
                evidence=m.group()[:80],
            ))

    def _check_legal_style(self, r: QAReport) -> None:
        text = r.prediction

        # Spelling convention: "judgement" -> "judgment"
        if BRITISH_JUDGMENT.search(text):
            r.flags.append(Flag(
                "info", "content", "WRONG_JUDGMENT_SPELLING",
                "'Judgement' should be 'judgment' in U.S. legal usage.",
                evidence="judgement",
            ))

        # Tense check: if the present-tense narrative verbs are dominant,
        # warn. Clearinghouse style is past tense for procedural history.
        pt = len(PRESENT_TENSE_VERBS.findall(text))
        past_tense_matches = len(re.findall(
            r"\b(?:filed|sued|alleged|argued|claimed|sought|brought|moved|"
            r"asked|demanded|challenged|held|ruled|found)\b", text,
        ))
        if pt >= 3 and pt > past_tense_matches:
            r.flags.append(Flag(
                "warning", "content", "WRONG_TENSE",
                f"Summary appears to use present tense ({pt} present-tense verbs "
                f"vs {past_tense_matches} past-tense) — Clearinghouse style is "
                f"past tense.",
                evidence=f"{pt} present / {past_tense_matches} past",
            ))

    def _check_suspicious_spelling(self, r: QAReport) -> None:
        """Detect the misspellings we saw from overfitting: doubled consonants,
        mangled vowels, dropped letters. This is heuristic — flags candidate
        tokens but lets human review confirm."""
        text = r.prediction
        suspicious = []
        # Tokens with 3+ consecutive same letters (except ll, ee, etc.)
        for m in re.finditer(r"\b\w*([a-zA-Z])\1{2,}\w*\b", text):
            suspicious.append(m.group())
        # Tokens with unusual consonant clusters
        patterns = [
            r"\b\w*iiv\w*\b",      # Juvinile, Juveine
            r"\b\w*iin\w*\b",      # Discriimination
            r"\b\w*(?:jn|hn|kz|wv)\w*\b",
            r"\bYour[kt]\b(?!shire)",  # Yourk instead of York
        ]
        for p in patterns:
            for m in re.finditer(p, text, re.IGNORECASE):
                tok = m.group()
                if tok.lower() not in {"you", "your"}:
                    suspicious.append(tok)

        if suspicious:
            unique = list(dict.fromkeys(suspicious))[:5]
            r.flags.append(Flag(
                "warning", "format", "SUSPICIOUS_SPELLING",
                f"Found {len(unique)} possibly misspelled token(s).",
                evidence=", ".join(unique),
            ))

    # ── Required-element checks (Clearinghouse rubric) ────────────────────────

    def _check_required_elements(self, r: QAReport) -> None:
        text = r.prediction

        if not FILING_DATE_CUES.search(text):
            r.flags.append(Flag(
                "warning", "element", "MISSING_FILING_DATE",
                "No filing date detected — required Clearinghouse element.",
            ))

        if not COURT_CUES.search(text):
            r.flags.append(Flag(
                "warning", "element", "MISSING_COURT",
                "No court name detected — required Clearinghouse element.",
            ))

        if not STATUTE_CUES.search(text):
            r.flags.append(Flag(
                "warning", "element", "MISSING_STATUTE",
                "No statutory or constitutional basis detected — required "
                "Clearinghouse element.",
            ))

        if not REMEDY_CUES.search(text):
            r.flags.append(Flag(
                "info", "element", "MISSING_REMEDY",
                "No remedy or relief language detected — may need one.",
            ))

        if not OUTCOME_CUES.search(text):
            r.flags.append(Flag(
                "info", "element", "MISSING_OUTCOME",
                "No disposition or outcome language detected — may need one.",
            ))

        if not CLASS_VS_INDIVIDUAL.search(text):
            r.flags.append(Flag(
                "info", "element", "MISSING_ACTION_TYPE",
                "Did not specify class vs. individual action — recommended.",
            ))

    # ── Metric checks ─────────────────────────────────────────────────────────

    def _check_length_ratio(self, r: QAReport) -> None:
        if not r.reference:
            return
        ref_wc = len(r.reference.split())
        if ref_wc == 0:
            return
        ratio = r.word_count / ref_wc
        r.metrics["length_ratio"] = round(ratio, 3)
        lo, hi = self.length_ratio_warn
        if ratio < lo:
            r.flags.append(Flag(
                "warning", "metric", "LENGTH_TOO_SHORT_VS_REF",
                f"Summary is {ratio:.0%} of reference length — likely missing "
                f"required content.",
                evidence=f"ratio={ratio:.2f}",
            ))
        elif ratio > hi:
            r.flags.append(Flag(
                "warning", "metric", "LENGTH_TOO_LONG_VS_REF",
                f"Summary is {ratio:.1f}× reference length — likely padded.",
                evidence=f"ratio={ratio:.2f}",
            ))

    def _check_rouge(self, r: QAReport) -> None:
        assert self._rouge_scorer is not None
        assert r.reference is not None
        scores = self._rouge_scorer.score(r.reference, r.prediction)
        r.metrics["rouge1"] = round(scores["rouge1"].fmeasure, 4)
        r.metrics["rouge2"] = round(scores["rouge2"].fmeasure, 4)
        r.metrics["rougeL"] = round(scores["rougeL"].fmeasure, 4)
        r.metrics["rougeLsum"] = round(scores["rougeLsum"].fmeasure, 4)
        rL = scores["rougeLsum"].fmeasure
        if rL < self.rouge_crit:
            r.flags.append(Flag(
                "critical", "metric", "LOW_ROUGE",
                f"ROUGE-Lsum={rL:.3f} — extremely low overlap with reference.",
                evidence=f"rougeLsum={rL:.3f}",
            ))
        elif rL < self.rouge_warn:
            r.flags.append(Flag(
                "warning", "metric", "LOW_ROUGE",
                f"ROUGE-Lsum={rL:.3f} — low overlap with reference.",
                evidence=f"rougeLsum={rL:.3f}",
            ))

    def _check_bertscore(self, r: QAReport) -> None:
        try:
            from bert_score import score as bertscore_fn
        except ImportError:
            return
        assert r.reference is not None
        P, R, F1 = bertscore_fn(
            [r.prediction], [r.reference], lang="en",
            rescale_with_baseline=True, verbose=False,
        )
        f1 = float(F1[0])
        r.metrics["bertscore_f1"] = round(f1, 4)
        if f1 < self.bertscore_crit:
            r.flags.append(Flag(
                "critical", "metric", "LOW_BERTSCORE",
                f"BERTScore F1={f1:.3f} — below random baseline.",
                evidence=f"bertscore={f1:.3f}",
            ))
        elif f1 < self.bertscore_warn:
            r.flags.append(Flag(
                "warning", "metric", "LOW_BERTSCORE",
                f"BERTScore F1={f1:.3f} — weak semantic similarity.",
                evidence=f"bertscore={f1:.3f}",
            ))

    # ── Judge score ingestion (no new API call; uses fields from eval JSONL) ──

    def ingest_judge_scores(self, r: QAReport, record: dict) -> None:
        """If the eval record already has judge_* fields, fold them in."""
        DIMS = ["factual_accuracy", "completeness", "conciseness_style",
                "legal_reasoning", "overall"]
        any_judge = False
        for d in DIMS:
            key = f"judge_{d}"
            if key in record and record[key] is not None:
                r.metrics[key] = record[key]
                any_judge = True
        if not any_judge:
            return
        overall = r.metrics.get("judge_overall")
        if overall is not None:
            if overall <= 1:
                r.flags.append(Flag(
                    "critical", "judge", "JUDGE_REJECT",
                    f"LLM judge gave overall score {overall}/5.",
                    evidence=f"judge_overall={overall}",
                ))
            elif overall == 2:
                r.flags.append(Flag(
                    "warning", "judge", "JUDGE_WEAK",
                    f"LLM judge gave overall score {overall}/5.",
                    evidence=f"judge_overall={overall}",
                ))


# ──────────────────────────────────────────────────────────────────────────────
# Batch runner & output formats
# ──────────────────────────────────────────────────────────────────────────────


def run_batch(
    records: Iterable[dict],
    checker: SummaryQAChecker,
    prediction_field: str = "prediction",
    reference_field: str = "reference",
    id_field: str = "case_id",
) -> list[QAReport]:
    reports: list[QAReport] = []
    for rec in records:
        pred = rec.get(prediction_field, "")
        ref = rec.get(reference_field)
        ident = str(rec.get(id_field) or rec.get("index") or len(reports))
        r = checker.check(prediction=pred, reference=ref, identifier=ident)
        checker.ingest_judge_scores(r, rec)
        reports.append(r)
    return reports


def write_jsonl(reports: list[QAReport], path: Path) -> None:
    with open(path, "w") as f:
        for r in reports:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def write_csv(reports: list[QAReport], path: Path) -> None:
    fieldnames = [
        "identifier", "status", "word_count",
        "critical_count", "warning_count", "info_count",
        "top_flags", "rouge1", "rougeLsum", "length_ratio",
        "judge_overall", "prediction_preview",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in reports:
            w.writerow({
                "identifier": r.identifier,
                "status": r.status,
                "word_count": r.word_count,
                "critical_count": r.critical_count,
                "warning_count": r.warning_count,
                "info_count": r.info_count,
                "top_flags": " | ".join(r.top_flag_codes(5)),
                "rouge1": r.metrics.get("rouge1", ""),
                "rougeLsum": r.metrics.get("rougeLsum", ""),
                "length_ratio": r.metrics.get("length_ratio", ""),
                "judge_overall": r.metrics.get("judge_overall", ""),
                "prediction_preview": (r.prediction or "")[:120].replace("\n", " "),
            })


# ── HTML dashboard ────────────────────────────────────────────────────────────


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Summary QA Report — {title}</title>
<style>
  :root {{
    --green: #16a34a; --yellow: #d97706; --red: #dc2626; --gray: #64748b;
    --bg: #f8fafc; --card: #ffffff; --border: #e2e8f0; --text: #0f172a;
    --text-muted: #475569; --accent: #2563eb;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    margin: 0; padding: 0; background: var(--bg); color: var(--text);
    line-height: 1.5;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 32px; }}
  header {{ margin-bottom: 32px; }}
  header h1 {{ margin: 0 0 8px; font-size: 28px; }}
  header .meta {{ color: var(--text-muted); font-size: 14px; }}

  .stat-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
    margin-bottom: 32px;
  }}
  .stat-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 20px; display: flex; flex-direction: column; gap: 4px;
  }}
  .stat-card .value {{ font-size: 36px; font-weight: 700; }}
  .stat-card .label {{ font-size: 13px; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--text-muted); }}
  .stat-card.pass .value {{ color: var(--green); }}
  .stat-card.review .value {{ color: var(--yellow); }}
  .stat-card.reject .value {{ color: var(--red); }}
  .stat-card.total .value {{ color: var(--accent); }}

  .section {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 24px; margin-bottom: 24px;
  }}
  .section h2 {{
    margin: 0 0 16px; font-size: 18px;
    border-bottom: 1px solid var(--border); padding-bottom: 10px;
  }}

  .flag-freq {{ display: grid; grid-template-columns: 1fr auto 200px; gap: 12px;
    align-items: center; }}
  .flag-freq .code {{ font-family: ui-monospace, SFMono-Regular, Menlo,
    monospace; font-size: 13px; }}
  .flag-freq .count {{ font-weight: 600; text-align: right; }}
  .flag-freq .bar {{ background: var(--border); border-radius: 4px;
    height: 8px; position: relative; overflow: hidden; }}
  .flag-freq .bar-fill {{ position: absolute; left: 0; top: 0; bottom: 0;
    background: var(--accent); border-radius: 4px; }}
  .sev-critical {{ color: var(--red); }}
  .sev-warning {{ color: var(--yellow); }}
  .sev-info {{ color: var(--gray); }}

  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: #f1f5f9; font-weight: 600; font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted); }}
  tr:hover {{ background: #f8fafc; }}

  .status-badge {{
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .status-pass {{ background: #dcfce7; color: #166534; }}
  .status-review {{ background: #fef3c7; color: #92400e; }}
  .status-reject {{ background: #fee2e2; color: #991b1b; }}

  details {{ margin: 0; }}
  details summary {{ cursor: pointer; padding: 6px 0; font-weight: 500;
    color: var(--accent); }}
  details[open] summary {{ margin-bottom: 12px; }}

  .drill {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
    background: #f8fafc; padding: 20px; border-radius: 8px;
    margin-top: 8px; margin-bottom: 16px; }}
  .drill h4 {{ margin: 0 0 8px; font-size: 13px; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--text-muted); }}
  .drill .text {{ background: white; padding: 12px; border-radius: 6px;
    border: 1px solid var(--border); max-height: 300px; overflow-y: auto;
    white-space: pre-wrap; font-size: 13px; line-height: 1.6; }}

  .flag-list {{ margin: 12px 0 0; padding: 0; list-style: none;
    grid-column: 1 / -1; }}
  .flag-list li {{ padding: 8px 12px; margin-bottom: 6px; border-radius: 6px;
    border-left: 3px solid var(--gray); background: white; font-size: 13px; }}
  .flag-list li.critical {{ border-left-color: var(--red); background: #fef2f2; }}
  .flag-list li.warning {{ border-left-color: var(--yellow); background: #fffbeb; }}
  .flag-list li.info {{ border-left-color: var(--gray); background: #f9fafb; }}
  .flag-list .code-label {{ font-family: ui-monospace, monospace;
    font-size: 11px; font-weight: 700; margin-right: 8px; padding: 2px 6px;
    border-radius: 3px; background: rgba(0,0,0,0.06); }}
  .flag-list .evidence {{ font-family: ui-monospace, monospace;
    font-size: 12px; color: var(--text-muted); display: block; margin-top: 4px; }}

  .metrics-inline {{ font-family: ui-monospace, monospace; font-size: 12px;
    color: var(--text-muted); }}
  .filter-bar {{ margin-bottom: 16px; }}
  .filter-bar input {{
    padding: 8px 12px; font-size: 14px; border: 1px solid var(--border);
    border-radius: 6px; width: 300px;
  }}
</style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Summary Quality Assurance Report</h1>
      <div class="meta">{subtitle}</div>
    </header>

    <div class="stat-grid">
      <div class="stat-card total"><div class="value">{total}</div>
        <div class="label">Total Summaries</div></div>
      <div class="stat-card pass"><div class="value">{n_pass}</div>
        <div class="label">✓ Pass ({pct_pass}%)</div></div>
      <div class="stat-card review"><div class="value">{n_review}</div>
        <div class="label">⚠ Needs Review ({pct_review}%)</div></div>
      <div class="stat-card reject"><div class="value">{n_reject}</div>
        <div class="label">✗ Reject ({pct_reject}%)</div></div>
    </div>

    <div class="section">
      <h2>Most Common Issues</h2>
      {flag_freq_html}
    </div>

    <div class="section">
      <h2>All Summaries ({total})</h2>
      <div class="filter-bar">
        <input id="filter-input" type="text"
               placeholder="Filter by ID, status, or flag code…"
               oninput="filterRows(this.value)">
      </div>
      <table id="records-table">
        <thead>
          <tr>
            <th>Status</th>
            <th>ID</th>
            <th>Words</th>
            <th>Flags</th>
            <th>Metrics</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>

    <footer style="text-align:center; color:var(--text-muted); font-size:12px;
                   padding: 20px 0;">
      Generated {generated_at} · CivilRightsSummarizedAI · summary_qa.py
    </footer>
  </div>

<script>
  function filterRows(query) {{
    query = query.toLowerCase().trim();
    const rows = document.querySelectorAll('#records-table tbody tr');
    rows.forEach(row => {{
      const text = row.getAttribute('data-search') || '';
      row.style.display = (!query || text.includes(query)) ? '' : 'none';
    }});
  }}
</script>
</body>
</html>
"""


def _escape(s: str) -> str:
    return html_lib.escape(s or "", quote=True)


def _render_row(r: QAReport, row_idx: int) -> str:
    badge_class = f"status-{r.status.lower()}"
    metrics_bits = []
    for k in ["rouge1", "rougeLsum", "length_ratio", "judge_overall", "bertscore_f1"]:
        if k in r.metrics:
            metrics_bits.append(f"{k}={r.metrics[k]}")
    metrics_str = " · ".join(metrics_bits) if metrics_bits else "—"

    flag_items = []
    for f in sorted(r.flags, key=lambda x: ["critical", "warning", "info"].index(x.severity)):
        ev_html = f'<span class="evidence">{_escape(f.evidence)}</span>' if f.evidence else ""
        flag_items.append(
            f'<li class="{f.severity}"><span class="code-label">{f.code}</span>'
            f'{_escape(f.message)}{ev_html}</li>'
        )
    flags_list_html = (
        '<ul class="flag-list">' + "".join(flag_items) + "</ul>"
        if flag_items else '<div style="grid-column:1/-1;color:var(--green);">✓ No flags</div>'
    )

    ref_html = f'<div class="text">{_escape(r.reference)}</div>' if r.reference \
               else '<div class="text" style="color:var(--text-muted);">(no reference)</div>'

    drill_html = f"""
      <div class="drill">
        <div>
          <h4>Reference (human)</h4>
          {ref_html}
        </div>
        <div>
          <h4>Generated summary</h4>
          <div class="text">{_escape(r.prediction)}</div>
        </div>
        {flags_list_html}
      </div>
    """

    top_codes = r.top_flag_codes(3)
    flags_cell = (
        " ".join(f'<span class="sev-{_sev_for_code(r, c)}">{_escape(c)}</span>'
                 for c in top_codes)
        if top_codes else '<span class="sev-info">—</span>'
    )

    search_text = (
        f"{r.identifier} {r.status} {' '.join(f.code for f in r.flags)}"
    ).lower()

    return f"""
        <tr data-search="{_escape(search_text)}">
          <td><span class="status-badge {badge_class}">{r.status}</span></td>
          <td><code>{_escape(r.identifier)}</code></td>
          <td>{r.word_count}</td>
          <td style="max-width:300px;">{flags_cell}</td>
          <td class="metrics-inline">{_escape(metrics_str)}</td>
          <td><details><summary>Expand</summary></details></td>
        </tr>
        <tr data-search="{_escape(search_text)}">
          <td colspan="6" style="padding:0;">{drill_html}</td>
        </tr>
    """


def _sev_for_code(r: QAReport, code: str) -> str:
    for f in r.flags:
        if f.code == code:
            return f.severity
    return "info"


def _flag_frequency_html(reports: list[QAReport]) -> str:
    counter = Counter()
    sev_map = {}
    for r in reports:
        for f in r.flags:
            counter[f.code] += 1
            sev_map[f.code] = f.severity
    if not counter:
        return '<div style="color:var(--green);">No flags raised across any summary.</div>'
    max_count = max(counter.values())
    rows = []
    for code, count in counter.most_common(20):
        sev = sev_map.get(code, "info")
        bar_width = int(100 * count / max_count)
        rows.append(
            f'<div class="code sev-{sev}">{code}</div>'
            f'<div class="count">{count}</div>'
            f'<div class="bar"><div class="bar-fill sev-{sev}" '
            f'style="width:{bar_width}%;background:var(--{"red" if sev=="critical" else "yellow" if sev=="warning" else "gray"});"></div></div>'
        )
    return f'<div class="flag-freq">{"".join(rows)}</div>'


def write_html(reports: list[QAReport], path: Path, title: str, subtitle: str) -> None:
    n = len(reports)
    n_pass = sum(1 for r in reports if r.status == "PASS")
    n_review = sum(1 for r in reports if r.status == "REVIEW")
    n_reject = sum(1 for r in reports if r.status == "REJECT")

    def pct(x):
        return f"{round(100 * x / n)}" if n else "0"

    rows_html = "\n".join(_render_row(r, i) for i, r in enumerate(reports))
    flag_freq_html = _flag_frequency_html(reports)

    html_out = HTML_TEMPLATE.format(
        title=_escape(title),
        subtitle=_escape(subtitle),
        total=n,
        n_pass=n_pass, n_review=n_review, n_reject=n_reject,
        pct_pass=pct(n_pass), pct_review=pct(n_review), pct_reject=pct(n_reject),
        flag_freq_html=flag_freq_html,
        rows_html=rows_html,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(path, "w") as f:
        f.write(html_out)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="JSONL file or .txt file")
    ap.add_argument("--output-dir", required=True, help="Directory for reports")
    ap.add_argument("--prediction-field", default="prediction")
    ap.add_argument("--reference-field", default="reference")
    ap.add_argument("--id-field", default="case_id")
    ap.add_argument("--title", default="Case Summary QA")
    ap.add_argument("--no-rouge", action="store_true",
                    help="Skip ROUGE (faster, no rouge_score dependency)")
    ap.add_argument("--bertscore", action="store_true",
                    help="Compute BERTScore (slow, needs bert_score package)")
    ap.add_argument("--min-words", type=int, default=80)
    ap.add_argument("--max-words", type=int, default=2000)
    return ap.parse_args()


def load_records(path: Path, pred_field: str) -> list[dict]:
    if path.suffix == ".jsonl":
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else [data]
    else:
        # Plain text file — single prediction
        return [{pred_field: path.read_text(), "case_id": path.stem}]


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        sys.exit(f"ERROR: input file not found: {input_path}")

    print(f"Loading {input_path}…")
    records = load_records(input_path, args.prediction_field)
    print(f"Loaded {len(records)} records")

    checker = SummaryQAChecker(
        min_words=args.min_words,
        max_words=args.max_words,
        enable_rouge=not args.no_rouge,
        enable_bertscore=args.bertscore,
    )

    print("Running checks…")
    reports = run_batch(
        records, checker,
        prediction_field=args.prediction_field,
        reference_field=args.reference_field,
        id_field=args.id_field,
    )

    # Summary stats
    n_pass = sum(1 for r in reports if r.status == "PASS")
    n_review = sum(1 for r in reports if r.status == "REVIEW")
    n_reject = sum(1 for r in reports if r.status == "REJECT")
    print()
    print(f"  PASS:   {n_pass:4d}")
    print(f"  REVIEW: {n_review:4d}")
    print(f"  REJECT: {n_reject:4d}")
    print()

    # Write outputs
    jsonl_path = out_dir / "qa_report.jsonl"
    csv_path = out_dir / "qa_report.csv"
    html_path = out_dir / "qa_report.html"
    write_jsonl(reports, jsonl_path)
    write_csv(reports, csv_path)
    subtitle = f"Source: {input_path.name} · {len(reports)} records"
    write_html(reports, html_path, title=args.title, subtitle=subtitle)

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {html_path}")
    print()
    print(f"Open in browser:  file://{html_path.resolve()}")


if __name__ == "__main__":
    main()
