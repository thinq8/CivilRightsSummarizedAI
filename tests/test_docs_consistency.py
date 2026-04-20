from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

KEY_DOCS = [
    "README.md",
    "INSTALL.md",
    "TUTORIAL.md",
    "FINAL_SUBMISSION.md",
    "SHARING_PLAN.md",
    "data/fixtures/README.md",
    "scripts/README.md",
    "tools/README.md",
    "notebooks/figure_instructions.ipynb",
]

FINAL_FIGURES = [
    "figure1_training_dynamics.png",
    "figure2_prompt_length_distribution.png",
    "figure3_checkpoint_comparison.png",
    "figure4_qa_triage_3systems.png",
    "figure5_source_attribution.png",
    "figure6_cost_quality.png",
    "figure7_flag_frequency.png",
]

OLD_FIGURES = [
    "figure1_training_loss.png",
    "figure2_fragmentation_distribution.png",
    "figure3_evaluation_metrics.png",
    "figure4_judge_scores.png",
]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_key_docs_reference_final_figures_only() -> None:
    combined = "\n".join(_read(path) for path in KEY_DOCS)

    for filename in OLD_FIGURES:
        assert filename not in combined

    for filename in FINAL_FIGURES:
        assert (ROOT / "figures" / filename).exists()
        assert filename in combined

    assert "all 4 figures" not in combined.lower()
    assert "shared teams" not in combined.lower()


def test_final_report_metrics_fixture_documents_final_plot_inputs() -> None:
    fixture_path = ROOT / "data" / "fixtures" / "final_report_metrics.json"
    assert fixture_path.exists()

    metrics = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected_sections = {
        "provenance",
        "checkpoint_comparison",
        "qa_triage",
        "source_attribution",
        "cost_quality",
        "flag_frequency",
    }
    assert expected_sections.issubset(metrics)

    assert metrics["provenance"]["contains_private_text"] is False
    assert metrics["provenance"]["contains_model_outputs"] is False


def test_top_level_markdown_links_resolve_to_repo_files() -> None:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for doc in ["README.md", "INSTALL.md", "TUTORIAL.md", "FINAL_SUBMISSION.md"]:
        for target in link_pattern.findall(_read(doc)):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target_path = target.split("#", 1)[0]
            if not target_path:
                continue
            assert (ROOT / target_path).exists(), f"{doc} links to missing {target}"
