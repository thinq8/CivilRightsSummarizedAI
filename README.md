# CivilRightsSummarizedAI

CivilRightsSummarizedAI is the working space for our CMSE capstone project focused on using modern LLM and NLP techniques to summarize lengthy U.S. civil-rights legal documents. This repository will evolve throughout the semester into a full research and engineering log for the project.

## Course Context
- **Institution / Course**: Michigan State University – CMSE Capstone alongside/for UofM in collaboration with Civil Rights Clearinghouse for Justice
- **Team Goal**: Deliver a reproducible pipeline that ingests a range of civil-rights cases and produces concise, accurate, and accessible summaries for stakeholders.

## Repository Guide
- `README.md` – Entry point for the project, describing scope, how to navigate the repo, and planned milestones.
- `LICENSE` – MIT License governing how the code and documentation may be used.
- `.gitignore` – Standard Python-oriented ignore rules to keep temporary and machine-specific files out of the repository.
- `environment.yml` – Minimal Conda environment specification for contributors spinning up a workspace quickly.
- `docs/` – Living documentation hub (see `docs/README.md` for layout and authoring tips).
- `Examples/` – Lightweight datasets and tutorials for demonstrating the workflow.
- `package_name/` – Python package placeholder that will evolve into the production code.
- `environments/` – Additional environment definitions for specialized hardware or deployment contexts.

## Repository Structure
```
.
├── docs/
│   ├── README.md
│   ├── images/
│   │   └── README.md
│   └── package_name/
│       └── README.md
├── Examples/
│   └── README.md
├── package_name/
│   ├── README.md
│   ├── __init__.py
│   ├── module1.py
│   ├── module2.py
│   └── test/
│       ├── __init__.py
│       ├── test_module1.py
│       └── test_module2.py
├── environments/
│   └── README.md
├── environment.yml
├── LICENSE
├── README.md
└── S2026-UofM_Civil_Rights_Summaries.pdf
```

## Planned Additions
As the semester progresses we still plan to incorporate:
- `docs/package_name/*.html` outputs for API references generated with a documentation toolchain.
- `docs/images/*` diagrams sourced from design sessions and presentations.
- `Examples/data/` sample CSV, TIFF, and XLS artifacts mirroring stakeholder inputs.
- `makefile` or CLI helpers for linting, testing, and deployment automation.
- `setup.py`/`pyproject.toml` once packaging and distribution requirements are finalized.

## Getting Started
1. Clone the repository (`git clone (https://github.com/thinq8/CivilRightsSummarizedAI.git)`) and move into the project directory.
2. Create and activate a Python environment (Conda, venv, or Poetry) that matches the forthcoming `environment.yml`.
3. Install project dependencies using `conda env create -f environment.yml` (or the richer specs inside `environments/`) and follow instructions in the `docs/` folder for running experiments or the application.

## License
This project is released under the MIT License (see `LICENSE`).
