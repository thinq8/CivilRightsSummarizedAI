# INSTALL.md

This guide is for instructors, classmates, and community partners to install, test, and demo the Civil Rights Summarized AI project with example data.

## 1) Software Requirements

- Git
- Conda distribution (recommended: [Miniforge](https://conda-forge.org/download/))
- Terminal (macOS/Linux shell or Windows Anaconda/Miniforge Prompt)

## 2) Clone the Repository

```bash
git clone https://github.com/thinq8/CivilRightsSummarizedAI.git
cd CivilRightsSummarizedAI
```

## 3) Create and Activate the Conda Environment

From the repository root:

```bash
conda env create --prefix ./envs --file environment.yml
conda activate ./envs
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Notes:
- `environment.yml` is intentionally minimal.
- Project/runtime dependencies are installed from `pyproject.toml` using `pip install -e ".[dev]"`.

## 4) Data Instructions

### Included example data (default for grading/demo)
- Use `data/fixtures/mock_dataset.json`.
- This mock dataset is safe to share and is the default for install testing and demo runs.

### Community partner/private data
- Private partner data is **not** required for the reproducible demo.
- Do not commit private data to this repository.
- If private files are used locally, store them under ignored paths such as `data/raw_documents/` or `data/tmp/`.

## 5) Test the Installation

Run from repository root (with env activated):

```bash
pytest -q
```

Expected result:
- `4 passed` (or higher if new tests are added later).

## 6) Run the Demo (Out-of-the-Box)

Run the end-to-end mock ingestion demo:

```bash
python -m clearinghouse.cli ingest-mock \
  --db-url sqlite:///data/dev.db \
  --fixture data/fixtures/mock_dataset.json \
  --checkpoint-key mock-default \
  --resume-from-checkpoint \
  --archive-raw-payloads
```

Expected output pattern:
- `Ingestion complete: run_id=... cases=... dockets=... documents=... errors=...`

Optional quick check:

```bash
python -m clearinghouse.cli ingest-mock --help
```

## 7) Optional Live API Demo (Token Required)

If you have approved Clearinghouse API credentials:

```bash
export CLEARINGHOUSE_API_TOKEN="YOUR_TOKEN_HERE"
python -m clearinghouse.cli ingest-live --db-url sqlite:///data/live.db --case-limit 25
```

This step is optional and not required for course install verification.

## 8) Remove Environment (Optional Cleanup)

```bash
conda deactivate
rm -rf ./envs
```

## 9) Windows UTF-8 Fix (If Needed)

If `environment.yml` encoding issues occur on Windows:

```powershell
conda env export --from-history | Set-Content -Encoding utf8 environment.yml
```
