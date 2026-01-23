# Environment Definitions

Store Conda, mamba, or virtualenv specification files in this directory to keep development and deployment setups reproducible.

Recommended files:
- `environment.yml` – Primary Conda environment for development and experimentation.
- `environment-gpu.yml` – Optional GPU-accelerated environment if the project requires CUDA.
- `requirements.txt` – Lightweight pip requirements for CI or deployment targets.

Whenever dependencies change, update the relevant file and document the change in the project changelog or README.
