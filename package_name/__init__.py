"""Top-level package for CivilRightsSummarizedAI utilities."""

from importlib import metadata


__all__ = ["__version__"]


def __getattr__(name: str) -> str:
    if name == "__version__":
        try:
            return metadata.version("civilrightssummarizedai")
        except metadata.PackageNotFoundError:  # pragma: no cover - package not installed yet
            return "0.0.0"
    raise AttributeError(name)
