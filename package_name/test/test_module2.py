"""Tests for module2."""

import pytest

from package_name import module2


def test_summarize_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        module2.summarize("example text")
