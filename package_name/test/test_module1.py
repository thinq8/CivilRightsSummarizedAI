"""Tests for module1."""

import pytest

from package_name import module1


def test_load_documents_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        module1.load_documents([])
