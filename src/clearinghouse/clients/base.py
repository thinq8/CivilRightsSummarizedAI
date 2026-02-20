from __future__ import annotations

from datetime import datetime
from typing import Iterable, Protocol

from clearinghouse.types import Case, Docket, Document


class ClearinghouseClient(Protocol):
    """Interface for retrieving Clearinghouse cases, dockets, and documents."""

    def list_cases(self, updated_after: datetime | None = None) -> Iterable[Case]:
        """Return cases updated after the timestamp (if provided)."""

    def list_dockets(self, case_id: str) -> Iterable[Docket]:
        """Return dockets for the given case."""

    def list_documents(self, case_id: str) -> Iterable[Document]:
        """Return documents for the given case."""

    def get_document(self, case_id: str, document_id: str) -> Document | None:
        """Return a single document when available."""
