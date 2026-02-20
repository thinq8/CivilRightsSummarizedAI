from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from clearinghouse.clients.base import ClearinghouseClient
from clearinghouse.types import Case, Docket, Document


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    value = value.replace("Z", "+00:00")
    return datetime.fromisoformat(value)


class MockClearinghouseClient(ClearinghouseClient):
    """Mock client backed by a static JSON fixture."""

    def __init__(self, fixture_path: Path):
        with fixture_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._cases: list[Case] = []
        self._dockets_by_case: dict[str, list[Docket]] = {}
        self._documents_by_case: dict[str, list[Document]] = {}

        for case_payload in payload.get("cases", []):
            case_id = str(case_payload["id"])
            case = Case(
                id=case_id,
                name=case_payload["name"],
                court=case_payload.get("court"),
                state=case_payload.get("state") or case_payload.get("jurisdiction"),
                status=case_payload.get("status"),
                last_checked=_parse_datetime(case_payload.get("updated_at")),
                documents_url=case_payload.get("documents_url")
                or f"mock://cases/{case_id}/documents",
                dockets_url=case_payload.get("dockets_url") or f"mock://cases/{case_id}/dockets",
                metadata=case_payload.get("metadata", {}),
            )
            self._cases.append(case)

            dockets: list[Docket] = []
            documents: list[Document] = []
            for docket_payload in case_payload.get("dockets", []):
                docket_id = str(docket_payload["id"])
                docket = Docket(
                    id=docket_id,
                    case_id=case_id,
                    docket_number=docket_payload.get("number"),
                    court=docket_payload.get("court", case.court),
                    state=docket_payload.get("state") or case.state,
                    is_main=docket_payload.get("is_main", True),
                    metadata=docket_payload.get("metadata", {}),
                )
                dockets.append(docket)

                for doc_payload in docket_payload.get("documents", []):
                    document = Document(
                        id=str(doc_payload["id"]),
                        case_id=case_id,
                        docket_id=docket_id,
                        title=doc_payload["title"],
                        document_type=doc_payload.get("document_type"),
                        date=_parse_datetime(doc_payload.get("filed_date")),
                        court=doc_payload.get("court", docket.court),
                        has_text=bool(doc_payload.get("text")),
                        text_url=doc_payload.get("text_url"),
                        external_url=doc_payload.get("source_url"),
                        text=doc_payload.get("text"),
                        metadata=doc_payload.get("metadata", {}),
                    )
                    documents.append(document)

            self._dockets_by_case[case_id] = dockets
            self._documents_by_case[case_id] = documents

        self._cases.sort(key=lambda c: (c.last_checked or datetime.min))

    def list_cases(self, updated_after: datetime | None = None) -> Iterable[Case]:
        for case in self._cases:
            if updated_after and case.last_checked and case.last_checked <= updated_after:
                continue
            yield case

    def list_dockets(self, case_id: str) -> Iterable[Docket]:
        yield from self._dockets_by_case.get(str(case_id), [])

    def list_documents(self, case_id: str) -> Iterable[Document]:
        yield from self._documents_by_case.get(str(case_id), [])

    def get_document(self, case_id: str, document_id: str) -> Document | None:
        for document in self._documents_by_case.get(str(case_id), []):
            if document.id == str(document_id):
                return document
        return None
