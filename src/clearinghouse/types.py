from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(slots=True)
class Case:
    id: str
    name: str
    court: str | None
    state: str | None
    status: str | None
    last_checked: datetime | None
    documents_url: str | None
    dockets_url: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Docket:
    id: str
    case_id: str
    docket_number: str | None
    court: str | None
    state: str | None
    is_main: bool | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    id: str
    case_id: str
    docket_id: str | None
    title: str
    document_type: str | None
    date: datetime | None
    court: str | None
    has_text: bool
    text_url: str | None
    external_url: str | None
    text: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
