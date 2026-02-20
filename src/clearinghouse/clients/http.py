from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from typing import Any, Iterable

import httpx

from clearinghouse.clients.base import ClearinghouseClient
from clearinghouse.types import Case, Docket, Document

logger = logging.getLogger(__name__)
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class HttpClearinghouseClient(ClearinghouseClient):
    """
    Client that talks to the public Clearinghouse API (v2.1).

    The client includes conservative retry/backoff behavior because ingestion jobs are often long
    lived and should tolerate transient API/network failures.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 30.0,
        user_agent: str = "CivilRightsSummarizedAI/0.1",
        max_retries: int = 4,
        backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 8.0,
    ) -> None:
        normalized_token = normalize_api_token(token)
        if not normalized_token:
            raise ValueError("A Clearinghouse API token is required")

        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Token {normalized_token}", "User-Agent": user_agent},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpClearinghouseClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def list_cases(self, updated_after: datetime | None = None) -> Iterable[Case]:
        params: dict[str, Any] = {}
        if updated_after:
            params["last_checked_date__gte"] = updated_after.isoformat()
        yield from (
            _case_from_api(item)
            for item in self._paginate("/cases/", params=params if params else None)
        )

    def list_dockets(self, case_id: str) -> Iterable[Docket]:
        path = f"/cases/{case_id}/dockets/"
        yield from (_docket_from_api(item, case_id) for item in self._paginate(path))

    def list_documents(self, case_id: str) -> Iterable[Document]:
        path = f"/cases/{case_id}/documents/"
        yield from (_document_from_api(item, case_id) for item in self._paginate(path))

    def get_document(self, case_id: str, document_id: str) -> Document | None:
        for document in self.list_documents(case_id):
            if document.id == str(document_id):
                return document
        return None

    def _paginate(self, path: str, params: dict[str, Any] | None = None) -> Iterable[dict[str, Any]]:
        url = path
        query = params
        while url:
            response = self._request_with_retry(url, params=query)
            payload = response.json()
            results = payload.get("results", [])
            logger.debug("Fetched %s records from %s", len(results), url)
            for item in results:
                yield item
            next_url = payload.get("next")
            if not next_url:
                break
            url = next_url
            query = None

    def _request_with_retry(
        self,
        url: str,
        *,
        params: dict[str, Any] | None,
    ) -> httpx.Response:
        """
        Execute GET requests with bounded retry/backoff for transient failures.

        Why this exists:
        - API crawling is high-volume and long-running; occasional failures are expected.
        - We retry only known transient conditions to avoid masking persistent errors.
        """

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(url, params=params)
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                    wait_seconds = self._compute_backoff(attempt, response=response)
                    logger.warning(
                        "Retrying API request after retryable status",
                        extra={
                            "url": url,
                            "status_code": response.status_code,
                            "attempt": attempt + 1,
                            "max_retries": self._max_retries,
                            "wait_seconds": wait_seconds,
                        },
                    )
                    time.sleep(wait_seconds)
                    continue
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self._max_retries:
                    raise
                wait_seconds = self._compute_backoff(attempt)
                logger.warning(
                    "Retrying API request after transport failure",
                    extra={
                        "url": url,
                        "attempt": attempt + 1,
                        "max_retries": self._max_retries,
                        "wait_seconds": wait_seconds,
                        "error": str(exc),
                    },
                )
                time.sleep(wait_seconds)

        raise RuntimeError("Unreachable retry state while requesting Clearinghouse API")

    def _compute_backoff(
        self,
        attempt: int,
        *,
        response: httpx.Response | None = None,
    ) -> float:
        # Respect Retry-After when present; otherwise use exponential backoff with jitter.
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after:
            try:
                wait = float(retry_after)
                return max(0.0, min(wait, self._max_backoff_seconds))
            except ValueError:
                pass
        expo = self._backoff_seconds * (2**attempt)
        jitter = random.uniform(0.0, self._backoff_seconds)
        return max(0.0, min(expo + jitter, self._max_backoff_seconds))


def _case_from_api(raw: dict[str, Any]) -> Case:
    return Case(
        id=str(raw.get("id")),
        name=raw.get("name", ""),
        court=raw.get("court"),
        state=raw.get("state"),
        status=raw.get("case_status"),
        last_checked=_parse_datetime(raw.get("last_checked_date")),
        documents_url=raw.get("case_documents_url"),
        dockets_url=raw.get("case_dockets_url"),
        metadata=raw,
    )


def _docket_from_api(raw: dict[str, Any], case_id: str) -> Docket:
    return Docket(
        id=str(raw.get("id")),
        case_id=str(case_id),
        docket_number=raw.get("docket_number_manual"),
        court=raw.get("court"),
        state=raw.get("state"),
        is_main=raw.get("is_main_docket"),
        metadata=raw,
    )


def _document_from_api(raw: dict[str, Any], case_id: str) -> Document:
    return Document(
        id=str(raw.get("id")),
        case_id=str(case_id),
        docket_id=str(raw.get("docket_id")) if raw.get("docket_id") else None,
        title=raw.get("title", "Untitled"),
        document_type=raw.get("document_type"),
        date=_parse_datetime(raw.get("date")),
        court=raw.get("court"),
        has_text=bool(raw.get("has_text")),
        text_url=raw.get("text_url"),
        external_url=raw.get("external_url") or raw.get("clearinghouse_link"),
        text=None,
        metadata=raw,
    )


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.fromisoformat(f"{value}T00:00:00")
        except ValueError:
            logger.debug("Unable to parse datetime value %s", value)
            return None


def normalize_api_token(token: str | None) -> str:
    """
    Normalize user-provided tokens.

    The CLI/docs often show "Token <value>". The HTTP header builder already prepends "Token ",
    so we strip optional prefixes here to avoid sending "Token Token <value>".
    """

    if not token:
        return ""
    stripped = token.strip()
    if not stripped:
        return ""
    if stripped.lower().startswith("token "):
        stripped = stripped.split(None, 1)[1].strip()
    return stripped
