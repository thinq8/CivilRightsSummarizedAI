from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from clearinghouse.clients import ClearinghouseClient
from clearinghouse.processing import HeuristicSummarizer
from clearinghouse.storage import (
    CaseRecord,
    DocumentRecord,
    DocketRecord,
    IngestionCheckpointRecord,
    IngestionRunRecord,
    RawApiPayloadRecord,
)
from clearinghouse.types import Case, Document, Docket

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """High-level counters returned to the CLI and tests."""

    run_id: str | None = None
    effective_since: datetime | None = None
    resumed_from_checkpoint: bool = False
    cases: int = 0
    dockets: int = 0
    documents: int = 0
    errors: int = 0


class IngestionPipeline:
    """
    Durable ingestion orchestrator.

    Design goals:
    1. Keep normalized tables (`cases`, `dockets`, `documents`) current.
    2. Keep a raw payload archive for future schema evolution and ML data lineage.
    3. Support resumable runs via checkpoints so interrupted jobs do not start from scratch.
    """

    def __init__(
        self,
        client: ClearinghouseClient,
        session_factory: sessionmaker,
        summarizer: HeuristicSummarizer | None = None,
        *,
        source: str = "unknown",
        checkpoint_key: str | None = None,
        archive_raw_payloads: bool = True,
        continue_on_error: bool = True,
    ) -> None:
        self.client = client
        self.session_factory = session_factory
        self.summarizer = summarizer or HeuristicSummarizer()
        self.source = source
        self.checkpoint_key = checkpoint_key
        self.archive_raw_payloads = archive_raw_payloads
        self.continue_on_error = continue_on_error

    def run(
        self,
        since: datetime | None = None,
        case_limit: int | None = None,
        *,
        resume_from_checkpoint: bool = False,
    ) -> IngestionStats:
        """
        Execute ingestion with optional incremental resume.

        When `resume_from_checkpoint` is enabled and `checkpoint_key` is configured, the pipeline
        uses the latest committed checkpoint timestamp as the effective "since" filter.
        """

        stats = IngestionStats()
        run_id = str(uuid.uuid4())
        stats.run_id = run_id
        requested_since = _normalize_datetime(since)
        effective_since = requested_since
        resumed = False

        if resume_from_checkpoint and self.checkpoint_key:
            checkpoint_since = self._get_checkpoint_since(self.checkpoint_key)
            if checkpoint_since and (effective_since is None or checkpoint_since > effective_since):
                effective_since = checkpoint_since
                resumed = True

        stats.effective_since = effective_since
        stats.resumed_from_checkpoint = resumed
        self._start_run_record(
            run_id=run_id,
            requested_since=requested_since,
            effective_since=effective_since,
            case_limit=case_limit,
            resumed_from_checkpoint=resumed,
        )

        logger.info(
            "Starting ingestion",
            extra={
                "run_id": run_id,
                "source": self.source,
                "requested_since": requested_since.isoformat() if requested_since else None,
                "effective_since": effective_since.isoformat() if effective_since else None,
                "resume_from_checkpoint": resume_from_checkpoint,
                "checkpoint_key": self.checkpoint_key,
            },
        )

        final_status = "success"
        fatal_error: Exception | None = None
        fatal_error_message: str | None = None

        try:
            for idx, case in enumerate(self.client.list_cases(effective_since)):
                if case_limit is not None and idx >= case_limit:
                    logger.info("Reached case limit", extra={"limit": case_limit, "run_id": run_id})
                    break

                try:
                    dockets_count, documents_count = self._ingest_case(run_id, case)
                    stats.cases += 1
                    stats.dockets += dockets_count
                    stats.documents += documents_count
                    if self.checkpoint_key:
                        self._update_checkpoint(self.checkpoint_key, case, run_id)
                except Exception as exc:  # pragma: no cover - exercised in integration environments
                    stats.errors += 1
                    logger.exception(
                        "Failed to ingest case",
                        extra={"run_id": run_id, "case_id": case.id},
                    )
                    if not self.continue_on_error:
                        raise
                    if final_status == "success":
                        final_status = "partial"
                    fatal_error_message = str(exc)
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            final_status = "failed"
            fatal_error = exc
            fatal_error_message = str(exc)
        finally:
            self._finish_run_record(
                run_id=run_id,
                status=final_status,
                stats=stats,
                error_message=fatal_error_message,
            )
            logger.info("Ingestion finished", extra={"run_id": run_id, **stats.__dict__, "status": final_status})

        if fatal_error is not None:
            raise fatal_error
        return stats

    def _ingest_case(self, run_id: str, case: Case) -> tuple[int, int]:
        """
        Ingest one case transactionally.

        We commit a case, its dockets, and its documents together to avoid partially written
        entities for a single case. This keeps downstream training exports consistent.
        """

        dockets_count = 0
        documents_count = 0
        with self.session_factory() as session:
            self._upsert_case(session, case)
            if self.archive_raw_payloads:
                self._archive_raw_payload(
                    session,
                    run_id=run_id,
                    resource_type="case",
                    resource_id=case.id,
                    payload=case.metadata,
                    case_id=case.id,
                )

            for docket in self.client.list_dockets(case.id):
                self._upsert_docket(session, docket)
                dockets_count += 1
                if self.archive_raw_payloads:
                    self._archive_raw_payload(
                        session,
                        run_id=run_id,
                        resource_type="docket",
                        resource_id=docket.id,
                        payload=docket.metadata,
                        case_id=docket.case_id,
                        docket_id=docket.id,
                    )

            for document in self.client.list_documents(case.id):
                summary = self.summarizer.summarize(document)
                self._upsert_document(session, document, summary)
                documents_count += 1
                if self.archive_raw_payloads:
                    self._archive_raw_payload(
                        session,
                        run_id=run_id,
                        resource_type="document",
                        resource_id=document.id,
                        payload=document.metadata,
                        case_id=document.case_id,
                        docket_id=document.docket_id,
                    )

            session.commit()
        return dockets_count, documents_count

    @staticmethod
    def _upsert_case(session: Session, case: Case) -> None:
        session.merge(
            CaseRecord(
                id=case.id,
                name=case.name,
                court=case.court,
                state=case.state,
                jurisdiction=case.metadata.get("jurisdiction") if case.metadata else None,
                status=case.status,
                updated_at_remote=case.last_checked,
                documents_url=case.documents_url,
                dockets_url=case.dockets_url,
                metadata_json=dict(case.metadata),
            )
        )

    @staticmethod
    def _upsert_docket(session: Session, docket: Docket) -> None:
        session.merge(
            DocketRecord(
                id=docket.id,
                case_id=docket.case_id,
                docket_number=docket.docket_number,
                court=docket.court,
                state=docket.state,
                is_main=docket.is_main,
                updated_at_remote=None,
                metadata_json=dict(docket.metadata),
            )
        )

    @staticmethod
    def _upsert_document(session: Session, document: Document, summary: str | None) -> None:
        session.merge(
            DocumentRecord(
                id=document.id,
                docket_id=document.docket_id,
                case_id=document.case_id,
                title=document.title,
                document_type=document.document_type,
                filed_date=document.date,
                court=document.court,
                external_url=document.external_url,
                text_url=document.text_url,
                has_text=document.has_text,
                text=document.text,
                summary=summary,
                metadata_json=dict(document.metadata),
            )
        )

    def _start_run_record(
        self,
        *,
        run_id: str,
        requested_since: datetime | None,
        effective_since: datetime | None,
        case_limit: int | None,
        resumed_from_checkpoint: bool,
    ) -> None:
        with self.session_factory() as session:
            session.add(
                IngestionRunRecord(
                    id=run_id,
                    source=self.source,
                    status="running",
                    requested_since=requested_since,
                    effective_since=effective_since,
                    case_limit=case_limit,
                    checkpoint_key=self.checkpoint_key,
                    resumed_from_checkpoint=resumed_from_checkpoint,
                )
            )
            session.commit()

    def _finish_run_record(
        self,
        *,
        run_id: str,
        status: str,
        stats: IngestionStats,
        error_message: str | None,
    ) -> None:
        with self.session_factory() as session:
            run = session.get(IngestionRunRecord, run_id)
            if run is None:
                return
            run.status = status
            run.finished_at = datetime.now(timezone.utc)
            run.cases_ingested = stats.cases
            run.dockets_ingested = stats.dockets
            run.documents_ingested = stats.documents
            run.errors = stats.errors
            run.error_message = error_message
            session.commit()

    def _get_checkpoint_since(self, checkpoint_key: str) -> datetime | None:
        with self.session_factory() as session:
            checkpoint = session.get(IngestionCheckpointRecord, checkpoint_key)
            if checkpoint is None:
                return None
            return _normalize_datetime(checkpoint.last_case_last_checked)

    def _update_checkpoint(self, checkpoint_key: str, case: Case, run_id: str) -> None:
        """
        Advance checkpoint only after a case is fully committed.

        This guarantees resume pointers never move ahead of durable writes.
        """

        with self.session_factory() as session:
            checkpoint = session.get(IngestionCheckpointRecord, checkpoint_key)
            if checkpoint is None:
                checkpoint = IngestionCheckpointRecord(
                    key=checkpoint_key,
                    source=self.source,
                )

            case_ts = _normalize_datetime(case.last_checked)
            if case_ts is not None and (
                checkpoint.last_case_last_checked is None
                or case_ts >= _normalize_datetime(checkpoint.last_case_last_checked)
            ):
                checkpoint.last_case_last_checked = case_ts
                checkpoint.last_case_id = case.id
            elif checkpoint.last_case_last_checked is None:
                checkpoint.last_case_id = case.id

            checkpoint.last_run_id = run_id
            session.merge(checkpoint)
            session.commit()

    def _archive_raw_payload(
        self,
        session: Session,
        *,
        run_id: str,
        resource_type: str,
        resource_id: str,
        payload: Mapping[str, Any],
        case_id: str | None,
        docket_id: str | None = None,
    ) -> None:
        """
        Store unique raw payload versions by SHA256.

        Keeping unique versions avoids exploding storage while preserving a change history that is
        useful for future model training and for debugging upstream API schema changes.
        """

        safe_payload = _to_json_safe(dict(payload))
        serialized = json.dumps(safe_payload, sort_keys=True, separators=(",", ":"))
        payload_sha = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        existing = session.scalar(
            select(RawApiPayloadRecord.id)
            .where(RawApiPayloadRecord.resource_type == resource_type)
            .where(RawApiPayloadRecord.resource_id == str(resource_id))
            .where(RawApiPayloadRecord.payload_sha256 == payload_sha)
            .limit(1)
        )
        if existing:
            return

        session.add(
            RawApiPayloadRecord(
                ingestion_run_id=run_id,
                source=self.source,
                resource_type=resource_type,
                resource_id=str(resource_id),
                case_id=str(case_id) if case_id else None,
                docket_id=str(docket_id) if docket_id else None,
                payload_sha256=payload_sha,
                payload_json=safe_payload,
            )
        )


def _to_json_safe(value: Any) -> Any:
    """Recursively coerce payload values into JSON-safe primitives."""

    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalize_datetime(value: datetime | None) -> datetime | None:
    """
    Normalize datetimes to timezone-aware UTC values.

    SQLite may round-trip timezone columns as naive datetimes. Normalizing here prevents
    type-errors when comparing checkpoint timestamps to API timestamps.
    """

    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
