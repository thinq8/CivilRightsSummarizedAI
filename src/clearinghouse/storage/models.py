from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )


class CaseRecord(TimestampMixin, Base):
    """Normalized case record used for downstream filtering and training-set joins."""

    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    court: Mapped[str | None] = mapped_column(String, nullable=True)
    state: Mapped[str | None] = mapped_column(String, nullable=True)
    jurisdiction: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at_remote: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    documents_url: Mapped[str | None] = mapped_column(String, nullable=True)
    dockets_url: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    dockets: Mapped[list[DocketRecord]] = relationship(
        back_populates="case", cascade="all, delete-orphan", passive_deletes=True
    )


class DocketRecord(TimestampMixin, Base):
    """Normalized docket record linked to a case."""

    __tablename__ = "dockets"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    docket_number: Mapped[str | None] = mapped_column(String, nullable=True)
    court: Mapped[str | None] = mapped_column(String, nullable=True)
    state: Mapped[str | None] = mapped_column(String, nullable=True)
    is_main: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    updated_at_remote: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    case: Mapped[CaseRecord] = relationship(back_populates="dockets")
    documents: Mapped[list[DocumentRecord]] = relationship(
        back_populates="docket", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentRecord(TimestampMixin, Base):
    """Normalized document record that stores text + summary artifacts."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    docket_id: Mapped[str | None] = mapped_column(ForeignKey("dockets.id", ondelete="SET NULL"), nullable=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    document_type: Mapped[str | None] = mapped_column(String, nullable=True)
    filed_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    court: Mapped[str | None] = mapped_column(String, nullable=True)
    external_url: Mapped[str | None] = mapped_column(String, nullable=True)
    text_url: Mapped[str | None] = mapped_column(String, nullable=True)
    has_text: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    docket: Mapped[DocketRecord | None] = relationship(back_populates="documents")
    case: Mapped[CaseRecord] = relationship()


class IngestionRunRecord(Base):
    """
    One row per pipeline execution.

    This table makes operations observable and auditable for a student team: when a run started,
    what parameters were used, and how many records were produced.
    """

    __tablename__ = "ingestion_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, nullable=False)  # e.g., "mock" or "live"
    status: Mapped[str] = mapped_column(String, nullable=False)  # running|success|failed|partial
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    requested_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    effective_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    case_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checkpoint_key: Mapped[str | None] = mapped_column(String, nullable=True)
    resumed_from_checkpoint: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    cases_ingested: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    dockets_ingested: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    documents_ingested: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    errors: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    raw_payloads: Mapped[list[RawApiPayloadRecord]] = relationship(back_populates="ingestion_run")


class IngestionCheckpointRecord(TimestampMixin, Base):
    """
    Resume pointer for incremental ingestion.

    We track the last successfully committed case checkpoint, so interrupted runs can continue
    near where they left off without re-scanning the full API history each time.
    """

    __tablename__ = "ingestion_checkpoints"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    last_case_id: Mapped[str | None] = mapped_column(String, nullable=True)
    last_case_last_checked: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_run_id: Mapped[str | None] = mapped_column(String, nullable=True)


class RawApiPayloadRecord(TimestampMixin, Base):
    """
    Bronze-layer raw payload archive.

    Storing raw API JSON is essential for future model training/debugging because it preserves
    schema history and fields that may not yet be represented in normalized tables.
    """

    __tablename__ = "raw_api_payloads"
    __table_args__ = (
        UniqueConstraint(
            "resource_type",
            "resource_id",
            "payload_sha256",
            name="uq_raw_api_payload_resource_hash",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ingestion_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    resource_type: Mapped[str] = mapped_column(String, nullable=False)  # case|docket|document
    resource_id: Mapped[str] = mapped_column(String, nullable=False)
    case_id: Mapped[str | None] = mapped_column(String, nullable=True)
    docket_id: Mapped[str | None] = mapped_column(String, nullable=True)
    payload_sha256: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    ingestion_run: Mapped[IngestionRunRecord | None] = relationship(back_populates="raw_payloads")
