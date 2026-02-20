from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select

from clearinghouse.clients import MockClearinghouseClient, normalize_api_token
from clearinghouse.ingest import IngestionPipeline
from clearinghouse.processing import HeuristicSummarizer
from clearinghouse.storage import (
    IngestionCheckpointRecord,
    IngestionRunRecord,
    RawApiPayloadRecord,
    create_session_factory,
    init_db,
)


def test_resume_from_checkpoint_processes_remaining_cases_only(tmp_path):
    db_path = tmp_path / "resume.db"
    session_factory, engine = create_session_factory(f"sqlite:///{db_path}")
    init_db(engine)

    fixture = Path("data/fixtures/mock_dataset.json")
    client = MockClearinghouseClient(fixture)
    pipeline = IngestionPipeline(
        client,
        session_factory,
        HeuristicSummarizer(max_sentences=2),
        source="mock",
        checkpoint_key="mock-default",
        archive_raw_payloads=False,
        continue_on_error=True,
    )

    first_stats = pipeline.run(case_limit=1, resume_from_checkpoint=True)
    assert first_stats.cases == 1

    second_stats = pipeline.run(resume_from_checkpoint=True)
    assert second_stats.resumed_from_checkpoint is True
    assert second_stats.cases == 1
    assert second_stats.dockets == 1
    assert second_stats.documents == 2

    with session_factory() as session:
        checkpoint = session.get(IngestionCheckpointRecord, "mock-default")
        assert checkpoint is not None
        assert checkpoint.last_case_id == "case-001"


def test_raw_payload_archive_dedupes_by_payload_hash(tmp_path):
    db_path = tmp_path / "raw_payloads.db"
    session_factory, engine = create_session_factory(f"sqlite:///{db_path}")
    init_db(engine)

    fixture = Path("data/fixtures/mock_dataset.json")
    client = MockClearinghouseClient(fixture)
    pipeline = IngestionPipeline(
        client,
        session_factory,
        HeuristicSummarizer(max_sentences=2),
        source="mock",
        checkpoint_key=None,
        archive_raw_payloads=True,
        continue_on_error=True,
    )

    pipeline.run()
    pipeline.run()

    with session_factory() as session:
        raw_payload_count = session.scalar(select(func.count()).select_from(RawApiPayloadRecord))
        run_count = session.scalar(select(func.count()).select_from(IngestionRunRecord))
        assert raw_payload_count == 8  # 2 cases + 2 dockets + 4 documents
        assert run_count == 2


def test_normalize_api_token_strips_optional_prefix():
    assert normalize_api_token("abc123") == "abc123"
    assert normalize_api_token("Token abc123") == "abc123"
    assert normalize_api_token("  token abc123  ") == "abc123"
    assert normalize_api_token("   ") == ""
