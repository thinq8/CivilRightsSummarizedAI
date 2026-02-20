from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from clearinghouse.clients import HttpClearinghouseClient, MockClearinghouseClient, normalize_api_token
from clearinghouse.config import Settings
from clearinghouse.ingest import IngestionPipeline
from clearinghouse.processing import HeuristicSummarizer
from clearinghouse.storage import create_session_factory, init_db

app = typer.Typer(help="Clearinghouse ingestion + summarization CLI")


@app.callback()
def configure_logging(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@app.command("ingest-mock")
def ingest_mock(
    since: Optional[str] = typer.Option(
        None,
        help="ISO timestamp to resume ingestion from. Example: 2023-01-01T00:00:00Z",
    ),
    case_limit: Optional[int] = typer.Option(None, help="Limit number of cases ingested"),
    db_url: Optional[str] = typer.Option(None, envvar="CLEARINGHOUSE_DATABASE_URL"),
    fixture: Optional[Path] = typer.Option(None, help="Path to fixture JSON"),
    checkpoint_key: str = typer.Option("mock-default", help="Checkpoint key used for resume mode"),
    resume_from_checkpoint: bool = typer.Option(
        False,
        "--resume-from-checkpoint/--no-resume-from-checkpoint",
        help="Resume from stored checkpoint instead of only using --since",
    ),
    archive_raw_payloads: bool = typer.Option(
        True,
        "--archive-raw-payloads/--no-archive-raw-payloads",
        help="Persist raw API/fixture payload JSON for data lineage",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--strict",
        help="Continue with other cases when one case fails ingestion",
    ),
) -> None:
    settings = Settings()
    database_url = db_url or settings.database_url
    fixture_path = fixture or settings.fixture_path

    session_factory, engine = create_session_factory(database_url)
    init_db(engine)

    client = MockClearinghouseClient(fixture_path)
    summarizer = HeuristicSummarizer()
    pipeline = IngestionPipeline(
        client,
        session_factory,
        summarizer,
        source="mock",
        checkpoint_key=checkpoint_key,
        archive_raw_payloads=archive_raw_payloads,
        continue_on_error=continue_on_error,
    )

    since_dt = _parse_since(since)
    stats = pipeline.run(
        since=since_dt,
        case_limit=case_limit,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    typer.echo(
        "Ingestion complete: "
        f"run_id={stats.run_id} "
        f"cases={stats.cases} dockets={stats.dockets} documents={stats.documents} errors={stats.errors}"
    )


@app.command("ingest-live")
def ingest_live(
    since: Optional[str] = typer.Option(None, help="ISO timestamp to resume ingestion from"),
    case_limit: Optional[int] = typer.Option(None, help="Limit number of cases ingested"),
    db_url: Optional[str] = typer.Option(None, envvar="CLEARINGHOUSE_DATABASE_URL"),
    api_token: Optional[str] = typer.Option(
        None,
        envvar="CLEARINGHOUSE_API_TOKEN",
        help="Clearinghouse API token value. Optional 'Token ' prefix is accepted.",
    ),
    checkpoint_key: Optional[str] = typer.Option(
        None,
        help="Checkpoint key used for incremental resume (defaults to CLEARINGHOUSE_LIVE_CHECKPOINT_KEY)",
    ),
    resume_from_checkpoint: Optional[bool] = typer.Option(
        None,
        "--resume-from-checkpoint/--no-resume-from-checkpoint",
        help="Resume from stored checkpoint timestamp before listing cases",
    ),
    archive_raw_payloads: Optional[bool] = typer.Option(
        None,
        "--archive-raw-payloads/--no-archive-raw-payloads",
        help="Persist raw API payload JSON for lineage and future training exports",
    ),
    continue_on_error: Optional[bool] = typer.Option(
        None,
        "--continue-on-error/--strict",
        help="Continue with other cases when one case fails ingestion",
    ),
) -> None:
    settings = Settings()
    database_url = db_url or settings.database_url
    token = normalize_api_token(api_token or settings.api_key)
    if not token:
        raise typer.BadParameter("An API token is required via --api-token or CLEARINGHOUSE_API_TOKEN")

    effective_checkpoint_key = checkpoint_key or settings.live_checkpoint_key
    effective_resume = (
        resume_from_checkpoint
        if resume_from_checkpoint is not None
        else settings.live_resume_from_checkpoint
    )
    effective_archive_raw_payloads = (
        archive_raw_payloads
        if archive_raw_payloads is not None
        else settings.archive_raw_payloads
    )
    effective_continue_on_error = (
        continue_on_error if continue_on_error is not None else settings.continue_on_error
    )

    session_factory, engine = create_session_factory(database_url)
    init_db(engine)

    with HttpClearinghouseClient(
        settings.api_base_url,
        token,
        timeout=settings.api_timeout,
        user_agent=settings.user_agent,
        max_retries=settings.api_max_retries,
        backoff_seconds=settings.api_backoff_seconds,
        max_backoff_seconds=settings.api_max_backoff_seconds,
    ) as client:
        pipeline = IngestionPipeline(
            client,
            session_factory,
            HeuristicSummarizer(),
            source="live",
            checkpoint_key=effective_checkpoint_key,
            archive_raw_payloads=effective_archive_raw_payloads,
            continue_on_error=effective_continue_on_error,
        )
        since_dt = _parse_since(since)
        stats = pipeline.run(
            since=since_dt,
            case_limit=case_limit,
            resume_from_checkpoint=effective_resume,
        )
    typer.echo(
        "Live ingestion complete: "
        f"run_id={stats.run_id} "
        f"cases={stats.cases} dockets={stats.dockets} documents={stats.documents} errors={stats.errors}"
    )


def _parse_since(value: Optional[str]) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        # Treat naive values as UTC so CLI behavior is deterministic across machines.
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


if __name__ == "__main__":
    app()
