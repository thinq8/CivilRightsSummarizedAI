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

# Main Typer app for all command-line actions in this project.
# This file is the entry point for running ingestion jobs from the terminal.
app = typer.Typer(help="Clearinghouse ingestion + summarization CLI")


@app.callback()
def configure_logging(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    # Set log level for the whole CLI.
    # Use --verbose to see more detailed debug output while running commands.
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
    # Load project settings, then allow CLI options to override them.
    settings = Settings()
    database_url = db_url or settings.database_url
    fixture_path = fixture or settings.fixture_path

    # Create the database session factory and initialize tables if needed.
    session_factory, engine = create_session_factory(database_url)
    init_db(engine)

    # Mock client reads from a local fixture instead of the live API.
    # This is useful for testing and development.
    client = MockClearinghouseClient(fixture_path)

    # The current summarizer is heuristic-based.
    # It generates simple summaries during ingestion.
    summarizer = HeuristicSummarizer()

    # The ingestion pipeline handles the actual workflow:
    # fetch cases, fetch related records, summarize, and save everything.
    pipeline = IngestionPipeline(
        client,
        session_factory,
        summarizer,
        source="mock",
        checkpoint_key=checkpoint_key,
        archive_raw_payloads=archive_raw_payloads,
        continue_on_error=continue_on_error,
    )

    # Parse the optional --since timestamp and run the ingestion job.
    since_dt = _parse_since(since)
    stats = pipeline.run(
        since=since_dt,
        case_limit=case_limit,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    # Print a short completion summary to the terminal.
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
    # Load default settings from the config, then override with CLI arguments if provided.
    settings = Settings()
    database_url = db_url or settings.database_url

    # Normalize the API token so users can pass it with or without the "Token " prefix.
    token = normalize_api_token(api_token or settings.api_key)
    if not token:
        raise typer.BadParameter("An API token is required via --api-token or CLEARINGHOUSE_API_TOKEN")

    # Decide which runtime settings to use.
    # CLI options take priority, otherwise fall back to project settings.
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

    # Set up the database connection and ensure tables exist.
    session_factory, engine = create_session_factory(database_url)
    init_db(engine)

    # Open a live HTTP client for the Clearinghouse API.
    # This context manager handles setup and cleanup of the client.
    with HttpClearinghouseClient(
        settings.api_base_url,
        token,
        timeout=settings.api_timeout,
        user_agent=settings.user_agent,
        max_retries=settings.api_max_retries,
        backoff_seconds=settings.api_backoff_seconds,
        max_backoff_seconds=settings.api_max_backoff_seconds,
    ) as client:
        # Build the ingestion pipeline for the live API.
        pipeline = IngestionPipeline(
            client,
            session_factory,
            HeuristicSummarizer(),
            source="live",
            checkpoint_key=effective_checkpoint_key,
            archive_raw_payloads=effective_archive_raw_payloads,
            continue_on_error=effective_continue_on_error,
        )

        # Parse the optional start time and run the live ingestion job.
        since_dt = _parse_since(since)
        stats = pipeline.run(
            since=since_dt,
            case_limit=case_limit,
            resume_from_checkpoint=effective_resume,
        )

    # Print a short completion summary to the terminal.
    typer.echo(
        "Live ingestion complete: "
        f"run_id={stats.run_id} "
        f"cases={stats.cases} dockets={stats.dockets} documents={stats.documents} errors={stats.errors}"
    )


def _parse_since(value: Optional[str]) -> datetime | None:
    # Convert a CLI timestamp string into a datetime object.
    # Accepts ISO format and treats naive timestamps as UTC for consistency.
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        # Treat naive values as UTC so CLI behavior is deterministic across machines.
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


if __name__ == "__main__":
    # Run the Typer app when this file is executed directly.
    app()
