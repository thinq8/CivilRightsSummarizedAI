from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import httpx
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from clearinghouse.clients.http import normalize_api_token
from clearinghouse.storage.models import DocumentRecord


DB_URL = os.getenv("CLEARINGHOUSE_DATABASE_URL", "sqlite:///data/live.db")
BASE_URL = os.getenv("CLEARINGHOUSE_API_BASE_URL", "https://clearinghouse.net")

MIN_INTERVAL_SECONDS = 1.2
BATCH_SIZE = 50

MAX_RETRIES = 5
BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0

EMPTY_TEXT_PLACEHOLDER = "[[NO_TEXT_RETURNED]]"


def compute_backoff(attempt: int, response: httpx.Response | None = None) -> float:
    retry_after = response.headers.get("Retry-After") if response is not None else None

    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass

    expo = BACKOFF_SECONDS * (2 ** attempt)
    jitter = random.uniform(0.0, BACKOFF_SECONDS)
    return max(0.0, min(expo + jitter, MAX_BACKOFF_SECONDS))


def fetch_text(client: httpx.Client, url: str, last_request_time: float) -> tuple[str, float]:
    elapsed = time.monotonic() - last_request_time
    wait_time = MIN_INTERVAL_SECONDS - elapsed

    if wait_time > 0:
        time.sleep(wait_time)

    for attempt in range(MAX_RETRIES + 1):
        request_started = time.monotonic()
        response = client.get(url)

        if response.status_code == 200:
            payload = response.json()
            text_value = payload.get("text")

            if text_value is None:
                return EMPTY_TEXT_PLACEHOLDER, request_started

            if isinstance(text_value, str) and text_value.strip() == "":
                return EMPTY_TEXT_PLACEHOLDER, request_started

            return str(text_value), request_started

        if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < MAX_RETRIES:
            time.sleep(compute_backoff(attempt, response))
            continue

        response.raise_for_status()

    return EMPTY_TEXT_PLACEHOLDER, time.monotonic()


def commit_with_retry(session, max_attempts: int = 3) -> None:
    for attempt in range(max_attempts):
        try:
            session.commit()
            return
        except OperationalError as exc:
            session.rollback()

            if "database is locked" not in str(exc).lower() or attempt == max_attempts - 1:
                raise

            wait_seconds = 5 * (attempt + 1)
            print(f"Database locked. Waiting {wait_seconds}s before retrying commit...")
            time.sleep(wait_seconds)


def main() -> None:
    token = normalize_api_token(os.getenv("CLEARINGHOUSE_API_TOKEN"))

    if not token:
        raise ValueError("CLEARINGHOUSE_API_TOKEN is not set")

    engine = create_engine(
        DB_URL,
        connect_args={"timeout": 60},
    )

    SessionLocal = sessionmaker(bind=engine)

    with httpx.Client(
        base_url=BASE_URL,
        headers={
            "Authorization": f"Token {token}",
            "User-Agent": "CivilRightsSummarizedAI/0.1",
        },
        timeout=60.0,
    ) as client, SessionLocal() as session:
        remaining_query = select(func.count()).select_from(DocumentRecord).where(
            DocumentRecord.has_text.is_(True),
            DocumentRecord.text_url.is_not(None),
            DocumentRecord.text.is_(None),
        )

        total_remaining = session.execute(remaining_query).scalar_one()
        print(f"Documents remaining to hydrate: {total_remaining}")

        last_request_time = 0.0
        processed = 0
        stored_text_count = 0
        placeholder_count = 0

        while True:
            docs_query = (
                select(DocumentRecord)
                .where(
                    DocumentRecord.has_text.is_(True),
                    DocumentRecord.text_url.is_not(None),
                    DocumentRecord.text.is_(None),
                )
                .order_by(DocumentRecord.id)
                .limit(BATCH_SIZE)
            )

            docs = session.execute(docs_query).scalars().all()

            if not docs:
                break

            for doc in docs:
                try:
                    text_value, last_request_time = fetch_text(
                        client,
                        doc.text_url,
                        last_request_time,
                    )

                    doc.text = text_value

                    if text_value == EMPTY_TEXT_PLACEHOLDER:
                        placeholder_count += 1
                    else:
                        stored_text_count += 1

                    processed += 1

                    if processed % 25 == 0:
                        print(
                            f"Processed {processed} docs "
                            f"(stored: {stored_text_count}, placeholders: {placeholder_count})"
                        )

                except Exception as exc:
                    print(f"Failed doc {doc.id}: {exc}")
                    doc.text = EMPTY_TEXT_PLACEHOLDER
                    placeholder_count += 1
                    processed += 1

            commit_with_retry(session)

        print(
            f"Done. Processed {processed} docs "
            f"(stored text: {stored_text_count}, placeholders: {placeholder_count})"
        )


if __name__ == "__main__":
    main()
