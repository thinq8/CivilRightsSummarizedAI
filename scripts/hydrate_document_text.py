from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import httpx
from sqlalchemy import create_engine, func, or_, select
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from clearinghouse.clients.http import normalize_api_token
from clearinghouse.storage.models import DocumentRecord


DB_URL = "sqlite:////Volumes/LaCie/clearinghouse_data/live.db"
BASE_URL = "https://clearinghouse.net"

MIN_INTERVAL_SECONDS = 1.5
BATCH_SIZE = 5  # keep small for first test

MAX_RETRIES = 5
BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0


def compute_backoff(attempt: int, response: httpx.Response | None = None) -> float:
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass

    expo = BACKOFF_SECONDS * (2**attempt)
    jitter = random.uniform(0.0, BACKOFF_SECONDS)
    return max(0.0, min(expo + jitter, MAX_BACKOFF_SECONDS))


def fetch_text(client: httpx.Client, url: str, last_request_time: float) -> tuple[str | None, float]:
    elapsed = time.monotonic() - last_request_time
    wait_time = MIN_INTERVAL_SECONDS - elapsed

    if wait_time > 0:
        time.sleep(wait_time)

    for attempt in range(MAX_RETRIES + 1):
        request_started = time.monotonic()
        response = client.get(url)

        if response.status_code == 200:
            payload = response.json()
            return payload.get("text"), request_started

        if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < MAX_RETRIES:
            time.sleep(compute_backoff(attempt, response))
            continue

        response.raise_for_status()

    return None, time.monotonic()


def main() -> None:
    token = normalize_api_token(os.getenv("CLEARINGHOUSE_API_TOKEN"))

    if not token:
        raise ValueError("CLEARINGHOUSE_API_TOKEN is not set")

    engine = create_engine(DB_URL)
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
            or_(DocumentRecord.text.is_(None), DocumentRecord.text == ""),
        )

        total_remaining = session.execute(remaining_query).scalar_one()

        print(f"Documents remaining to hydrate: {total_remaining}")

        last_request_time = 0.0
        processed = 0

        while True:
            docs_query = select(DocumentRecord).where(
                DocumentRecord.has_text.is_(True),
                DocumentRecord.text_url.is_not(None),
                or_(DocumentRecord.text.is_(None), DocumentRecord.text == ""),
            ).limit(BATCH_SIZE)

            docs = session.execute(docs_query).scalars().all()

            if not docs:
                break

            for doc in docs:
                try:
                    text_value, last_request_time = fetch_text(client, doc.text_url, last_request_time)

                    doc.text = text_value if text_value is not None else ""

                    processed += 1

                    if processed % 25 == 0:
                        print(f"Hydrated {processed} documents...")

                except Exception as exc:
                    print(f"Failed document {doc.id}: {exc}")

            session.commit()

        print(f"Done. Hydrated {processed} documents.")


if __name__ == "__main__":
    main()
