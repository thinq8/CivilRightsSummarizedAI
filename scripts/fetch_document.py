#!/usr/bin/env python3
"""Fetch a single Clearinghouse document for quick smoke-testing."""

from __future__ import annotations

import argparse
import json
import sys

import httpx

from clearinghouse.clients import HttpClearinghouseClient, normalize_api_token
from clearinghouse.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id", help="Clearinghouse case ID (integer)")
    parser.add_argument("document_id", help="Clearinghouse document ID (integer)")
    parser.add_argument(
        "--api-token",
        dest="api_token",
        default=None,
        help="Clearinghouse API token (Token XXXXX). Defaults to CLEARINGHOUSE_API_TOKEN/.env",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help="Override API base URL (default https://clearinghouse.net/api/v2p1)",
    )
    parser.add_argument(
        "--download-text",
        dest="download_text",
        action="store_true",
        help="Attempt to download the document text via the text_url if available",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings()
    token = normalize_api_token(args.api_token or settings.api_key)
    if not token:
        print("ERROR: API token required via --api-token or CLEARINGHOUSE_API_TOKEN", file=sys.stderr)
        return 2

    base_url = args.base_url or settings.api_base_url
    with HttpClearinghouseClient(
        base_url,
        token,
        timeout=settings.api_timeout,
        user_agent=settings.user_agent,
        max_retries=settings.api_max_retries,
        backoff_seconds=settings.api_backoff_seconds,
        max_backoff_seconds=settings.api_max_backoff_seconds,
    ) as client:
        document = client.get_document(args.case_id, args.document_id)

    if not document:
        print("Document not found", file=sys.stderr)
        return 1

    text_content: str | None = document.text
    if args.download_text and document.text_url:
        with httpx.Client(headers={"Authorization": f"Token {token}", "User-Agent": settings.user_agent}) as http_client:
            response = http_client.get(document.text_url)
            response.raise_for_status()
            text_content = response.text

    payload = {
        "id": document.id,
        "title": document.title,
        "case_id": document.case_id,
        "docket_id": document.docket_id,
        "document_type": document.document_type,
        "date": document.date.isoformat() if document.date else None,
        "court": document.court,
        "has_text": document.has_text,
        "text_url": document.text_url,
        "external_url": document.external_url,
        "metadata": document.metadata,
        "text": text_content,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
