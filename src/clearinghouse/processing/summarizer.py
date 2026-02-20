from __future__ import annotations

import re
from typing import Iterable

from clearinghouse.types import Document


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class HeuristicSummarizer:
    """Extremely lightweight summarizer based on metadata + leading sentences."""

    def __init__(self, max_sentences: int = 4):
        self.max_sentences = max_sentences

    def summarize(self, document: Document) -> str:
        meta_bits = [document.document_type or "Document"]
        if document.court:
            meta_bits.append(document.court)
        if document.metadata.get("subject"):
            meta_bits.append(str(document.metadata["subject"]))
        header = f"Summary for {document.title} ({', '.join(meta_bits)})"

        sentences = list(_first_sentences(document.text or "", self.max_sentences))
        if not sentences:
            sentences = ["No text available for summarization yet."]

        bullets = "\n".join(f"- {sentence}" for sentence in sentences)
        return f"{header}\n{bullets}"


def _first_sentences(text: str, limit: int) -> Iterable[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    remaining = limit
    for paragraph in cleaned.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            normalized = sentence.strip()
            if not normalized:
                continue
            yield normalized
            remaining -= 1
            if remaining <= 0:
                return


def summarize_document(document: Document, max_sentences: int = 4) -> str:
    return HeuristicSummarizer(max_sentences=max_sentences).summarize(document)
