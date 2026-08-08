"""
AFE — Dataset Chunking Engine

Splits documents into model-ready sequences using token-approximate,
sentence-boundary, or paragraph-aware chunking with customizable overlap.
"""

from __future__ import annotations

import re
from typing import Dict, List, Any


class DatasetChunker:
    """Chunks large document records into fixed or boundary-aware token sequences."""

    SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

    @classmethod
    def chunk_by_words(
        cls,
        text: str,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> List[str]:
        """Sliding window chunking by approximate word count."""
        words = text.split()
        if not words:
            return []
        
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        step = max(1, chunk_size - overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + chunk_size >= len(words):
                break

        return chunks

    @classmethod
    def chunk_by_sentences(
        cls,
        text: str,
        max_words_per_chunk: int = 512,
    ) -> List[str]:
        """Groups whole sentences until chunk size limit is reached."""
        sentences = cls.SENTENCE_SPLIT_PATTERN.split(text.strip())
        chunks = []
        current_chunk: List[str] = []
        current_word_count = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            words_in_sent = len(sentence.split())
            if current_word_count + words_in_sent > max_words_per_chunk and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = words_in_sent
            else:
                current_chunk.append(sentence)
                current_word_count += words_in_sent

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @classmethod
    def chunk_record(
        cls,
        record: Dict[str, Any],
        text_field: str = "text",
        chunk_size: int = 512,
        overlap: int = 64,
        strategy: str = "words",
    ) -> List[Dict[str, Any]]:
        """Splits a single record dict into multiple chunked record dicts."""
        raw_text = str(record.get(text_field, ""))
        if not raw_text:
            return []

        if strategy == "sentences":
            chunks = cls.chunk_by_sentences(raw_text, chunk_size)
        else:
            chunks = cls.chunk_by_words(raw_text, chunk_size, overlap)

        chunked_records = []
        for idx, chunk in enumerate(chunks):
            r = dict(record)
            r[text_field] = chunk
            r["_chunk_id"] = idx
            r["_chunk_total"] = len(chunks)
            chunked_records.append(r)

        return chunked_records
