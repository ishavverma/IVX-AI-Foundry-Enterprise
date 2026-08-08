"""
AFE — Dataset Cleaner

Provides text normalization, sanitization, boilerplate removal,
HTML/markup stripping, and regex-based filtering.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class CleaningOptions:
    normalize_unicode: bool = True
    strip_html: bool = True
    remove_control_chars: bool = True
    normalize_whitespace: bool = True
    mask_emails: bool = False
    mask_ips: bool = False
    min_length: int = 10
    max_length: int = 1_000_000


class DatasetCleaner:
    """Sanitizes text according to enterprise data preparation standards."""

    # Regex patterns
    HTML_PATTERN = re.compile(r"<[^>]+>")
    EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
    IP_PATTERN = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
    CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
    MULTI_WHITESPACE_PATTERN = re.compile(r"[ \t]+")
    MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")

    @classmethod
    def clean_text(cls, text: str, options: CleaningOptions | None = None) -> str:
        """Apply configured sanitization rules to a raw string."""
        if not text or not isinstance(text, str):
            return ""
        
        opts = options or CleaningOptions()
        cleaned = text

        if opts.normalize_unicode:
            cleaned = unicodedata.normalize("NFKC", cleaned)

        if opts.strip_html:
            cleaned = cls.HTML_PATTERN.sub(" ", cleaned)

        if opts.remove_control_chars:
            cleaned = cls.CONTROL_CHAR_PATTERN.sub("", cleaned)

        if opts.mask_emails:
            cleaned = cls.EMAIL_PATTERN.sub("<EMAIL>", cleaned)

        if opts.mask_ips:
            cleaned = cls.IP_PATTERN.sub("<IP_ADDRESS>", cleaned)

        if opts.normalize_whitespace:
            cleaned = cls.MULTI_WHITESPACE_PATTERN.sub(" ", cleaned)
            cleaned = cls.MULTI_NEWLINE_PATTERN.sub("\n\n", cleaned)
            cleaned = cleaned.strip()

        if len(cleaned) < opts.min_length or len(cleaned) > opts.max_length:
            return ""

        return cleaned

    @classmethod
    def clean_record(cls, record: Dict[str, Any], text_field: str = "text", options: CleaningOptions | None = None) -> Dict[str, Any] | None:
        """Sanitize record dictionary. Returns None if filtered out."""
        raw_text = record.get(text_field, "")
        cleaned_text = cls.clean_text(str(raw_text), options)
        if not cleaned_text:
            return None
        
        res = dict(record)
        res[text_field] = cleaned_text
        res["_cleaned_len"] = len(cleaned_text)
        return res
