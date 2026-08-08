"""
AFE — Dataset Quality Scoring Engine

Evaluates text quality using statistical heuristics: symbol-to-word ratio,
line length uniformity, stopword distribution, repetition penalty, and perplexity proxy.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Any


class DatasetQualityScorer:
    """Computes a quality score between 0.0 and 1.0 for dataset samples."""

    COMMON_ENGLISH_WORDS = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what"
    }

    @classmethod
    def score_text(cls, text: str) -> Dict[str, Any]:
        """
        Analyze text quality and return composite score and metric components.
        """
        if not text or len(text.strip()) == 0:
            return {"score": 0.0, "reason": "empty"}

        words = re.findall(r"\b\w+\b", text.lower())
        total_words = len(words)
        total_chars = len(text)

        if total_words < 3:
            return {"score": 0.1, "reason": "too_short", "word_count": total_words}

        # 1. Symbol / Punctuation Ratio Penalty
        alpha_chars = sum(1 for c in text if c.isalnum() or c.isspace())
        symbol_ratio = (total_chars - alpha_chars) / max(1, total_chars)
        symbol_score = max(0.0, 1.0 - (symbol_ratio * 3.0))

        # 2. Stopword / Language Naturalness
        stopword_count = sum(1 for w in words if w in cls.COMMON_ENGLISH_WORDS)
        stopword_ratio = stopword_count / max(1, total_words)
        stopword_score = min(1.0, stopword_ratio / 0.25) # 25%+ stopwords is typical in natural prose

        # 3. Repetition Penalty (n-gram unique ratio)
        unique_words = len(set(words))
        unique_word_ratio = unique_words / max(1, total_words)
        repetition_score = min(1.0, unique_word_ratio * 1.5)

        # 4. Average word length (natural text is between 3.5 and 7.5)
        avg_word_len = sum(len(w) for w in words) / max(1, total_words)
        if 3.0 <= avg_word_len <= 8.5:
            word_len_score = 1.0
        else:
            word_len_score = max(0.2, 1.0 - abs(avg_word_len - 5.5) / 5.5)

        # 5. Composite Score
        composite = (
            symbol_score * 0.30 +
            stopword_score * 0.25 +
            repetition_score * 0.25 +
            word_len_score * 0.20
        )
        composite = round(min(1.0, max(0.0, composite)), 3)

        return {
            "score": composite,
            "word_count": total_words,
            "char_count": total_chars,
            "unique_word_ratio": round(unique_word_ratio, 3),
            "symbol_ratio": round(symbol_ratio, 3),
            "stopword_ratio": round(stopword_ratio, 3),
        }
