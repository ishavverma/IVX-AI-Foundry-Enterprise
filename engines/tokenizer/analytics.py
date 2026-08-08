"""
AFE — Tokenizer Analytics

Calculates token compression metrics, multilingual script / CJK coverage,
fertility ratios (tokens/word), and vocabulary frequency distribution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Any

from engines.tokenizer.engine import BaseTokenizer


@dataclass
class TokenizerMetrics:
    total_tokens: int
    total_chars: int
    total_bytes: int
    bytes_per_token: float
    chars_per_token: float
    tokens_per_word: float
    cjk_coverage_pct: float
    unknown_token_count: int
    unknown_token_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "total_chars": self.total_chars,
            "total_bytes": self.total_bytes,
            "bytes_per_token": self.bytes_per_token,
            "chars_per_token": self.chars_per_token,
            "tokens_per_word": self.tokens_per_word,
            "cjk_coverage_pct": self.cjk_coverage_pct,
            "unknown_token_count": self.unknown_token_count,
            "unknown_token_pct": self.unknown_token_pct,
        }


class TokenizerAnalytics:
    """Provides deep diagnostic inspection for trained tokenizers."""

    CJK_REGEX = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")

    @classmethod
    def evaluate_corpus(cls, tokenizer: BaseTokenizer, sample_texts: List[str]) -> TokenizerMetrics:
        """Run diagnostic suite over a representative text sample."""
        total_tokens = 0
        total_chars = 0
        total_bytes = 0
        total_words = 0
        unknown_tokens = 0
        cjk_chars = 0
        cjk_matched_tokens = 0

        unk_id = tokenizer.special_tokens.get("<|unk|>", 1)

        for text in sample_texts:
            if not text:
                continue
            encoded = tokenizer.encode(text)
            
            t_count = len(encoded.token_ids)
            c_count = len(text)
            b_count = len(text.encode("utf-8"))
            w_count = len(re.findall(r"\b\w+\b", text))

            total_tokens += t_count
            total_chars += c_count
            total_bytes += b_count
            total_words += max(1, w_count)

            unknown_tokens += sum(1 for tid in encoded.token_ids if tid == unk_id)

            # CJK detection
            cjk_in_text = cls.CJK_REGEX.findall(text)
            cjk_chars += len(cjk_in_text)
            for tok_str in encoded.tokens:
                if cls.CJK_REGEX.search(tok_str):
                    cjk_matched_tokens += 1

        bytes_per_tok = round(total_bytes / max(1, total_tokens), 2)
        chars_per_tok = round(total_chars / max(1, total_tokens), 2)
        toks_per_word = round(total_tokens / max(1, total_words), 2)
        
        cjk_coverage = (
            round((cjk_matched_tokens / max(1, cjk_chars)) * 100, 1)
            if cjk_chars > 0 else 100.0
        )
        unk_pct = round((unknown_tokens / max(1, total_tokens)) * 100, 2)

        return TokenizerMetrics(
            total_tokens=total_tokens,
            total_chars=total_chars,
            total_bytes=total_bytes,
            bytes_per_token=bytes_per_tok,
            chars_per_token=chars_per_tok,
            tokens_per_word=toks_per_word,
            cjk_coverage_pct=cjk_coverage,
            unknown_token_count=unknown_tokens,
            unknown_token_pct=unk_pct,
        )

    @classmethod
    def search_vocab(cls, tokenizer: BaseTokenizer, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search vocabulary tokens by exact, substring, or ID."""
        results = []
        q_lower = query.lower()
        
        for tid, token_str in sorted(tokenizer.id_to_vocab.items()):
            if query.isdigit() and int(query) == tid:
                results.append({"id": tid, "token": token_str, "is_special": tid in tokenizer.id_to_special})
            elif q_lower in token_str.lower():
                results.append({"id": tid, "token": token_str, "is_special": tid in tokenizer.id_to_special})
            if len(results) >= limit:
                break

        return results
