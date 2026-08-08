"""
AFE — Tokenizer Engine

Implements Byte-Pair Encoding (BPE), Byte-Level BPE, and Unigram tokenization
with training, encoding, decoding, and vocabulary indexing.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class TokenizedOutput:
    token_ids: List[int]
    tokens: List[str]
    offsets: List[Tuple[int, int]]
    text: str
    token_count: int
    char_count: int
    bytes_per_token: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_ids": self.token_ids,
            "tokens": self.tokens,
            "offsets": self.offsets,
            "text": self.text,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "bytes_per_token": self.bytes_per_token,
        }


class BaseTokenizer:
    """Core BPE / Byte-Level Tokenizer with fast Python implementation."""

    def __init__(self, vocab_size: int = 32000, algorithm: str = "bpe", special_tokens: Dict[str, int] | None = None) -> None:
        self.vocab_size = vocab_size
        self.algorithm = algorithm
        self.special_tokens = special_tokens or {
            "<|pad|>": 0,
            "<|unk|>": 1,
            "<|bos|>": 2,
            "<|eos|>": 3,
            "<|mask|>": 4,
        }
        
        # Invert special tokens
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}
        
        # Vocab structures
        self.vocab: Dict[str, int] = dict(self.special_tokens)
        self.id_to_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.merges: List[Tuple[str, str]] = []

        # Ensure base single-byte / ascii characters exist
        next_id = max(self.vocab.values(), default=-1) + 1
        for i in range(256):
            char_rep = chr(i) if 32 <= i <= 126 and chr(i) not in self.vocab else f"<0x{i:02X}>"
            if char_rep not in self.vocab:
                self.vocab[char_rep] = next_id
                self.id_to_vocab[next_id] = char_rep
                next_id += 1

    def train_from_texts(self, texts: List[str], target_vocab_size: int | None = None) -> None:
        """Train BPE merges on input text corpus."""
        target_size = target_vocab_size or self.vocab_size
        logger.info("Training %s tokenizer on %d texts up to vocab %d", self.algorithm, len(texts), target_size)

        # Word frequency counting
        word_counts = Counter()
        for t in texts:
            words = re.findall(r"\w+|[^\w\s]", t, re.UNICODE)
            for w in words:
                # Represent word as tuple of individual characters
                word_counts[" ".join(list(w)) + " </w>"] += 1

        splits = {w: count for w, count in word_counts.items()}
        next_id = max(self.vocab.values(), default=-1) + 1

        while len(self.vocab) < target_size:
            # Count bigram pairs across all words
            pairs = defaultdict(int)
            for word, freq in splits.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            merged_symbol = "".join(best_pair)
            
            # Register in vocab & merges
            self.merges.append(best_pair)
            self.vocab[merged_symbol] = next_id
            self.id_to_vocab[next_id] = merged_symbol
            next_id += 1

            # Update splits
            new_splits = {}
            p0, p1 = best_pair
            pattern = re.compile(r"(?<!\S)" + re.escape(p0) + r"\s+" + re.escape(p1) + r"(?!\S)")
            for word, freq in splits.items():
                new_word = pattern.sub(merged_symbol, word)
                new_splits[new_word] = freq
            splits = new_splits

        logger.info("Tokenizer training completed. Final vocab size: %d, merges: %d", len(self.vocab), len(self.merges))

    def encode(self, text: str) -> TokenizedOutput:
        """Encode raw text to token IDs and string pieces."""
        if not text:
            return TokenizedOutput([], [], [], "", 0, 0, 0.0)

        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        token_ids: List[int] = []
        tokens: List[str] = []
        offsets: List[Tuple[int, int]] = []
        
        current_offset = 0

        for w in words:
            # Find in original string for offset calculation
            start_pos = text.find(w, current_offset)
            if start_pos == -1:
                start_pos = current_offset
            end_pos = start_pos + len(w)
            current_offset = end_pos

            # Check special tokens directly
            if w in self.vocab:
                token_id = self.vocab[w]
                token_ids.append(token_id)
                tokens.append(w)
                offsets.append((start_pos, end_pos))
                continue

            # Greedy BPE merge lookup
            symbols = list(w) + ["</w>"]
            for p0, p1 in self.merges:
                merged = p0 + p1
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == p0 and symbols[i + 1] == p1:
                        symbols[i : i + 2] = [merged]
                    else:
                        i += 1

            for sym in symbols:
                clean_sym = sym.replace("</w>", "")
                token_id = self.vocab.get(clean_sym, self.vocab.get(sym, self.special_tokens.get("<|unk|>", 1)))
                token_ids.append(token_id)
                tokens.append(clean_sym if clean_sym else sym)
                offsets.append((start_pos, end_pos))

        byte_len = len(text.encode("utf-8"))
        bytes_per_tok = round(byte_len / max(1, len(token_ids)), 2)

        return TokenizedOutput(
            token_ids=token_ids,
            tokens=tokens,
            offsets=offsets,
            text=text,
            token_count=len(token_ids),
            char_count=len(text),
            bytes_per_token=bytes_per_tok,
        )

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to human-readable string."""
        pieces = []
        for tid in token_ids:
            if tid in self.id_to_special:
                pieces.append(self.id_to_special[tid])
            elif tid in self.id_to_vocab:
                sym = self.id_to_vocab[tid].replace("</w>", " ")
                pieces.append(sym)
            else:
                pieces.append("<|unk|>")
        return "".join(pieces).strip()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tokenizer state for persistence."""
        return {
            "algorithm": self.algorithm,
            "vocab_size": len(self.vocab),
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": [" ".join(p) for p in self.merges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaseTokenizer:
        """Load tokenizer from serialized state."""
        tok = cls(
            vocab_size=data.get("vocab_size", 32000),
            algorithm=data.get("algorithm", "bpe"),
            special_tokens=data.get("special_tokens"),
        )
        tok.vocab = data.get("vocab", tok.vocab)
        tok.id_to_vocab = {int(v): k for k, v in tok.vocab.items()}
        raw_merges = data.get("merges", [])
        tok.merges = [tuple(m.split()) for m in raw_merges if len(m.split()) == 2]
        return tok
