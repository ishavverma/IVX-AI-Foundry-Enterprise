"""
AFE — Dataset Deduplication Engine

Implements exact cryptographic deduplication (SHA-256) and approximate
near-deduplication using MinHash and N-gram shingles (Jaccard similarity).
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Set, Tuple


class DatasetDeduplicator:
    """Detects and filters exact and near-duplicate text items."""

    def __init__(self, num_perm: int = 64, jaccard_threshold: float = 0.8, n_gram: int = 5) -> None:
        self.num_perm = num_perm
        self.jaccard_threshold = jaccard_threshold
        self.n_gram = n_gram
        self._exact_hashes: Set[str] = set()
        self._minhashes: List[Tuple[str, List[int]]] = [] # (id/hash, signature)

    def _get_shingles(self, text: str) -> Set[str]:
        """Convert text to n-gram word shingles."""
        words = re.findall(r"\w+", text.lower())
        if len(words) < self.n_gram:
            return {" ".join(words)}
        return {" ".join(words[i : i + self.n_gram]) for i in range(len(words) - self.n_gram + 1)}

    def _compute_minhash(self, shingles: Set[str]) -> List[int]:
        """Generate a compact integer MinHash signature from shingles."""
        sig = []
        for i in range(self.num_perm):
            min_val = float("inf")
            for shingle in shingles:
                # Deterministic salt hash
                h = int(hashlib.md5(f"{i}:{shingle}".encode("utf-8")).hexdigest()[:8], 16)
                if h < min_val:
                    min_val = h
            sig.append(int(min_val) if min_val != float("inf") else 0)
        return sig

    def _estimate_jaccard(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        if not sig1 or not sig2 or len(sig1) != len(sig2):
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def is_duplicate(self, text: str) -> Tuple[bool, str, float]:
        """
        Check if text is an exact or near duplicate.
        Returns: (is_duplicate: bool, match_type: 'exact'|'near'|'none', similarity: float)
        """
        if not text or not text.strip():
            return True, "exact", 1.0

        # 1. Exact hash check
        exact_h = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if exact_h in self._exact_hashes:
            return True, "exact", 1.0

        # 2. Near-duplicate MinHash check
        shingles = self._get_shingles(text)
        sig = self._compute_minhash(shingles)

        for stored_id, stored_sig in self._minhashes:
            sim = self._estimate_jaccard(sig, stored_sig)
            if sim >= self.jaccard_threshold:
                return True, "near", sim

        # Not duplicate — register
        self._exact_hashes.add(exact_h)
        self._minhashes.append((exact_h, sig))
        return False, "none", 0.0

    def reset(self) -> None:
        """Clear registered deduplication hashes."""
        self._exact_hashes.clear()
        self._minhashes.clear()
