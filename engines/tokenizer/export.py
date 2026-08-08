"""
AFE — Tokenizer Exporter

Exports trained tokenizer instances to standard formats:
HuggingFace `tokenizer.json`, GGUF metadata schema, and raw vocab files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from engines.tokenizer.engine import BaseTokenizer


class TokenizerExporter:
    """Exports tokenizer parameters to standard open formats."""

    @staticmethod
    def to_huggingface_json(tokenizer: BaseTokenizer) -> str:
        """Produce a HuggingFace-compatible `tokenizer.json` representation."""
        vocab_dict = {str(k): int(v) for k, v in tokenizer.vocab.items()}
        merges_list = [" ".join(p) for p in tokenizer.merges]
        
        added_tokens = [
            {"id": int(tid), "content": token_str, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
            for tid, token_str in tokenizer.id_to_special.items()
        ]

        hf_data = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFKC"}]},
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
            "post_processor": None,
            "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": "<|unk|>",
                "continuing_subword_prefix": None,
                "end_of_word_suffix": "</w>",
                "fuse_unk": False,
                "byte_fallback": True,
                "vocab": vocab_dict,
                "merges": merges_list,
            },
        }
        return json.dumps(hf_data, indent=2)

    @staticmethod
    def to_gguf_metadata(tokenizer: BaseTokenizer) -> Dict[str, Any]:
        """Produce GGUF tokenizer metadata key-values for llama.cpp integration."""
        return {
            "tokenizer.ggml.model": "gpt2" if tokenizer.algorithm == "bpe" else "llama",
            "tokenizer.ggml.tokens": list(tokenizer.vocab.keys()),
            "tokenizer.ggml.scores": [1.0] * len(tokenizer.vocab),
            "tokenizer.ggml.token_type": [1] * len(tokenizer.vocab),
            "tokenizer.ggml.bos_token_id": tokenizer.special_tokens.get("<|bos|>", 2),
            "tokenizer.ggml.eos_token_id": tokenizer.special_tokens.get("<|eos|>", 3),
            "tokenizer.ggml.unknown_token_id": tokenizer.special_tokens.get("<|unk|>", 1),
            "tokenizer.ggml.padding_token_id": tokenizer.special_tokens.get("<|pad|>", 0),
        }
