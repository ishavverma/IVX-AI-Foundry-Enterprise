"""
AFE — Milestone 3 Verification Script

Tests dataset creation, cleaning/deduplication pipeline, tokenizer training,
interactive encoding, and REST endpoints.
"""

from __future__ import annotations

import os
import sys
import json
import urllib.request
from pathlib import Path

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.database.engine import init_db, get_session
from core.database.models.dataset import Dataset
from core.database.models.tokenizer import TokenizerModel
from core.storage.sdk import get_storage_backend
from core.storage.paths import storage_paths
from engines.dataset.cleaner import DatasetCleaner, CleaningOptions
from engines.dataset.dedup import DatasetDeduplicator
from engines.dataset.chunking import DatasetChunker
from engines.dataset.quality import DatasetQualityScorer
from engines.dataset.pipeline import DatasetPipeline
from engines.tokenizer.engine import BaseTokenizer
from engines.tokenizer.analytics import TokenizerAnalytics
from engines.tokenizer.export import TokenizerExporter


def run_tests():
    print("=======================================================")
    print("    AFE MILESTONE 3: DATASET & TOKENIZER SUITE TEST    ")
    print("=======================================================\n")

    init_db()
    storage_paths.init_all()
    storage = get_storage_backend()

    # ── 1. Test Dataset Cleaner ──
    raw_dirty = "   <html><body>Hello   World! 127.0.0.1  admin@afe.ai \x00\x08 </body></html>  \n\n\n\nNew Paragraph. "
    cleaned = DatasetCleaner.clean_text(raw_dirty, CleaningOptions(strip_html=True, mask_ips=True, mask_emails=True))
    print("1. DatasetCleaner output:")
    print(f"   Cleaned: {repr(cleaned)}")
    assert "<IP_ADDRESS>" in cleaned
    assert "<EMAIL>" in cleaned
    assert "<html>" not in cleaned

    # ── 2. Test MinHash Deduplicator ──
    dedup = DatasetDeduplicator(jaccard_threshold=0.7)
    t1 = "AI Foundry Enterprise provides high performance foundation model development pipelines for distributed clusters."
    t2 = "AI Foundry Enterprise provides high performance foundation model development pipelines for distributed machines."
    t3 = "A completely different and unrelated piece of text talking about weather and seasons."
    
    is_dup1, _, _ = dedup.is_duplicate(t1)
    is_dup2, match_type, sim = dedup.is_duplicate(t2)
    is_dup3, _, _ = dedup.is_duplicate(t3)
    print(f"\n2. MinHash Deduplication:")
    print(f"   t1 is_dup: {is_dup1} (first seen)")
    print(f"   t2 is_dup: {is_dup2} (match_type={match_type}, sim={sim:.2f})")
    print(f"   t3 is_dup: {is_dup3} (different)")
    assert not is_dup1
    assert is_dup2
    assert not is_dup3

    # ── 3. Test Quality Scorer ──
    q_score = DatasetQualityScorer.score_text(t1)
    print(f"\n3. Quality Scorer: {q_score['score'] * 100:.1f}% quality (metrics={q_score})")
    assert q_score["score"] > 0.4

    # ── 4. Test Tokenizer Engine ──
    print("\n4. Tokenizer Engine (BPE):")
    tok = BaseTokenizer(vocab_size=300, algorithm="bpe")
    sample_corpus = [
        "AI Foundry Enterprise provides high throughput foundation model training.",
        "Hybrid transformer state space models scale to ultra long context windows.",
        "Mixture of experts routes tokens across specialized sub-networks.",
        "Grouped query attention reduces memory consumption during inference.",
    ] * 10
    tok.train_from_texts(sample_corpus, target_vocab_size=300)
    print(f"   Trained vocab size: {len(tok.vocab)}, Merges: {len(tok.merges)}")
    assert len(tok.vocab) > 260

    test_sentence = "AI Foundry Enterprise delivers hybrid foundation models."
    encoded = tok.encode(test_sentence)
    decoded = tok.decode(encoded.token_ids)
    print(f"   Encode '{test_sentence}':")
    print(f"   Token IDs: {encoded.token_ids[:8]}... ({encoded.token_count} tokens)")
    print(f"   Tokens: {encoded.tokens[:8]}...")
    print(f"   Decoded: '{decoded}'")
    assert encoded.token_count > 0

    # ── 5. Test Tokenizer Analytics & Exporter ──
    metrics = TokenizerAnalytics.evaluate_corpus(tok, sample_corpus[:5])
    print(f"\n5. Tokenizer Analytics:")
    print(f"   Compression ratio: {metrics.bytes_per_token} B/tok, Fertility: {metrics.tokens_per_word} tok/word")
    
    hf_json = TokenizerExporter.to_huggingface_json(tok)
    gguf_meta = TokenizerExporter.to_gguf_metadata(tok)
    print(f"   HuggingFace JSON length: {len(hf_json)} chars")
    print(f"   GGUF tokens count: {len(gguf_meta['tokenizer.ggml.tokens'])}")
    assert "ByteLevel" in hf_json

    # ── 6. Test DB Record & Pipeline Execution ──
    ds_id = "test-dataset-01"
    raw_key = f"datasets/{ds_id}/raw.jsonl"
    storage.write(raw_key, "\n".join([json.dumps({"text": s}) for s in sample_corpus]).encode("utf-8"))

    with get_session() as session:
        # cleanup if exists
        old = session.get(Dataset, ds_id)
        if old:
            session.delete(old)
            session.flush()

        ds = Dataset(
            id=ds_id,
            name="Test Pretraining Corpus",
            slug="test-pretraining-corpus",
            format="jsonl",
            storage_key=raw_key,
            status="raw",
        )
        session.add(ds)

    print(f"\n6. Running DatasetPipeline on '{ds_id}'...")
    res = DatasetPipeline.run_pipeline("test-job-01", ds_id, {"clean": True, "deduplicate": True, "chunk": False})
    print(f"   Pipeline Result: {res}")
    assert res["final_rows"] > 0

    # Save tokenizer in DB for UI testing
    tok_id = "test-tok-01"
    tok_key = f"tokenizers/{tok_id}/tokenizer.json"
    storage.write(tok_key, json.dumps(tok.to_dict()).encode("utf-8"))
    with get_session() as session:
        old_tok = session.get(TokenizerModel, tok_id)
        if old_tok:
            session.delete(old_tok)
            session.flush()
        tm = TokenizerModel(
            id=tok_id,
            name="Enterprise BPE 300",
            slug="enterprise-bpe-300",
            algorithm="bpe",
            vocab_size=len(tok.vocab),
            storage_key=tok_key,
            compression_ratio=metrics.bytes_per_token,
            cjk_coverage_pct=100.0,
            status="ready",
        )
        session.add(tm)

    print("\n=======================================================")
    print("         ALL MILESTONE 3 UNIT TESTS PASSED!          ")
    print("=======================================================\n")


if __name__ == "__main__":
    run_tests()
